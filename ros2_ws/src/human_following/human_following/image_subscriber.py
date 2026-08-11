import csv
import os
from datetime import datetime

import cv2
import rclpy

from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image
from ultralytics import YOLO


class ImageSubscriber(Node):

    def __init__(self):
        super().__init__('image_subscriber')

        self.bridge = CvBridge()

        self.model = YOLO(
            '/home/el0615/Projects/02_ros2_human_following/'
            'ros2_ws/yolov8n.pt'
        )

        self.subscription = self.create_subscription(
            Image,
            '/rgb',
            self.image_callback,
            10
        )

        self.cmd_vel_publisher = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        self.last_log_time = self.get_clock().now()

        self.stop_state = False
        self.stop_count = 0

        self.filtered_ratio = None
        self.previous_filtered_ratio = None
        self.previous_ratio_time = None

        self.forward_speed = 0.0

        self.filter_alpha = 0.25

        self.max_angular_speed = 0.8
        self.max_linear_speed = 0.9
        self.normal_max_speed = 0.7
        self.slowdown_ratio = 0.26

        self.target_ratio = 0.315
        self.target_ratio_low = 0.300
        self.target_ratio_high = 0.330

        self.stop_ratio = 0.70
        self.resume_ratio = 0.65

        self.kp = 2.0
        self.kd = 0.6

        self.test_records = []
        self.first_record_time = None
        self.evaluation_saved = False

        self.run_timestamp = datetime.now().strftime(
            '%Y%m%d_%H%M%S'
        )

        speed_tag = (
            f'{self.max_linear_speed:.2f}'
            .replace('.', 'p')
        )

        self.result_directory = os.path.expanduser(
            '~/Projects/02_ros2_human_following/'
            f'results/dn013/max_{speed_tag}/'
            f'run_{self.run_timestamp}'
        )

        os.makedirs(
            self.result_directory,
            exist_ok=True
        )

        self.get_logger().info(
            'Image subscriber, YOLO model, '
            'cmd_vel publisher, and DN-013 evaluator started'
        )

        self.get_logger().info(
            f'Evaluation results will be saved to: '
            f'{self.result_directory}'
        )

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='bgr8'
            )

            results = self.model(
                cv_image,
                classes=[0],
                verbose=False
            )

            annotated_image = results[0].plot()
            boxes = results[0].boxes

            if len(boxes) > 0:
                box = boxes[0]

                x1, y1, x2, y2 = map(
                    float,
                    box.xyxy[0].tolist()
                )

                confidence = float(box.conf[0])
                class_id = int(box.cls[0])

                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                image_height, image_width = cv_image.shape[:2]

                box_height = y2 - y1

                box_height_ratio = (
                    box_height / image_height
                )

                current_time = self.get_clock().now()

                if self.filtered_ratio is None:
                    self.filtered_ratio = box_height_ratio

                else:
                    self.filtered_ratio = (
                        self.filter_alpha
                        * box_height_ratio
                        + (
                            1.0 - self.filter_alpha
                        )
                        * self.filtered_ratio
                    )

                if (
                    self.previous_filtered_ratio is None
                    or self.previous_ratio_time is None
                ):
                    dt = 0.0
                    ratio_rate = 0.0

                else:
                    dt = (
                        current_time
                        - self.previous_ratio_time
                    ).nanoseconds / 1_000_000_000.0

                    if dt > 0.001:
                        ratio_rate = (
                            self.filtered_ratio
                            - self.previous_filtered_ratio
                        ) / dt

                    else:
                        ratio_rate = 0.0

                self.previous_filtered_ratio = (
                    self.filtered_ratio
                )

                self.previous_ratio_time = current_time

                image_center_x = image_width / 2

                error_x = center_x - image_center_x
                dead_zone = 80

                if error_x < -dead_zone:
                    direction = 'LEFT'

                elif error_x > dead_zone:
                    direction = 'RIGHT'

                else:
                    direction = 'CENTER'

                normalized_error_x = (
                    error_x / image_center_x
                )

                normalized_error_x = max(
                    -1.0,
                    min(
                        1.0,
                        normalized_error_x
                    )
                )

                if abs(error_x) <= dead_zone:
                    target_angular_z = 0.0
                    turn_factor = 1.0

                else:
                    target_angular_z = (
                        self.max_angular_speed
                        * normalized_error_x
                    )

                    turn_factor = (
                        1.0
                        - abs(normalized_error_x)
                    )

                if (
                    self.filtered_ratio
                    < self.target_ratio_low
                ):
                    ratio_error = (
                        self.target_ratio
                        - self.filtered_ratio
                    )

                elif (
                    self.filtered_ratio
                    > self.target_ratio_high
                ):
                    ratio_error = (
                        self.target_ratio
                        - self.filtered_ratio
                    )

                else:
                    ratio_error = 0.0

                if dt > 0.0:
                    speed_adjustment = (
                        self.kp * ratio_error
                        - self.kd * ratio_rate
                    )

                    self.forward_speed += (
                        speed_adjustment * dt
                    )

                self.forward_speed = max(
                    0.0,
                    min(
                        self.max_linear_speed,
                        self.forward_speed
                    )
                )

                if (
                    self.filtered_ratio
                    < self.slowdown_ratio
                ):
                    distance_speed_limit = (
                        self.max_linear_speed
                    )

                elif (
                    self.filtered_ratio
                    < self.target_ratio_low
                ):
                    slowdown_progress = (
                        self.filtered_ratio
                        - self.slowdown_ratio
                    ) / (
                        self.target_ratio_low
                        - self.slowdown_ratio
                    )

                    distance_speed_limit = (
                        self.max_linear_speed
                        - slowdown_progress
                        * (
                            self.max_linear_speed
                            - self.normal_max_speed
                        )
                    )

                elif (
                    self.filtered_ratio
                    <= self.target_ratio_high
                ):
                    distance_speed_limit = (
                        self.normal_max_speed
                    )

                else:
                    distance_speed_limit = (
                        self.normal_max_speed
                        * (
                            self.stop_ratio
                            - self.filtered_ratio
                        )
                        / (
                            self.stop_ratio
                            - self.target_ratio_high
                        )
                    )

                    distance_speed_limit = max(
                        0.0,
                        min(
                            self.normal_max_speed,
                            distance_speed_limit
                        )
                    )

                self.forward_speed = min(
                    self.forward_speed,
                    distance_speed_limit
                )

                target_linear_x = (
                    self.forward_speed
                    * turn_factor
                )

                previous_stop_state = self.stop_state

                if self.stop_state:
                    if (
                        self.filtered_ratio
                        <= self.resume_ratio
                    ):
                        self.stop_state = False

                else:
                    if (
                        self.filtered_ratio
                        >= self.stop_ratio
                    ):
                        self.stop_state = True

                if (
                    not previous_stop_state
                    and self.stop_state
                ):
                    self.stop_count += 1

                if self.stop_state:
                    self.forward_speed = 0.0
                    target_linear_x = 0.0
                    state_text = 'STOP'

                else:
                    state_text = 'FOLLOW'

                twist = Twist()

                twist.linear.x = target_linear_x
                twist.linear.y = 0.0
                twist.linear.z = 0.0

                twist.angular.x = 0.0
                twist.angular.y = 0.0
                twist.angular.z = target_angular_z

                self.cmd_vel_publisher.publish(twist)

                if self.first_record_time is None:
                    self.first_record_time = current_time
                    elapsed_time = 0.0

                else:
                    elapsed_time = (
                        current_time
                        - self.first_record_time
                    ).nanoseconds / 1_000_000_000.0

                self.test_records.append(
                    {
                        'time_s': elapsed_time,
                        'raw_ratio': box_height_ratio,
                        'filtered_ratio': self.filtered_ratio,
                        'ratio_error': ratio_error,
                        'ratio_rate': ratio_rate,
                        'linear_x': twist.linear.x,
                        'angular_z': twist.angular.z,
                        'speed_limit': distance_speed_limit,
                        'direction': direction,
                        'state': state_text,
                        'confidence': confidence
                    }
                )

                cv2.circle(
                    annotated_image,
                    (
                        int(center_x),
                        int(center_y)
                    ),
                    7,
                    (0, 0, 255),
                    -1
                )

                cv2.line(
                    annotated_image,
                    (
                        int(image_center_x),
                        0
                    ),
                    (
                        int(image_center_x),
                        image_height
                    ),
                    (255, 0, 0),
                    2
                )

                text = (
                    f'Direction: {direction} | '
                    f'{state_text}'
                )

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.3
                thickness = 3

                (
                    text_width,
                    text_height
                ), baseline = cv2.getTextSize(
                    text,
                    font,
                    font_scale,
                    thickness
                )

                text_x = int(
                    (
                        image_width
                        - text_width
                    ) / 2
                )

                text_y = 60

                cv2.rectangle(
                    annotated_image,
                    (
                        text_x - 15,
                        text_y
                        - text_height
                        - 15
                    ),
                    (
                        text_x
                        + text_width
                        + 15,
                        text_y
                        + baseline
                        + 10
                    ),
                    (0, 0, 0),
                    -1
                )

                cv2.putText(
                    annotated_image,
                    text,
                    (
                        text_x,
                        text_y
                    ),
                    font,
                    font_scale,
                    (0, 255, 255),
                    thickness
                )

                if (
                    current_time
                    - self.last_log_time
                ).nanoseconds >= 1_000_000_000:

                    self.get_logger().info(
                        f'Raw Ratio={box_height_ratio:.3f}, '
                        f'Filtered Ratio='
                        f'{self.filtered_ratio:.3f}, '
                        f'Target Ratio='
                        f'{self.target_ratio:.3f}, '
                        f'Ratio Error='
                        f'{ratio_error:.3f}, '
                        f'Ratio Rate='
                        f'{ratio_rate:.3f}, '
                        f'Speed Limit='
                        f'{distance_speed_limit:.3f}, '
                        f'State={state_text}, '
                        f'Direction={direction}, '
                        f'Linear X='
                        f'{twist.linear.x:.3f}, '
                        f'Angular Z='
                        f'{twist.angular.z:.3f}, '
                        f'Confidence='
                        f'{confidence:.2f}, '
                        f'Class={class_id}'
                    )

                    self.last_log_time = current_time

            cv2.imshow(
                'YOLO Person Tracking',
                annotated_image
            )

            cv2.waitKey(1)

        except Exception as error:
            self.get_logger().error(
                f'Image processing failed: {error}'
            )

    def save_evaluation(self):

        if self.evaluation_saved:
            return

        self.evaluation_saved = True

        if len(self.test_records) == 0:
            print()
            print(
                'No DN-013 evaluation data recorded.'
            )
            print()
            return

        csv_path = os.path.join(
            self.result_directory,
            'data.csv'
        )

        summary_path = os.path.join(
            self.result_directory,
            'summary.txt'
        )

        ratio_graph_path = os.path.join(
            self.result_directory,
            'ratio.png'
        )

        speed_graph_path = os.path.join(
            self.result_directory,
            'speed.png'
        )

        ratio_rate_graph_path = os.path.join(
            self.result_directory,
            'ratio_rate.png'
        )

        fieldnames = [
            'time_s',
            'raw_ratio',
            'filtered_ratio',
            'ratio_error',
            'ratio_rate',
            'linear_x',
            'angular_z',
            'speed_limit',
            'direction',
            'state',
            'confidence'
        ]

        with open(
            csv_path,
            'w',
            newline='',
            encoding='utf-8'
        ) as csv_file:

            writer = csv.DictWriter(
                csv_file,
                fieldnames=fieldnames
            )

            writer.writeheader()

            writer.writerows(
                self.test_records
            )

        filtered_ratios = [
            record['filtered_ratio']
            for record in self.test_records
        ]

        linear_speeds = [
            record['linear_x']
            for record in self.test_records
        ]

        absolute_errors = [
            abs(
                self.target_ratio
                - ratio
            )
            for ratio in filtered_ratios
        ]

        mae = (
            sum(absolute_errors)
            / len(absolute_errors)
        )

        target_band_count = sum(
            1
            for ratio in filtered_ratios
            if (
                self.target_ratio_low
                <= ratio
                <= self.target_ratio_high
            )
        )

        target_band_percentage = (
            target_band_count
            / len(filtered_ratios)
            * 100.0
        )

        minimum_ratio = min(
            filtered_ratios
        )

        maximum_ratio = max(
            filtered_ratios
        )

        average_linear_speed = (
            sum(linear_speeds)
            / len(linear_speeds)
        )

        maximum_linear_speed_recorded = max(
            linear_speeds
        )

        detected_duration = (
            self.test_records[-1]['time_s']
        )

        summary_lines = [
            '===== DN-013 Evaluation =====',
            (
                f'Detected Duration      : '
                f'{detected_duration:.2f} s'
            ),
            (
                f'Detected Samples       : '
                f'{len(self.test_records)}'
            ),
            (
                f'Catch-up Max Speed     : '
                f'{self.max_linear_speed:.3f} m/s'
            ),
            (
                f'Normal Max Speed       : '
                f'{self.normal_max_speed:.3f} m/s'
            ),
            (
                f'Slowdown Ratio         : '
                f'{self.slowdown_ratio:.3f}'
            ),
            (
                f'Target Ratio           : '
                f'{self.target_ratio:.3f}'
            ),
            (
                f'Target Band            : '
                f'{self.target_ratio_low:.3f} '
                f'~ '
                f'{self.target_ratio_high:.3f}'
            ),
            (
                f'Mean Absolute Error    : '
                f'{mae:.4f}'
            ),
            (
                f'Target Band Samples    : '
                f'{target_band_percentage:.2f} %'
            ),
            (
                f'Minimum Ratio          : '
                f'{minimum_ratio:.3f}'
            ),
            (
                f'Maximum Ratio          : '
                f'{maximum_ratio:.3f}'
            ),
            (
                f'Average Linear Speed   : '
                f'{average_linear_speed:.3f} m/s'
            ),
            (
                f'Maximum Linear Speed   : '
                f'{maximum_linear_speed_recorded:.3f} m/s'
            ),
            (
                f'STOP Count             : '
                f'{self.stop_count}'
            ),
            (
                'Evaluation Scope        : '
                'person-detected frames only'
            ),
            '============================='
        ]

        summary_text = '\n'.join(
            summary_lines
        )

        print()
        print(summary_text)
        print()

        with open(
            summary_path,
            'w',
            encoding='utf-8'
        ) as summary_file:
            summary_file.write(
                summary_text
            )

        try:
            import matplotlib

            matplotlib.use('Agg')

            import matplotlib.pyplot as plt

            times = [
                record['time_s']
                for record in self.test_records
            ]

            raw_ratios = [
                record['raw_ratio']
                for record in self.test_records
            ]

            ratio_rates = [
                record['ratio_rate']
                for record in self.test_records
            ]

            speed_limits = [
                record['speed_limit']
                for record in self.test_records
            ]

            plt.figure(
                figsize=(11, 6)
            )

            plt.plot(
                times,
                raw_ratios,
                label='Raw Ratio',
                alpha=0.35
            )

            plt.plot(
                times,
                filtered_ratios,
                label='Filtered Ratio',
                linewidth=2
            )

            plt.axhline(
                self.target_ratio,
                linestyle='--',
                label='Target Ratio 0.315'
            )

            plt.axhspan(
                self.target_ratio_low,
                self.target_ratio_high,
                alpha=0.15,
                label='Target Band'
            )

            plt.axhline(
                self.slowdown_ratio,
                linestyle='-.',
                label=(
                    f'Slowdown Ratio '
                    f'{self.slowdown_ratio:.2f}'
                )
            )

            plt.axhline(
                self.stop_ratio,
                linestyle=':',
                label='STOP Ratio 0.70'
            )

            plt.xlabel(
                'Time (s)'
            )

            plt.ylabel(
                'Bounding Box Height Ratio'
            )

            plt.title(
                'DN-013 Target Ratio Tracking'
            )

            plt.legend()

            plt.grid(
                True,
                alpha=0.3
            )

            plt.tight_layout()

            plt.savefig(
                ratio_graph_path,
                dpi=200
            )

            plt.close()

            plt.figure(
                figsize=(11, 6)
            )

            plt.plot(
                times,
                linear_speeds,
                label='Linear X',
                linewidth=2
            )

            plt.plot(
                times,
                speed_limits,
                label='Distance Speed Limit',
                linestyle='--'
            )

            plt.axhline(
                self.max_linear_speed,
                linestyle=':',
                label=(
                    f'Max Speed '
                    f'{self.max_linear_speed:.2f}'
                )
            )

            plt.xlabel(
                'Time (s)'
            )

            plt.ylabel(
                'Linear Speed (m/s)'
            )

            plt.title(
                'DN-013 Linear Speed Response'
            )

            plt.legend()

            plt.grid(
                True,
                alpha=0.3
            )

            plt.tight_layout()

            plt.savefig(
                speed_graph_path,
                dpi=200
            )

            plt.close()

            plt.figure(
                figsize=(11, 6)
            )

            plt.plot(
                times,
                ratio_rates,
                label='Ratio Rate',
                linewidth=1.5
            )

            plt.axhline(
                0.0,
                linestyle='--',
                label='Zero Relative Change'
            )

            plt.xlabel(
                'Time (s)'
            )

            plt.ylabel(
                'Ratio Rate (1/s)'
            )

            plt.title(
                'DN-013 Bounding Box Ratio Rate'
            )

            plt.legend()

            plt.grid(
                True,
                alpha=0.3
            )

            plt.tight_layout()

            plt.savefig(
                ratio_rate_graph_path,
                dpi=200
            )

            plt.close()

            print(
                'DN-013 result files saved:'
            )

            print(
                f'Run Directory : '
                f'{self.result_directory}'
            )

            print(
                f'CSV           : '
                f'{csv_path}'
            )

            print(
                f'Summary       : '
                f'{summary_path}'
            )

            print(
                f'Ratio Graph   : '
                f'{ratio_graph_path}'
            )

            print(
                f'Speed Graph   : '
                f'{speed_graph_path}'
            )

            print(
                f'Ratio Rate    : '
                f'{ratio_rate_graph_path}'
            )

        except ImportError:
            print(
                'matplotlib is not installed.'
            )

            print(
                'CSV and summary were saved, '
                'but graphs were not generated.'
            )

        except Exception as error:
            print(
                f'Graph generation failed: {error}'
            )


def main(args=None):
    rclpy.init(args=args)

    node = ImageSubscriber()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    except ExternalShutdownException:
        pass

    finally:
        try:
            node.save_evaluation()

        except Exception as error:
            print(
                f'Evaluation save failed: {error}'
            )

        cv2.destroyAllWindows()

        try:
            node.destroy_node()

        except Exception:
            pass

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()