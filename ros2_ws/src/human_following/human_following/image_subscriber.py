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

        self.max_angular_speed = 1.0
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

        self.center_dead_zone_px = 20
        self.soft_zone_limit_px = 40

        self.turn_switch_epsilon = 0.01

        self.no_detection_timeout = 0.3
        self.search_delay = 3.0
        self.search_angular_speed = 1.0
        self.no_detection_start_time = None
        self.last_valid_twist = Twist()
        self.last_turn_direction = 0.0
        self.loss_state = 'NO_TARGET'

        self.test_records = []
        self.first_record_time = None
        self.evaluation_saved = False

        self.run_timestamp = datetime.now().strftime(
            '%Y%m%d_%H%M%S'
        )

        angular_tag = (
            f'{self.max_angular_speed:.2f}'
            .replace('.', 'p')
        )

        self.result_directory = os.path.expanduser(
            '~/Projects/02_ros2_human_following/'
            'results/dn014/tracking_eval/'
            f'soft_{self.center_dead_zone_px:03d}_'
            f'{self.soft_zone_limit_px:03d}_'
            f'linear_ang_{angular_tag}/'
            f'run_{self.run_timestamp}'
        )

        os.makedirs(
            self.result_directory,
            exist_ok=True
        )

        self.get_logger().info(
            'Image subscriber, YOLO model, '
            'cmd_vel publisher, and DN-014 '
            'Soft Zone evaluator started'
        )

        self.get_logger().info(
            f'Center Dead Zone: '
            f'{self.center_dead_zone_px} px'
        )

        self.get_logger().info(
            f'Soft Zone Limit: '
            f'{self.soft_zone_limit_px} px'
        )

        self.get_logger().info(
            f'Max Angular Speed: '
            f'{self.max_angular_speed:.3f} rad/s'
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

            current_time = self.get_clock().now()
            image_height, image_width = cv_image.shape[:2]

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

                box_height = y2 - y1

                box_height_ratio = (
                    box_height / image_height
                )

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

                error_x = (
                    center_x
                    - image_center_x
                )

                abs_error_x = abs(
                    error_x
                )

                normalized_error_x = (
                    error_x
                    / image_center_x
                )

                normalized_error_x = max(
                    -1.0,
                    min(
                        1.0,
                        normalized_error_x
                    )
                )

                base_angular_z = (
                    self.max_angular_speed
                    * normalized_error_x
                )

                if (
                    abs_error_x
                    <= self.center_dead_zone_px
                ):
                    direction = 'CENTER'
                    turn_mode = 'CENTER'
                    soft_gain = 0.0
                    target_angular_z = 0.0
                    turn_factor = 1.0

                elif (
                    abs_error_x
                    <= self.soft_zone_limit_px
                ):
                    if error_x < 0:
                        direction = 'LEFT'

                    else:
                        direction = 'RIGHT'

                    turn_mode = 'SOFT'

                    soft_gain = (
                        abs_error_x
                        - self.center_dead_zone_px
                    ) / (
                        self.soft_zone_limit_px
                        - self.center_dead_zone_px
                    )

                    soft_gain = max(
                        0.0,
                        min(
                            1.0,
                            soft_gain
                        )
                    )

                    target_angular_z = (
                        base_angular_z
                        * soft_gain
                    )

                    turn_factor = (
                        1.0
                        - abs(normalized_error_x)
                    )

                else:
                    if error_x < 0:
                        direction = 'LEFT'

                    else:
                        direction = 'RIGHT'

                    turn_mode = 'FULL'
                    soft_gain = 1.0

                    target_angular_z = (
                        base_angular_z
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

                self.cmd_vel_publisher.publish(
                    twist
                )

                self.no_detection_start_time = None
                self.loss_state = 'TRACKING'

                self.last_valid_twist = Twist()
                self.last_valid_twist.linear.x = twist.linear.x
                self.last_valid_twist.linear.y = twist.linear.y
                self.last_valid_twist.linear.z = twist.linear.z
                self.last_valid_twist.angular.x = twist.angular.x
                self.last_valid_twist.angular.y = twist.angular.y
                self.last_valid_twist.angular.z = twist.angular.z

                if twist.angular.z > self.turn_switch_epsilon:
                    self.last_turn_direction = 1.0
                elif twist.angular.z < -self.turn_switch_epsilon:
                    self.last_turn_direction = -1.0

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
                        'x_error_px': error_x,
                        'abs_x_error_px': abs_error_x,
                        'normalized_error_x': normalized_error_x,
                        'center_dead_zone_px':
                            self.center_dead_zone_px,
                        'soft_zone_limit_px':
                            self.soft_zone_limit_px,
                        'soft_gain': soft_gain,
                        'turn_mode': turn_mode,
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

                dead_left_x = int(
                    image_center_x
                    - self.center_dead_zone_px
                )

                dead_right_x = int(
                    image_center_x
                    + self.center_dead_zone_px
                )

                soft_left_x = int(
                    image_center_x
                    - self.soft_zone_limit_px
                )

                soft_right_x = int(
                    image_center_x
                    + self.soft_zone_limit_px
                )

                cv2.line(
                    annotated_image,
                    (
                        dead_left_x,
                        0
                    ),
                    (
                        dead_left_x,
                        image_height
                    ),
                    (0, 255, 0),
                    2
                )

                cv2.line(
                    annotated_image,
                    (
                        dead_right_x,
                        0
                    ),
                    (
                        dead_right_x,
                        image_height
                    ),
                    (0, 255, 0),
                    2
                )

                cv2.line(
                    annotated_image,
                    (
                        soft_left_x,
                        0
                    ),
                    (
                        soft_left_x,
                        image_height
                    ),
                    (0, 255, 255),
                    1
                )

                cv2.line(
                    annotated_image,
                    (
                        soft_right_x,
                        0
                    ),
                    (
                        soft_right_x,
                        image_height
                    ),
                    (0, 255, 255),
                    1
                )

                text = (
                    f'Direction: {direction} | '
                    f'{state_text} | '
                    f'{turn_mode} | '
                    f'X Error: {error_x:.0f}px'
                )

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.0
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
                        f'Raw Ratio='
                        f'{box_height_ratio:.3f}, '
                        f'Filtered Ratio='
                        f'{self.filtered_ratio:.3f}, '
                        f'Target Ratio='
                        f'{self.target_ratio:.3f}, '
                        f'Ratio Error='
                        f'{ratio_error:.3f}, '
                        f'Ratio Rate='
                        f'{ratio_rate:.3f}, '
                        f'X Error='
                        f'{error_x:.1f}px, '
                        f'Turn Mode='
                        f'{turn_mode}, '
                        f'Soft Gain='
                        f'{soft_gain:.3f}, '
                        f'Speed Limit='
                        f'{distance_speed_limit:.3f}, '
                        f'State='
                        f'{state_text}, '
                        f'Direction='
                        f'{direction}, '
                        f'Linear X='
                        f'{twist.linear.x:.3f}, '
                        f'Angular Z='
                        f'{twist.angular.z:.3f}, '
                        f'Confidence='
                        f'{confidence:.2f}, '
                        f'Class='
                        f'{class_id}'
                    )

                    self.last_log_time = current_time

            else:
                if self.no_detection_start_time is None:
                    self.no_detection_start_time = current_time

                lost_duration = (
                    current_time
                    - self.no_detection_start_time
                ).nanoseconds / 1_000_000_000.0

                if lost_duration <= self.no_detection_timeout:
                    self.loss_state = 'TEMP_LOST'

                    twist = Twist()
                    twist.linear.x = self.last_valid_twist.linear.x
                    twist.linear.y = self.last_valid_twist.linear.y
                    twist.linear.z = self.last_valid_twist.linear.z
                    twist.angular.x = self.last_valid_twist.angular.x
                    twist.angular.y = self.last_valid_twist.angular.y
                    twist.angular.z = self.last_valid_twist.angular.z

                else:
                    stop_duration = (
                        lost_duration
                        - self.no_detection_timeout
                    )

                    self.forward_speed = 0.0
                    self.filtered_ratio = None
                    self.previous_filtered_ratio = None
                    self.previous_ratio_time = None
                    self.stop_state = False

                    twist = Twist()

                    if stop_duration <= self.search_delay:
                        self.loss_state = 'LOST_STOP'

                    else:
                        self.loss_state = 'SEARCHING'
                        twist.linear.x = 0.0
                        twist.angular.z = (
                            self.search_angular_speed
                            * self.last_turn_direction
                        )

                self.cmd_vel_publisher.publish(twist)

                text = (
                    f'NO PERSON | {self.loss_state} | '
                    f'Lost: {lost_duration:.2f}s'
                )

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.0
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
                        f'No person detected | '
                        f'State={self.loss_state}, '
                        f'Lost Duration={lost_duration:.2f}s, '
                        f'Linear X={twist.linear.x:.3f}, '
                        f'Angular Z={twist.angular.z:.3f}'
                    )

                    self.last_log_time = current_time

            cv2.imshow(
                'YOLO Person Tracking',
                annotated_image
            )

            cv2.waitKey(1)

        except Exception as error:
            self.get_logger().error(
                f'Image processing failed: '
                f'{error}'
            )

    def save_evaluation(self):

        if self.evaluation_saved:
            return

        self.evaluation_saved = True

        if len(self.test_records) == 0:
            print()
            print(
                'No DN-014 evaluation data recorded.'
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

        x_error_graph_path = os.path.join(
            self.result_directory,
            'x_error.png'
        )

        angular_graph_path = os.path.join(
            self.result_directory,
            'angular_response.png'
        )

        fieldnames = [
            'time_s',
            'raw_ratio',
            'filtered_ratio',
            'ratio_error',
            'ratio_rate',
            'x_error_px',
            'abs_x_error_px',
            'normalized_error_x',
            'center_dead_zone_px',
            'soft_zone_limit_px',
            'soft_gain',
            'turn_mode',
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

        angular_speeds = [
            record['angular_z']
            for record in self.test_records
        ]

        x_errors = [
            record['x_error_px']
            for record in self.test_records
        ]

        abs_x_errors = [
            record['abs_x_error_px']
            for record in self.test_records
        ]

        normalized_x_errors = [
            record['normalized_error_x']
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

        mean_abs_x_error = (
            sum(abs_x_errors)
            / len(abs_x_errors)
        )

        max_abs_x_error = max(
            abs_x_errors
        )

        mean_abs_normalized_x_error = (
            sum(
                abs(error)
                for error in normalized_x_errors
            )
            / len(normalized_x_errors)
        )

        center_20_count = sum(
            1
            for error in abs_x_errors
            if error <= 20
        )

        center_20_percentage = (
            center_20_count
            / len(abs_x_errors)
            * 100.0
        )

        center_40_count = sum(
            1
            for error in abs_x_errors
            if error <= 40
        )

        center_40_percentage = (
            center_40_count
            / len(abs_x_errors)
            * 100.0
        )

        center_80_count = sum(
            1
            for error in abs_x_errors
            if error <= 80
        )

        center_80_percentage = (
            center_80_count
            / len(abs_x_errors)
            * 100.0
        )

        center_mode_count = sum(
            1
            for record in self.test_records
            if record['turn_mode'] == 'CENTER'
        )

        soft_mode_count = sum(
            1
            for record in self.test_records
            if record['turn_mode'] == 'SOFT'
        )

        full_mode_count = sum(
            1
            for record in self.test_records
            if record['turn_mode'] == 'FULL'
        )

        center_mode_percentage = (
            center_mode_count
            / len(self.test_records)
            * 100.0
        )

        soft_mode_percentage = (
            soft_mode_count
            / len(self.test_records)
            * 100.0
        )

        full_mode_percentage = (
            full_mode_count
            / len(self.test_records)
            * 100.0
        )

        mean_abs_angular_z = (
            sum(
                abs(speed)
                for speed in angular_speeds
            )
            / len(angular_speeds)
        )

        max_abs_angular_z = max(
            abs(speed)
            for speed in angular_speeds
        )

        angular_change_rates = []

        for index in range(
            1,
            len(self.test_records)
        ):
            current_record = (
                self.test_records[index]
            )

            previous_record = (
                self.test_records[index - 1]
            )

            angular_dt = (
                current_record['time_s']
                - previous_record['time_s']
            )

            if angular_dt > 0.001:
                angular_change_rate = abs(
                    (
                        current_record['angular_z']
                        - previous_record['angular_z']
                    )
                    / angular_dt
                )

                angular_change_rates.append(
                    angular_change_rate
                )

        if len(angular_change_rates) > 0:
            mean_angular_change_rate = (
                sum(angular_change_rates)
                / len(angular_change_rates)
            )

            max_angular_change_rate = max(
                angular_change_rates
            )

        else:
            mean_angular_change_rate = 0.0
            max_angular_change_rate = 0.0

        turn_switch_count = 0
        previous_turn_sign = 0

        for angular_speed in angular_speeds:

            if (
                angular_speed
                > self.turn_switch_epsilon
            ):
                current_turn_sign = 1

            elif (
                angular_speed
                < -self.turn_switch_epsilon
            ):
                current_turn_sign = -1

            else:
                current_turn_sign = 0

            if current_turn_sign != 0:

                if (
                    previous_turn_sign != 0
                    and current_turn_sign
                    != previous_turn_sign
                ):
                    turn_switch_count += 1

                previous_turn_sign = (
                    current_turn_sign
                )

        detected_duration = (
            self.test_records[-1]['time_s']
        )

        if detected_duration > 0.0:
            turn_switch_rate_per_min = (
                turn_switch_count
                / detected_duration
                * 60.0
            )

        else:
            turn_switch_rate_per_min = 0.0

        summary_lines = [
            '===== DN-014 Evaluation =====',
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
                f'Max Angular Speed      : '
                f'{self.max_angular_speed:.3f} rad/s'
            ),
            (
                f'Center Dead Zone       : '
                f'{self.center_dead_zone_px} px'
            ),
            (
                f'Soft Zone Limit        : '
                f'{self.soft_zone_limit_px} px'
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
            '----- Horizontal Tracking -----',
            (
                f'Mean Abs X Error       : '
                f'{mean_abs_x_error:.2f} px'
            ),
            (
                f'Max Abs X Error        : '
                f'{max_abs_x_error:.2f} px'
            ),
            (
                f'Mean Abs X Error Norm  : '
                f'{mean_abs_normalized_x_error:.4f}'
            ),
            (
                f'Center +/-20 px        : '
                f'{center_20_percentage:.2f} %'
            ),
            (
                f'Center +/-40 px        : '
                f'{center_40_percentage:.2f} %'
            ),
            (
                f'Center +/-80 px        : '
                f'{center_80_percentage:.2f} %'
            ),
            (
                f'CENTER Mode Samples    : '
                f'{center_mode_percentage:.2f} %'
            ),
            (
                f'SOFT Mode Samples      : '
                f'{soft_mode_percentage:.2f} %'
            ),
            (
                f'FULL Mode Samples      : '
                f'{full_mode_percentage:.2f} %'
            ),
            (
                f'Mean Abs Angular Z     : '
                f'{mean_abs_angular_z:.4f} rad/s'
            ),
            (
                f'Max Abs Angular Z      : '
                f'{max_abs_angular_z:.4f} rad/s'
            ),
            (
                f'Mean Angular Change    : '
                f'{mean_angular_change_rate:.4f} rad/s^2'
            ),
            (
                f'Max Angular Change     : '
                f'{max_angular_change_rate:.4f} rad/s^2'
            ),
            (
                f'Turn Direction Switch  : '
                f'{turn_switch_count}'
            ),
            (
                f'Turn Switch Rate       : '
                f'{turn_switch_rate_per_min:.2f} /min'
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
                'DN-014 Target Ratio Tracking'
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
                'DN-014 Linear Speed Response'
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
                'DN-014 Bounding Box Ratio Rate'
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

            plt.figure(
                figsize=(11, 6)
            )

            plt.plot(
                times,
                x_errors,
                label='X Error',
                linewidth=1.5
            )

            plt.axhline(
                0.0,
                linestyle='--',
                label='Image Center'
            )

            plt.axhspan(
                -self.center_dead_zone_px,
                self.center_dead_zone_px,
                alpha=0.15,
                label=(
                    f'Dead Zone '
                    f'+/-{self.center_dead_zone_px}px'
                )
            )

            plt.axhline(
                self.soft_zone_limit_px,
                linestyle=':',
                label=(
                    f'Soft Zone '
                    f'+/-{self.soft_zone_limit_px}px'
                )
            )

            plt.axhline(
                -self.soft_zone_limit_px,
                linestyle=':'
            )

            plt.xlabel(
                'Time (s)'
            )

            plt.ylabel(
                'Horizontal Error (px)'
            )

            plt.title(
                'DN-014 Horizontal Tracking Error'
            )

            plt.legend()

            plt.grid(
                True,
                alpha=0.3
            )

            plt.tight_layout()

            plt.savefig(
                x_error_graph_path,
                dpi=200
            )

            plt.close()

            plt.figure(
                figsize=(11, 6)
            )

            plt.plot(
                times,
                angular_speeds,
                label='Angular Z',
                linewidth=1.5
            )

            plt.axhline(
                0.0,
                linestyle='--',
                label='Zero Angular Speed'
            )

            plt.xlabel(
                'Time (s)'
            )

            plt.ylabel(
                'Angular Speed (rad/s)'
            )

            plt.title(
                'DN-014 Angular Control Response'
            )

            plt.legend()

            plt.grid(
                True,
                alpha=0.3
            )

            plt.tight_layout()

            plt.savefig(
                angular_graph_path,
                dpi=200
            )

            plt.close()

            print(
                'DN-014 result files saved:'
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

            print(
                f'X Error Graph : '
                f'{x_error_graph_path}'
            )

            print(
                f'Angular Graph : '
                f'{angular_graph_path}'
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
                f'Graph generation failed: '
                f'{error}'
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
                f'Evaluation save failed: '
                f'{error}'
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