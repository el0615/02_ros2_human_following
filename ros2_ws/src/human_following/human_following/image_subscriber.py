import cv2
import rclpy

from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
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

        self.filtered_ratio = None
        self.previous_filtered_ratio = None
        self.previous_ratio_time = None

        self.forward_speed = 0.0

        self.get_logger().info(
            'Image subscriber, YOLO model, '
            'and cmd_vel publisher started'
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

                filter_alpha = 0.25

                if self.filtered_ratio is None:
                    self.filtered_ratio = box_height_ratio

                else:
                    self.filtered_ratio = (
                        filter_alpha
                        * box_height_ratio
                        + (
                            1.0 - filter_alpha
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
                    min(1.0, normalized_error_x)
                )

                max_angular_speed = 0.8
                max_linear_speed = 0.3

                target_ratio = 0.315

                target_ratio_low = 0.300
                target_ratio_high = 0.330

                stop_ratio = 0.70
                resume_ratio = 0.65

                kp = 2.0
                kd = 0.6

                if abs(error_x) <= dead_zone:
                    target_angular_z = 0.0
                    turn_factor = 1.0

                else:
                    target_angular_z = (
                        max_angular_speed
                        * normalized_error_x
                    )

                    turn_factor = (
                        1.0
                        - abs(normalized_error_x)
                    )

                if self.filtered_ratio < target_ratio_low:
                    ratio_error = (
                        target_ratio
                        - self.filtered_ratio
                    )

                elif self.filtered_ratio > target_ratio_high:
                    ratio_error = (
                        target_ratio
                        - self.filtered_ratio
                    )

                else:
                    ratio_error = 0.0

                if dt > 0.0:
                    speed_adjustment = (
                        kp * ratio_error
                        - kd * ratio_rate
                    )

                    self.forward_speed += (
                        speed_adjustment * dt
                    )

                self.forward_speed = max(
                    0.0,
                    min(
                        max_linear_speed,
                        self.forward_speed
                    )
                )

                if self.filtered_ratio <= target_ratio:
                    distance_speed_limit = (
                        max_linear_speed
                    )

                else:
                    distance_speed_limit = (
                        max_linear_speed
                        * (
                            stop_ratio
                            - self.filtered_ratio
                        )
                        / (
                            stop_ratio
                            - target_ratio
                        )
                    )

                    distance_speed_limit = max(
                        0.0,
                        min(
                            max_linear_speed,
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

                if self.stop_state:
                    if self.filtered_ratio <= resume_ratio:
                        self.stop_state = False

                else:
                    if self.filtered_ratio >= stop_ratio:
                        self.stop_state = True

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

                cv2.circle(
                    annotated_image,
                    (int(center_x), int(center_y)),
                    7,
                    (0, 0, 255),
                    -1
                )

                cv2.line(
                    annotated_image,
                    (int(image_center_x), 0),
                    (int(image_center_x), image_height),
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
                    (image_width - text_width) / 2
                )

                text_y = 60

                cv2.rectangle(
                    annotated_image,
                    (
                        text_x - 15,
                        text_y - text_height - 15
                    ),
                    (
                        text_x + text_width + 15,
                        text_y + baseline + 10
                    ),
                    (0, 0, 0),
                    -1
                )

                cv2.putText(
                    annotated_image,
                    text,
                    (text_x, text_y),
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
                        f'Filtered Ratio={self.filtered_ratio:.3f}, '
                        f'Target Ratio={target_ratio:.3f}, '
                        f'Ratio Error={ratio_error:.3f}, '
                        f'Ratio Rate={ratio_rate:.3f}, '
                        f'Speed Limit={distance_speed_limit:.3f}, '
                        f'State={state_text}, '
                        f'Direction={direction}, '
                        f'Linear X={twist.linear.x:.3f}, '
                        f'Angular Z={twist.angular.z:.3f}, '
                        f'Confidence={confidence:.2f}, '
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


def main(args=None):
    rclpy.init(args=args)

    node = ImageSubscriber()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()