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
                base_linear_speed = 0.3

                if abs(error_x) <= dead_zone:
                    target_angular_z = 0.0
                    target_linear_x = base_linear_speed
                else:
                    target_angular_z = (
                        max_angular_speed
                        * normalized_error_x
                    )

                    target_linear_x = (
                        base_linear_speed
                        * (
                            1.0
                            - abs(normalized_error_x)
                        )
                    )

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

                text = f'Direction: {direction}'

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

                self.get_logger().info(
                    f'Bounding Box: '
                    f'x1={x1:.1f}, y1={y1:.1f}, '
                    f'x2={x2:.1f}, y2={y2:.1f}'
                )

                self.get_logger().info(
                    f'Center: '
                    f'x={center_x:.1f}, '
                    f'y={center_y:.1f}'
                )

                self.get_logger().info(
                    f'Image Center X={image_center_x:.1f}, '
                    f'Error X={error_x:.1f}, '
                    f'Direction={direction}'
                )

                self.get_logger().info(
                    f'Normalized Error X='
                    f'{normalized_error_x:.3f}, '
                    f'Target Linear X='
                    f'{target_linear_x:.3f}, '
                    f'Target Angular Z='
                    f'{target_angular_z:.3f}'
                )

                self.get_logger().info(
                    f'Published cmd_vel: '
                    f'linear.x={twist.linear.x:.3f}, '
                    f'angular.z={twist.angular.z:.3f}'
                )

                self.get_logger().info(
                    f'Confidence={confidence:.2f}, '
                    f'Class={class_id}'
                )

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