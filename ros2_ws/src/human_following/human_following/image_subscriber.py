import cv2
import rclpy

from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from ultralytics import YOLO


class ImageSubscriber(Node):

    def __init__(self):
        super().__init__('image_subscriber')

        self.bridge = CvBridge()

        self.model = YOLO(
            '/home/el0615/Projects/02_ros2_human_following/ros2_ws/yolov8n.pt'
        )

        self.subscription = self.create_subscription(
            Image,
            '/rgb',
            self.image_callback,
            10
        )

        self.get_logger().info(
            'Image subscriber and YOLO model started'
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

            cv2.imshow(
                'YOLO Person Detection',
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