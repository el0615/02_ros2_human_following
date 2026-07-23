import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image


class ImageSubscriber(Node):

    def __init__(self):
        super().__init__('image_subscriber')

        self.subscription = self.create_subscription(
            Image,
            '/rgb',
            self.image_callback,
            10
        )

        self.get_logger().info('Image subscriber node started')

    def image_callback(self, msg):
        self.get_logger().info(
            f'Image received: width={msg.width}, height={msg.height}'
        )


def main(args=None):
    rclpy.init(args=args)

    node = ImageSubscriber()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()