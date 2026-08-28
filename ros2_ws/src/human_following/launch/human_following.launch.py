from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='human_following',
            executable='image_subscriber',
            name='image_subscriber',
            output='screen',
        )
    ])
