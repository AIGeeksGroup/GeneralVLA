#!/usr/bin/env python3
"""接收 compressedDepth 格式深度图像。

用法: source /opt/ros/jazzy/setup.bash && python3 capture_depth_compressedDepth.py
"""
import sys
import zlib

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np


class DepthCompressedSubscriber(Node):
    def __init__(self, topic_name='/camera/aligned_depth_to_color/image_raw/compressedDepth'):
        super().__init__('depth_compressedDepth_subscriber')
        self.topic_name = topic_name
        self.frame_count = 0
        self.last_data_size = 0

        # QoS 匹配发布者
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.subscription = self.create_subscription(
            CompressedImage,
            topic_name,
            self.listener_callback,
            qos
        )
        self.get_logger().info(f'订阅 {topic_name} ...')
        self.get_logger().info('QoS: RELIABLE + TRANSIENT_LOCAL')

    def decode_compressed_depth(self, msg):
        """解码 compressedDepth 格式的 PNG 压缩深度图。"""
        # compressedDepth 格式: "16UC1; compressedDepth png" 或 "16UC1; compressedDepth rvl"
        fmt_parts = msg.format.split(';')
        if len(fmt_parts) < 2:
            self.get_logger().error(f'未知格式: {msg.format}')
            return None

        encoding = fmt_parts[0].strip()  # e.g., "16UC1"
        compression = fmt_parts[1].strip()  # e.g., "compressedDepth png"

        data = bytes(msg.data)
        data_size = len(data)

        if data_size == 0:
            return None

        # compressedDepth PNG 格式：前12字节是头，之后是PNG数据
        # 格式: [4 bytes unknown][4 bytes width][4 bytes ???] + PNG
        if 'png' in compression.lower():
            if len(data) < 16:
                return None

            # 尝试不同的偏移找到PNG签名
            png_offset = None
            for offset in [8, 12, 16]:
                if len(data) > offset + 4:
                    if data[offset:offset+4] == b'\x89PNG':
                        png_offset = offset
                        break

            if png_offset is None:
                # 回退：尝试从头解码
                png_offset = 0

            # 解析 width/height（如果已知偏移）
            if png_offset >= 12:
                height = int.from_bytes(data[0:4], byteorder='little')
                width = int.from_bytes(data[4:8], byteorder='little')
                self.get_logger().debug(f'解析尺寸: {width}x{height}')

            png_data = data[png_offset:]

            # 解压 PNG
            nparr = np.frombuffer(png_data, np.uint8)
            depth_img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

            if depth_img is not None:
                return depth_img

        return None

    def listener_callback(self, msg):
        self.frame_count += 1
        data_size = len(msg.data)

        self.get_logger().info(
            f'帧 #{self.frame_count}: 格式={msg.format}, 数据大小={data_size} bytes'
        )

        if data_size == 0:
            self.get_logger().warn('收到空数据！')
            return

        # 解码深度图
        depth_img = self.decode_compressed_depth(msg)

        if depth_img is not None:
            h, w = depth_img.shape
            depth_min = depth_img[depth_img > 0].min() if np.any(depth_img > 0) else 0
            depth_max = depth_img.max()

            self.get_logger().info(
                f'  解码成功: {w}x{h}, 深度范围=[{depth_min}, {depth_max}] mm '
                f'(={depth_min/1000:.2f}m - {depth_max/1000:.2f}m)'
            )

            # 保存可视化图像
            if self.frame_count <= 30:
                # 归一化到 0-255 用于可视化
                depth_vis = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                # 应用 colormap
                depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

                filename = f'/tmp/depth_compressedDepth_frame_{self.frame_count}.png'
                cv2.imwrite(filename, depth_colormap)
                self.get_logger().info(f'  保存可视化图像到: {filename}')
        else:
            self.get_logger().error('解码失败')


def main(args=None):
    rclpy.init(args=args)

    # 选择话题
    topic = '/camera/aligned_depth_to_color/image_raw/compressedDepth' # '/camera/color/image_raw' # '/camera/aligned_depth_to_color/image_raw/compressedDepth'
    if len(sys.argv) > 1:
        topic = sys.argv[1]

    node = DepthCompressedSubscriber(topic)

    try:
        import time
        start_time = time.time()
        while rclpy.ok() and (time.time() - start_time) < 30 and node.frame_count < 10:
            rclpy.spin_once(node, timeout_sec=0.1)

        if node.frame_count == 0:
            node.get_logger().error('30秒内未收到任何帧！')
        else:
            node.get_logger().info(f'共收到 {node.frame_count} 帧')

    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
