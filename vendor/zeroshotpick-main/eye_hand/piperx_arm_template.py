#!/usr/bin/env python3
# -*-coding:utf8-*-
# This file controls a single robotic arm node and handles the movement of the robotic arm with a gripper.
import sys,os

sys.path.append("/py_venvs/brainarm/lib/python3.12/site-packages")
sys.path.append("/opt/ros/jazzy/lib/python3.12/site-packages")
sys.path.append(os.environ.get("HOME")+"/brainarm-ws/install/piper_msgs/lib/python3.12/site-packages")

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
import time
import threading
import argparse
import math
from piper_sdk import *
from piper_sdk import C_PiperInterface
from piper_msgs.msg import PiperStatusMsg, PosCmd
from piper_msgs.srv import Enable
from geometry_msgs.msg import Pose, PoseStamped
from scipy.spatial.transform import Rotation as R  # For Euler angle to quaternion conversion
from numpy import clip
from builtin_interfaces.msg import Time
import numpy as np
import time

class PiperX_Arm(Node):
    # Position Unit: mm, Angle Unit: deg
    def __init__(self) -> None:
        super().__init__('piperx_arm')

        self.x, self.y, self.z, self.rx,self.ry,self.rz = 0,0,0,0,0,0
        self.joint_states = [0.0,0.0,0.0,0.0,0.0,0.0]
        self.ori_q = None
        self.is_pose_available, self.is_joint_available = False, False
        self.gripper_pos,self.gripper_effort = 0,0
        self.timestamp = None
        self.factor = 180 / 3.1415926
        
        # sub data
        self.create_subscription(PoseStamped, 'end_pose_stamped', self.end_pose_stamp_callback, 10)
        self.create_subscription(JointState, 'joint_states_single', self.joint_states_callback, 10)
        # self.create_subscription(PiperStatusMsg, 'arm_status', self.arm_status_callback, 1)
        # pub cmd
        self.end_pose_pub = self.create_publisher(PosCmd, 'pos_cmd', 10)
        self.joint_pub = self.create_publisher(JointState, 'joint_cmd', 10)
        
        while self.end_pose_pub.get_subscription_count()==0 or self.joint_pub.get_subscription_count()==0:
            print("connecting to piper node")
            time.sleep(0.1)

        # print("Go to zero...")
        # self.go_back_to_zero_point()

        
        # self.publisher_thread = threading.Thread(target=self.publish_thread)
        # self.publisher_thread.start()

    def end_pose_stamp_callback(self, data):
        self.timestamp = data.header.stamp.sec+data.header.stamp.nanosec/1e9

        self.x = data.pose.position.x
        self.y = data.pose.position.y
        self.z = data.pose.position.z 
        self.ori_q = data.pose.orientation
        
        euler = R.from_quat([self.ori_q.x,self.ori_q.y,self.ori_q.z,self.ori_q.w]).as_euler('xyz',degrees=True)
        self.rx = (euler[0])
        self.ry = (euler[1])
        self.rz = (euler[2])
        self.is_pose_available = True
        # print(self.timestamp,'    ',self.x,self.y,self.z,'    ',self.rx,self.ry,self.rz)
                
    def joint_states_callback(self, data):
        
        self.gripper_pos = data.position[-1]
        self.gripper_effort = data.effort[-1]
        self.joint_states = [angle/0.017444 for angle in data.position[:6]]
        self.is_joint_available = True

        # print(self.joint_states)
        
    def get_all_states(self):
        if self.is_joint_available and self.is_pose_available:
            return [angle*0.017444 for angle in self.joint_states]+[self.x/1000,self.y/1000,self.z/1000,self.ori_q.x,self.ori_q.y,self.ori_q.z,self.ori_q.w,self.gripper_pos],\
                [self.rx,self.ry,self.rz]
        else:
            return None
                

    def arm_status_callback(self, data):
        
        a=1
        # print(data)
        
    def go_back_to_zero_point(self):
        joint_cmd = JointState()
        joint_cmd.position = [0,0,0,0,0,0]
        self.joint_pub.publish(joint_cmd)
        print(joint_cmd)
        
    def send_pose_gripper_cmd(self, x, y, z, rx, ry, rz, grip, vec=50):
        pos_cmd = PosCmd()
        pos_cmd.x, pos_cmd.y, pos_cmd.z = float(x), float(y), float(z)
        pos_cmd.roll, pos_cmd.pitch, pos_cmd.yaw = float(rx), float(ry), float(rz)
        pos_cmd.gripper = float(grip)
        pos_cmd.mode1 = int(vec)
        self.end_pose_pub.publish(pos_cmd)


    def publish_thread(self):
        """Publish messages to the robotic arm
        """
        rate = self.create_rate(200)
        
        t = 0
        cnt = 0
        t0 = time.time()
        while rclpy.ok():
            t += 0.01
            cnt+=1

            
            # print((time.time()-t0)/cnt)
            # self.send_pose_gripper_cmd(240, 30*float(np.cos(t)), 250.0+30*float(np.sin(t)), 
            #                           -120, 11, -98, 40+40*float(np.cos(t)), 50)
            
            rate.sleep()

        print('sent ',pos_cmd)

        
        

if __name__ == '__main__':
    
    rclpy.init()
    piperx_arm_node = PiperX_Arm()
    try:
        rclpy.spin(piperx_arm_node)
    except KeyboardInterrupt:
        pass
    finally:
        piperx_arm_node.destroy_node()
        rclpy.shutdown()
    