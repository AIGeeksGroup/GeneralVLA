# 手眼标定及验证

## 在终端按序输入

1.source /opt/ros/jazzy/setup.bash

2.spyder   #打开/home/common/eye_hand/0308_modified.py

3.ros2 run piper piper_single_ctrl --ros-args -p can_port:=can0  #启动机械臂（进入示教状态，绿灯亮）

4.ros2 launch realsense2_camera rs_launch.py  serial_no:=f1271134  align_depth.enable:=true  #启动相机L515，注意对齐的深度图像需额外启动，这里已加align_depth.enable:=true

5.#上述启动均无问题，运行 0308_modified.py
#手眼标定文件，鼠标左键点击即可记录相机坐标（非实时），移动机械臂末端至标定点，按s即可完成一次采样，总共20组样本
   # 采样完成后，会生成以下文件

eye_to_hand_result.npz  #标定结果总文件

> base_point.npy  #机械臂下的采样坐标

>cam_points.npy  #相机下的采样坐标

>R.npy  #标定得到的R矩阵

>t.npy  #标定得到的t矩阵

>errors.npy  #标定误差

>rmse.npy  #均方根误差

6.运行yanzheng.py
 #手眼标定验证脚本，输入R和t（注意这里输入的是R.txt和t.txt），得到验证结果


# 给定坐标运动到指定位置
common用户
source /py_venvs/brainarm/bin/activate

1.source /opt/ros/jazzy/setup.bash

2.spyder   #打开/home/common/eye_hand/click_to_above_move_fixed_init.py

3.ros2 run piper piper_single_ctrl --ros-args -p can_port:=can0  #启动机械臂（取消示教状态，机械臂绿灯不亮）

4.ros2 launch realsense2_camera rs_launch.py  serial_no:=f1271134  align_depth.enable:=true 

5.运行click_to_above_move_fixed_init.py

6.初始位置写死（138.26，-2.74，160.77，-171.41，-1.31，-82.54，-0.00）

7.按空格键回到初始位置，鼠标左键点击相机彩色图像中指定位置，按回车键则运动到该点位置，点击下一个点，回车键运动到下一个点，或回车键返回初始位置。




