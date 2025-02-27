This a modified version of TSDF++ in order to run on Ubuntu20.04.
1.Mask RCNN were replaced by detectron2
2.ROS Topic names were changed to run with realsense camera
3.Modifications on some of the CMake files to avoid compilation error
To run TSDF++:
roslaunch realsense2_camera rs_camera.launch (use rosbag if no camera available)
conda activate det2 (python environment for detectron2)
roslaunch tsdf_plusplus_ros tsdf_plusplus_pipeline.launch
rosrun tf static_transform_publisher 0 0 0 0 0 0 world camera_color_optical_frame 10
