# Modified Version of TSDF++

This is a modified version of TSDF++ to run on Ubuntu 20.04.

## Modifications
1. **Mask R-CNN Replaced by Detectron2**: The original Mask R-CNN has been replaced by Detectron2.
2. **ROS Topic Names Changed**: ROS topic names have been modified to work with the RealSense camera.
3. **CMake File Adjustments**: Some CMake files have been modified to avoid compilation errors.

## Running TSDF++ with realsense camera
Follow these steps to run TSDF++:

1. **Launch RealSense Camera** (or use rosbag if no camera is available):
   ```bash
   roslaunch realsense2_camera rs_camera.launch
   ```
2. **Activate Environment for Detectron2** :
   ```bash
   conda activate det2
   ```
3. **Launch TSDF++ Pipeline**:
   ```bash
   roslaunch tsdf_plusplus_ros tsdf_plusplus_pipeline.launch
   ```
4. **Use Static tf**:
   ```bash
   rosrun tf static_transform_publisher 0 0 0 0 0 0 world camera_color_optical_frame 10
   ```
## Running TSDF++ with synthesized demo dataset
Download dataset with link <http://visual.cs.ucl.ac.uk/pubs/cofusion/data/car4-full.tar.gz>

1. **Activate Environment for Detectron2** :
   ```bash
   conda activate det2
   ```
2. **Launch TSDF++ Pipeline**:
   ```bash
   roslaunch tsdf_plusplus_ros tsdf_plusplus_pipeline.launch
   ```
3. **Use the publisher node(remember to change data path)**:
   ```bash
   python demo_data_pub.py
   ```