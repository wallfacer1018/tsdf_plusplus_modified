import rospy
import cv2
import os
import pyexr
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster

def load_camera_poses(pose_file_path):
    poses = []
    with open(pose_file_path, 'r') as file:
        for line in file:
            components = line.strip().split()
            if len(components) != 8:
                continue
            tx, ty, tz = map(float, components[1:4])
            qx, qy, qz, qw = map(float, components[4:8])
            poses.append(((tx, ty, tz), (qx, qy, qz, qw)))
    return poses

def main():
    rospy.init_node('dataset_publisher_node')
    bridge = CvBridge()
    tf_broadcaster = TransformBroadcaster()

    color_publisher = rospy.Publisher('/camera/color/image_raw', Image, queue_size=10)
    depth_publisher = rospy.Publisher('/camera/aligned_depth_to_color/image_raw', Image, queue_size=10)
    camera_info_publisher = rospy.Publisher('/camera/color/camera_info', CameraInfo, queue_size=10)

    dataset_path = '/home/wby/data/car4-full'
    color_dir = os.path.join(dataset_path, 'colour')
    depth_dir = os.path.join(dataset_path, 'depth_original')
    pose_file = os.path.join(dataset_path, 'trajectories', 'gt-cam-0.txt')

    camera_poses = load_camera_poses(pose_file)
    if len(camera_poses) != 480:
        rospy.logerr("Incorrect number of camera poses: %d", len(camera_poses))
        return

    camera_info = CameraInfo()
    camera_info.height = 540
    camera_info.width = 960
    camera_info.distortion_model = "plumb_bob"
    camera_info.D = [0.0, 0.0, 0.0, 0.0, 0.0]
    camera_info.K = [564.3, 0.0, 480.0, 0.0, 564.3, 270.0, 0.0, 0.0, 1.0]
    camera_info.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    camera_info.P = [564.3, 0.0, 480.0, 0.0, 0.0, 564.3, 270.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    camera_info.binning_x = 0
    camera_info.binning_y = 0
    camera_info.roi.x_offset = 0
    camera_info.roi.y_offset = 0
    camera_info.roi.height = 0
    camera_info.roi.width = 0
    camera_info.roi.do_rectify = False

    rate = rospy.Rate(1)  # 1 Hz

    for frame_num in range(480):
        if rospy.is_shutdown():
            break

        current_time = rospy.Time.now()

        # Process and publish color image
        color_image_path = os.path.join(color_dir, f"Color{frame_num + 1:04d}.png")
        color_image = cv2.imread(color_image_path)
        if color_image is None:
            rospy.logerr(f"Failed to load color image: {color_image_path}")
            continue
        color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        try:
            color_msg = bridge.cv2_to_imgmsg(color_image_rgb, encoding="rgb8")
        except Exception as e:
            rospy.logerr(f"Color image conversion error: {e}")
            continue
        color_msg.header.seq = frame_num + 1
        color_msg.header.stamp = current_time
        color_msg.header.frame_id = "camera_color_optical_frame"

        # Process and publish depth image
        depth_image_path = os.path.join(depth_dir, f"Depth{frame_num + 1:04d}.exr")
        try:
            depth_data = pyexr.read(depth_image_path)
            if depth_data.ndim == 3:
                depth_data = depth_data[:, :, 0]
            depth_msg = bridge.cv2_to_imgmsg(depth_data, encoding="32FC1")
        except Exception as e:
            rospy.logerr(f"Depth image error: {e}")
            continue
        depth_msg.header.seq = frame_num + 1
        depth_msg.header.stamp = current_time
        depth_msg.header.frame_id = "camera_color_optical_frame"

        # Publish camera pose transform
        if frame_num >= len(camera_poses):
            rospy.logerr("Insufficient poses for current frame")
            break
        (tx, ty, tz), (qx, qy, qz, qw) = camera_poses[frame_num]
        transform = TransformStamped()
        transform.header.seq = frame_num + 1
        transform.header.stamp = current_time
        transform.header.frame_id = "world"
        transform.child_frame_id = "camera_color_optical_frame"
        transform.transform.translation.x = tx
        transform.transform.translation.y = ty
        transform.transform.translation.z = tz
        transform.transform.rotation.x = qx
        transform.transform.rotation.y = qy
        transform.transform.rotation.z = qz
        transform.transform.rotation.w = qw
        tf_broadcaster.sendTransform(transform)

        # Publish camera info
        camera_info.header.seq = frame_num + 1
        camera_info.header.stamp = current_time
        camera_info.header.frame_id = "camera_color_optical_frame"
        camera_info_publisher.publish(camera_info)

        # Publish images
        color_publisher.publish(color_msg)
        depth_publisher.publish(depth_msg)
        print(frame_num)

        rate.sleep()

if __name__ == '__main__':
    main()