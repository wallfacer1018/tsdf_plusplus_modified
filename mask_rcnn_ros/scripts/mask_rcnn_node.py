#!/usr/bin/env python
import os
import threading
import numpy as np
import cv2
from cv_bridge import CvBridge
import matplotlib.pyplot as plt

import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import RegionOfInterest
from mask_rcnn_ros.msg import Result

# Detectron2 imports
import multiprocessing as mp
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from predictor import VisualizationDemo

# 配置模型路径和参数
CONFIG_FILE = "/home/wby/sr_ws/src/mask_rcnn_ros/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"
MODEL_WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x/137260431/model_final_a54504.pkl"
CONFIDENCE_THRESHOLD = 0.5

# COCO类别名称（必须与Detectron2的元数据顺序一致）
CLASS_NAMES = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

class MaskRCNNNode(object):
    def __init__(self):
        self._cv_bridge = CvBridge()
        
        # 初始化Detectron2配置
        mp.set_start_method("spawn", force=True)
        cfg = get_cfg()
        cfg.merge_from_file(CONFIG_FILE)
        cfg.MODEL.WEIGHTS = MODEL_WEIGHTS
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = CONFIDENCE_THRESHOLD
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = CONFIDENCE_THRESHOLD
        cfg.freeze()
        
        # 创建可视化演示对象
        self.demo = VisualizationDemo(cfg)
        
        # ROS参数设置
        self._rgb_input_topic = rospy.get_param('~input', '/camera/color/image_raw')
        self._visualization = rospy.get_param('~visualization', True)
        
        self._last_msg = None
        self._msg_lock = threading.Lock()

    def run(self):
        self._result_pub = rospy.Publisher('~result', Result, queue_size=1)
        vis_pub = rospy.Publisher('~visualization', Image, queue_size=1)

        rospy.Subscriber(self._rgb_input_topic, Image,
                         self._image_callback, queue_size=1)

        rate = rospy.Rate(100)  # 100Hz
        while not rospy.is_shutdown():
            msg = self._get_latest_msg()
            if msg is not None:
                self._process_image(msg, vis_pub)
            rate.sleep()

    def _get_latest_msg(self):
        if self._msg_lock.acquire(False):
            msg = self._last_msg
            self._last_msg = None
            self._msg_lock.release()
            return msg
        return None

    def _process_image(self, msg, vis_pub):
        try:
            # 转换ROS图像消息为OpenCV格式
            cv_image = self._cv_bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # 运行目标检测
            predictions, visualized_output = self.demo.run_on_image(cv_image)
            instances = predictions["instances"]
            
            # 创建并发布结果消息
            result_msg = self._create_result_msg(msg, instances)
            self._result_pub.publish(result_msg)
            
            # 发布可视化结果
            if self._visualization:
                vis_image = visualized_output.get_image()[:, :, ::-1]  # RGB转BGR
                vis_msg = self._cv_bridge.cv2_to_imgmsg(vis_image, "bgr8")
                vis_pub.publish(vis_msg)
                
        except Exception as e:
            rospy.logerr("Error processing image: %s" % str(e))

    def _create_result_msg(self, msg, instances):
        result = Result()
        result.header = msg.header
        
        # 获取图像尺寸
        image_height, image_width = instances.image_size
        
        for i in range(len(instances)):
            box = instances.pred_boxes.tensor[i].cpu().numpy()
            
            # 坐标转换和边界约束
            x1 = max(0, int(round(box[0])))
            y1 = max(0, int(round(box[1])))
            x2 = min(image_width, int(round(box[2])))
            y2 = min(image_height, int(round(box[3])))

            # 填充ROI消息
            roi = RegionOfInterest()
            roi.x_offset = x1
            roi.y_offset = y1
            roi.width = max(0, x2 - x1)
            roi.height = max(0, y2 - y1)
            result.boxes.append(roi)
            
            # 处理类别信息
            class_id = instances.pred_classes[i].item()
            result.class_ids.append(class_id)
            result.class_names.append(CLASS_NAMES[class_id])
            
            # 处理置信度
            score = instances.scores[i].item()
            result.scores.append(score)
            
            # 处理掩码
            mask = instances.pred_masks[i].cpu().numpy().astype(np.uint8)
            mask_msg = self._cv_bridge.cv2_to_imgmsg(mask * 255, "mono8")
            mask_msg.header = msg.header
            result.masks.append(mask_msg)
            
        return result

    def _image_callback(self, msg):
        if self._msg_lock.acquire(False):
            self._last_msg = msg
            self._msg_lock.release()

def main():
    rospy.init_node('mask_rcnn')
    node = MaskRCNNNode()
    node.run()

if __name__ == '__main__':
    main()