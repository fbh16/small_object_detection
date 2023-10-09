# coding=utf-8

# python video2bag.py results/railway.mp4 railway.bag
# rosbag compress --lz4 railway.bag

import time, sys, os
from ros import rosbag
import roslib, rospy
roslib.load_manifest('sensor_msgs')
from sensor_msgs.msg import Image

from cv_bridge import CvBridge
import cv2

TOPIC = '/usb_cam/image_raw'

def CreateVideoBag(videopath, bagname):
    '''Creates a bag file with a video file'''
    print('videopath', videopath)
    print('bagname', bagname)
    bag = rosbag.Bag(bagname, 'w') #创建ROS bag文件对象
    cap = cv2.VideoCapture(videopath) #read frames from a video file or camera stream
    cb = CvBridge()
    # prop_fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    prop_fps = cap.get(cv2.CAP_PROP_FPS) #视频的帧率
    # print(f'rosbag compress --lz4 {bagname}')
    print('视频帧率：',prop_fps)

    """
     设置帧率
    """
    if prop_fps != prop_fps or prop_fps <= 1e-2:
        print ("Warning: can't get FPS. Assuming 24.")
        prop_fps = 30

    ret = True
    seq = 1
    frame_id = 0
    while(ret):
        ret, frame = cap.read()
        if not ret:
            break
        # stamp = rospy.rostime.Time.from_sec(float(frame_id) / prop_fps)
        stamp = rospy.Time.now()
        frame_id += 1
        image = cb.cv2_to_imgmsg(frame, encoding='bgr8')
        image.header.seq = seq
        seq += 1
        image.header.stamp = stamp
        image.header.frame_id = "camera"
        bag.write(TOPIC, image, stamp) # (topic, msg, t)    
        """
          write()参数：
            topic:字符串类型，表示要写入的消息所属的话题名称。
            msg:ROS消息类型,表示要写入的消息对象。
            t:ROS时间类型,可选参数,表示要写入消息的时间戳。如果不指定该参数，则默认使用当前系统时间戳。
        """
    cap.release()
    bag.close()

if __name__ == "__main__":
    rospy.init_node("v2b", anonymous=True)
    if len( sys.argv ) == 3:
        CreateVideoBag(*sys.argv[1:])
    else:
        print( "Usage: video2bag videofilename bagfilename")
