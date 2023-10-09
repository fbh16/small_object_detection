#!/usr/bin/env python3
from __future__ import print_function

import cv2
import sys
import time
import rospy
import signal
import numpy as np
from collections import deque
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from filterpy.kalman import KalmanFilter
from darknet_ros_msgs.msg import BoundingBox,BoundingBoxes

try:
  from numba import jit
except:
  def jit(func):
    return func
np.random.seed(0)

def signal_handler(signal, frame): # ctrl + c -> exit program
        sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)


def linear_assignment(cost_matrix):
  try:
    import lap
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array([[y[i],i] for i in x if i >= 0]) #
  except ImportError:
    from scipy.optimize import linear_sum_assignment
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))

@jit
def iou(bb_test, bb_gt):
  """
  Computes IUO between two bboxes in the form [x1,y1,x2,y2]
  """
  
  xx1 = np.maximum(bb_test[0], bb_gt[0])
  yy1 = np.maximum(bb_test[1], bb_gt[1])
  xx2 = np.minimum(bb_test[2], bb_gt[2])
  yy2 = np.minimum(bb_test[3], bb_gt[3])
  w = np.maximum(0., xx2 - xx1)
  h = np.maximum(0., yy2 - yy1)
  wh = w * h
  o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
    + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)

  return(o)

# 将bbox由[x1,y1,x2,y2]形式转为 [框中心点x,框中心点y,框面积s,宽高比例r]^T
def convert_bbox_to_z(bbox):
  """
  Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio

  input bbox: [x1, y1, x2, y2]
  output:     [center x, center y, s, r] 4行1列  s是面积, r是w/h

  """
  w = bbox[2] - bbox[0]
  h = bbox[3] - bbox[1]
  x = bbox[0] + w/2.
  y = bbox[1] + h/2.
  s = w * h    #scale is just area
  r = w / float(h)
  return np.array([x, y, s, r]).reshape((4, 1))

# 将[x,y,s,r]形式的bbox，转为[x1,y1,x2,y2]形式
def convert_x_to_bbox(x,score=None):
  """
  Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right

  input:       [center x, center y, s, r]
  output bbox: [x1, y1, x2, y2]   

  """
  w = np.sqrt(x[2] * x[3])
  h = x[2] / w
  if(score==None):
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
  else:
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))


class KalmanBoxTracker(object):
  count = 0
  def __init__(self, bbox):
    """
    Initialises a tracker using initial bounding box.
    使用初始边界框初始化跟踪器
    """
    # define constant velocity model  #定义匀速模型
    self.kf = KalmanFilter(dim_x=7, dim_z=4) #状态变量是7维， 观测值是4维的

    self.kf.F = np.array([[1,0,0,0,1,0,0], # transition Matrix
                          [0,1,0,0,0,1,0],
                          [0,0,1,0,0,0,1],
                          [0,0,0,1,0,0,0],  
                          [0,0,0,0,1,0,0],
                          [0,0,0,0,0,1,0],
                          [0,0,0,0,0,0,1]])

    self.kf.H = np.array([[1,0,0,0,0,0,0], # measurement Matrix
                          [0,1,0,0,0,0,0],
                          [0,0,1,0,0,0,0],
                          [0,0,0,1,0,0,0]])

    self.kf.R[2:,2:] *= 10.
    self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities #对未观测到的初始速度给出高的不确定性
    self.kf.P *= 10. #默认定义的协方差矩阵是np.eye(dim_x)，将P中的数值与10， 1000相乘，赋值不确定性
    self.kf.Q[-1,-1] *= 0.01
    self.kf.Q[4:,4:] *= 0.01
    self.kf.x[:4] = convert_bbox_to_z(bbox) #将bbox转为 [x,y,s,r]^T 形式，赋给状态变量X的前4位
    self.time_since_update = 0
    self.id = KalmanBoxTracker.count
    KalmanBoxTracker.count += 1
    self.history = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0

  def update(self,bbox):
    """
    Updates the state vector with observed bbox.
    用观察到的bbox更新状态向量
    """
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1
    self.kf.update(convert_bbox_to_z(bbox))

  def predict(self):
    """
    Advances the state vector and returns the predicted bounding box estimate.
    推进状态向量 并返回预测的边界框估计值
    """
    if((self.kf.x[6]+self.kf.x[2])<=0):
        self.kf.x[6] *= 0.0
    self.kf.predict()
    self.age += 1
    if(self.time_since_update>0):
        self.hit_streak = 0
    self.time_since_update += 1
    self.history.append(convert_x_to_bbox(self.kf.x))
    return self.history[-1]

  def get_state(self):
    """
    Returns the current bounding box estimate.
    返回当前边框估计值
    input:       [center x, center y, s, r]
    output: bbox [x1, y1, x2, y2] 
    
    """
    return convert_x_to_bbox(self.kf.x)

def associate_detections_to_trackers(detections,trackers,iou_threshold = 0.3):
    """
    将detections分配给被跟踪对象(都表示为bbox)
    返回3个列表: matches(tracking), unmatched_detections(new track), unmatched_trackers(delet)

    input:  检测框detections, 跟踪器trackers, iou阈值

    output: 匹配的数组matches,
            未匹配检测数组np.array(unmatched_detections),
            未匹配跟踪器数组np.array(unmatched_trackers)
    """
    if(len(trackers)==0):
        return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)
    iou_matrix = np.zeros((len(detections),len(trackers)),dtype=np.float32)

    for d,det in enumerate(detections):
        for t,trk in enumerate(trackers):
            iou_matrix[d,t] = iou(det,trk)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)  # 利用匈牙利算法关联前后两帧的坐标信息 | 得到匹配项
    else:
        matched_indices = np.empty(shape=(0,2))
  
    # ---- 记录未匹配的检测框及轨迹
    unmatched_detections = []
    for d, det in enumerate(detections):
        if(d not in matched_indices[:,0]):
            unmatched_detections.append(d)    # 获得未匹配的检测框

    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if(t not in matched_indices[:,1]):
            unmatched_trackers.append(t)      # 获得未匹配的跟踪器

    # 过滤小于iou阈值的matched_box
    matches = [] 
    for m in matched_indices:  # 利用iou阈值过滤干扰框
        if(iou_matrix[m[0], m[1]] < iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1,2))
      
    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
    def __init__(self, max_age=200, min_hits=1):
        rospy.init_node('sort', anonymous=True)

        input_topic = rospy.get_param("~input_box_topic")
        self.subb = rospy.Subscriber(input_topic, BoundingBoxes, self.boxcallback, queue_size=10)

        output_topic = rospy.get_param("~output_box_topic")
        self.pubb = rospy.Publisher(output_topic, BoundingBoxes, queue_size=50) # default = 50

        self.rate = rospy.Rate(100)
        display = rospy.get_param("~display", False)
        max_age = rospy.get_param("~max_age", max_age)
        min_hits = rospy.get_param("~min_hits", min_hits)
        self.iou_threshold = rospy.get_param("~iou_threshold")

        if display:
            self.display = display
            img_topic = rospy.get_param("~input_img_topic", '/usb_cam/image_raw')  
            self.subimage = rospy.Subscriber(img_topic, Image, self.imgcallback)

            tracking_topic = rospy.get_param("~output_tracking_topic")
            self.pubimage = rospy.Publisher(tracking_topic, Image, queue_size=10)

        self.max_age = max_age   # 未被检测框更新的跟踪器，随帧数增加，超过max_age之后被删除
        self.min_hits = min_hits # 新出现的目标，没有被跟踪，需要连续match min_hits次才会被跟踪
        self.trackers = []
        self.frame_count = 0
        self.img_in = 0
        self.bbox_checkin = 0
        self.bridge = CvBridge()

    def imgcallback(self, img_msg):
    
        self.img = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        # self.img = cv2.resize(img, (640,480))
        self.img_in = 1  # flag
        return

    def boxcallback(self, msg):
        dets = []
        boxes = []
        for i in range(len(msg.bounding_boxes)):
            bbox = msg.bounding_boxes[i] 
            boxes.append(np.array([bbox.probability, bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax, bbox.id, bbox.Class]))
            dets.append(np.array([bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax, bbox.probability]))

        self.dets = np.array(dets)
        self.boxes = np.array(boxes)
        self.bbox_checkin=1  # flag
        return

    def update(self, dets=np.empty((0, 5))):
        """
        参数:
        dets—以[[x1,y1,x2,y2,score], [x1,y1,x2,y2,score]，…
        要求:即使检测为空，也必须对每个帧调用此方法一次
        返回一个类似的数组, 其中最后一列是对象ID
        注意:返回的对象数量可能与提供的检测数量不同
        """
        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):  # 删除
            self.trackers.pop(t)

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)
        self.matched = matched
        self.unmatched_dets = unmatched_dets 
        self.unmatched_trks = unmatched_trks

        # 用当前匹配的检测框和预测框加权，更新替换该帧的预测框
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        # 为新出现的检测框初始化跟踪器
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i,:])
            self.trackers.append(trk)

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            # 返回当前边框估计值, output: bbox [x1, y1, x2, y2], d=x1
            d = trk.get_state()[0]
            # 新出现目标必须连续match min_hits次才认为这条轨迹可靠
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive
            i -= 1
            # 删除消失超过max_age的跟踪目标
            if(trk.time_since_update > self.max_age):
                # print("id-{}'s missing time > max age={}".format(i, self.max_age))
                self.trackers.pop(i)
        if(len(ret)>0):
        # np.concatenate() 函数将所有元素按行拼接成一个矩阵并返回
            return np.concatenate(ret)
        return np.empty((0,5))



if __name__ == '__main__':
    mot_tracker = Sort(max_age=7, min_hits=1) # create instance of the SORT tracker
    pts = [deque(maxlen=20) for _ in range(999)]  # 设置轨迹长度
    print('sort is ready')
    while True:
        try:  
            start_time = time.time()
            if mot_tracker.bbox_checkin == 1:
                trackers = mot_tracker.update(mot_tracker.dets)  # 将当前帧yolo检测到的bbox送入sort初始化的kf跟踪器，获得 预测结果
                mot_tracker.bbox_checkin = 0
            else:
                trackers = mot_tracker.update(np.empty((0,5)))  # 没有检测到时发送空数组

            line_thickness = 3
            r = BoundingBoxes()
            for d in range(len(trackers)):  #遍历预测结果
                line_thickness = line_thickness or round(0.002 * (640 + 480) / 2) + 1
                rb = BoundingBox()
                rb.probability = mot_tracker.dets[0][4]
                rb.xmin = int(trackers[d,0])
                rb.ymin = int(trackers[d,1])
                rb.xmax = int(trackers[d,2])
                rb.ymax = int(trackers[d,3])
                rb.id = int(trackers[d,4])  # TODO: 用检测结果赋id，空数组会不会使id+1？用dets的id是否能解决
                rb.Class = mot_tracker.boxes[0][6]
                r.bounding_boxes.append(rb)

                c1, c2 = (int(rb.xmin), int(rb.ymin)), (int(rb.xmax), int(rb.ymax))
                c = ((round((c1[0] + c2[0]) / 2), round((c1[1] + c2[1]) / 2)))  # round()四舍五入，获取box中心

                pts[rb.id].append(c)  # 将box中心坐标存入对应id

                cv2.circle(mot_tracker.img, (c[0],c[1]) , 1, (0,0,200), 5)

                for j in range(1, len(pts[rb.id])):
                    if  pts[rb.id][j - 1] is None or  pts[rb.id][j] is None:
                        continue
                    thickness = int(np.sqrt(64 / float(j + 1)) * 2)
                    cv2.line(mot_tracker.img, ( pts[rb.id][j - 1]), ( pts[rb.id][j]), (0,255,0), thickness = 2)

                if mot_tracker.img_in==1 and mot_tracker.display:
                    color = (0, 0, 255)
                    c1, c2 = (rb.xmin, rb.ymin), (rb.xmax, rb.ymax)
                    cv2.rectangle(mot_tracker.img, c1, c2, color, thickness=2, lineType=cv2.LINE_AA)
                    font_thickness = 1
                    t_size = cv2.getTextSize(rb.Class, 0, fontScale=1, thickness=font_thickness)[0]
                    # 填充label的框的尺寸 t_size[0]=72, t_size[1]=15
                    c2 = c1[0] + t_size[0] + 137, c1[1] - t_size[1]
                    # cv2.rectangle(mot_tracker.img, c1, c2, color, -1, cv2.LINE_AA)  # 填充label
                    cv2.putText(mot_tracker.img,'ID:{} '.format(rb.id), (c1[0], c1[1] - 2), 0, 0.5,
                        [255, 255, 0], thickness=font_thickness, lineType=cv2.LINE_AA)
            
            end_time = time.time()
            # print(end_time - start_time)

            if mot_tracker.img_in==1 and mot_tracker.display:
                cv2.imshow('YoloSort', mot_tracker.img)
                cv2.waitKey(1)
                
            if len(r.bounding_boxes) > 0: # prevent empty box
                r.header.stamp = rospy.Time.now()
                mot_tracker.pubb.publish(r)
             
            mot_tracker.rate.sleep()

        except (rospy.ROSInterruptException, SystemExit, KeyboardInterrupt):
            sys.exit(0)
