#!/usr/bin/env python3
import os
import cv2
import sys
import rospy
import torch
import time
import shutil
import numpy as np
from pathlib import Path
from cv_bridge import CvBridge
import torch.backends.cudnn as cudnn
from rostopic import get_topic_type
from sensor_msgs.msg import Image
from darknet_ros_msgs.msg import BoundingBox, BoundingBoxes

# add yolov5 submodule to path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] / "yolov5"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative path
# import from yolov5 submodules
from utils.plots import Annotator
from utils.augmentations import letterbox
from models.common import DetectMultiBackend
from utils.torch_utils import select_device
from utils.general import (check_img_size,non_max_suppression,scale_coords)

def xyxy2ltwh(x):
    """
        inputs: [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        outputs:[x, y, w, h] where xy is left top 
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0]  # top left x
    y[:, 1] = x[:, 1]  # top left y
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

def tran2rawsz(det, imw, imh):
    Kw = imw / 640
    Kh = imh / 480
    det[:,0] = det[:,0] * Kw
    det[:,1] = det[:,1] * Kh
    det[:,2] = det[:,2] * Kw
    det[:,3] = det[:,3] * Kh
    return det

def iou_batch(bb_test, bb_gt):
    """
        input: [x1,y1,x2,y2]
    """
    bb_gt = np.expand_dims(bb_gt, 0)     
    bb_test = np.expand_dims(bb_test, 0) # default = 1
    # print('bbgt',bb_gt,'\n','bbtest',bb_test)
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0]) # 第1个元素，xmin
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1]) # 第2个元素，ymin
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2]) # xmax
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3]) # ymax
    w = np.maximum(0., xx2 - xx1) # 交集的宽，array([1.49])格式
    h = np.maximum(0., yy2 - yy1) # 交集的高
    wh = w * h # 交集, array([0.89])格式
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1]) # 预测框的面积                                     
            + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)   # 检测框的面积  
    # print(o)                           
    return(o)  


def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i],i] for i in x if i >= 0]) #
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def associate_detections_to_trackers(detections, trackers, iou_threshold = 0.3):

    if(len(trackers) ==0):
        return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)
    iou_matrix = np.zeros((len(detections),len(trackers)),dtype=np.float32) 
    # 每个检测框依次和所有预测框iou匹配一次
    for d,det in enumerate(detections):
        for t,trk in enumerate(trackers):
            iou_matrix[d,t] = iou_batch(det,trk)
    """
    array([[0.        , 0.        , 0.9063041 ],
           [0.9402054 , 0.02309724, 0.        ],
           [0.        , 0.9373464 , 0.        ]])
    """
    # print(iou_matrix)
    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        # a.sum(1).max(): 矩阵a的1范数 # a.sum(0).max(): 矩阵a的无穷范数
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
            # print('matched index: ',matched_indices)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0,2))
    # print(matched_indices)

    unmatched_detections = [] 
    for d, det in enumerate(detections):
        # first appear, need a new track
        if(d not in matched_indices[:,0]):
            unmatched_detections.append(d)

    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        # blocked or disappear, need to delete
        if(t not in matched_indices[:,1]):
            tid = [t, int(trk[4])] # t与track的id的对应关系
            unmatched_trackers.append(t)

    matches = []
    # filter out matched with low IOU
    for m in matched_indices:
        if(iou_matrix[m[0], m[1]] < iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1,2))
            # matches_bak.append(m)
    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)
    
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

@torch.no_grad()
class YOLOv5_MultiSOT:
    def __init__(self):
        bs = 1  # batch size
        self.detframe = 1
        self.trkframe = 0
        self.max_age = 5
        self.max_det = 500
        self.iou_thres = 0.3  # track
        self.iou_thresh = 0.6 # det
        self.conf_thres = 0.6
        self.half = True

        self.evaluate = True
        self.view_image = True
        
        cudnn.benchmark = True # set True to speed up constant image size inference
        self.agnostic_nms = True
        self.trks_dict = {}
        self.matched = []
        self.unmatched_trks = []
        self.bridge = CvBridge()
        self.img_size = [480,640]
        weights = rospy.get_param("~weights")
        self.view_image = rospy.get_param("~view_image")
        self.hide_label = rospy.get_param("~hide_label")
        self.device = select_device(str(rospy.get_param("~device","")))
        self.model = DetectMultiBackend(weights, device=self.device, dnn=False, data=rospy.get_param("~data"))
        self.stride, self.names, self.pt, self.engine = (self.model.stride,self.model.names,self.model.pt,self.model.engine)
        self.img_size = check_img_size(self.img_size, s=self.stride)
        self.half &= self.engine and self.device.type != "cpu"  # FP16 supported on limited backends with CUDA
        self.model.warmup(imgsz=(1 if self.pt else bs, 3, *self.img_size), half=self.half)  
        # topic
        input_image_type, input_image_topic, _ = get_topic_type(rospy.get_param("~input_image_topic"), blocking = True)        
        self.image_sub = rospy.Subscriber(input_image_topic, Image, self.callback, queue_size=1)
        self.pred_pub = rospy.Publisher(rospy.get_param("~output_topic"), BoundingBoxes, queue_size=10)
        self.single_cls = rospy.get_param("~classes","")
        
        if self.single_cls != "":
            self.classes = self.single_cls
        else:
            self.classes = None
        print(weights)
        
        self.mk_new_train = rospy.get_param("~mk_new_train")
        if self.evaluate:
            seq_name = rospy.get_param("~seq_name")
            det_root = "/home/fbh/2023_goal/test/Evaluate/TrackEval/data/gt/mot_challenge"
            trk_root = "/home/fbh/2023_goal/test/Evaluate/TrackEval/data/trackers/mot_challenge"
            det_result_pth = self.gen_exp(det_root, seq_name, 'det')
            trk_result_pth = self.gen_exp(trk_root, seq_name, 'trk')
            self.det_f = open(os.path.join(det_result_pth, f'det.txt'), 'w') 
            self.trk_f = open(os.path.join(trk_result_pth, f'{seq_name}.txt'), 'w') 

    def callback(self, data):
        t1 = time.time()
        im = self.bridge.imgmsg_to_cv2(data, "bgr8") 
        im, im0 = self.preprocess(im) #im0是备份

        im = torch.from_numpy(im).to(self.device) 
        im = im.half() if self.model.fp16 else im.float()
        im /= 255
        if len(im.shape) == 3:
            im = im[None]

        pred = self.model(im, augment=False, visualize=False) # 检测
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)
        # Process predictions 
        det = pred[0].cpu().numpy() # resize后的坐标
        bbs = BoundingBoxes()
        bbs.header = data.header
        bbs.image_header = data.header
        annotator = Annotator(im0, line_width=2.5, example=str(self.names)) 
        
        dets = []
        if len(det):
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round() # 还原成原图尺寸为基准的坐标
            # det[:, :4] = tran2rawsz(det[:,:4], im.shape[1], im.shape[0])
            id = 0
            for *xyxy, conf, cls in reversed(det):
                bb = BoundingBox()
                c = int(cls)
                bb.id = id
                bb.Class = self.names[c]
                bb.probability = conf 
                bb.xmin = int(xyxy[0]) 
                bb.ymin = int(xyxy[1])
                bb.xmax = int(xyxy[2])
                bb.ymax = int(xyxy[3])
                bbs.bounding_boxes.append(bb)
                # if self.view_image:  # Add bbox to image
                #     if self.hide_label:
                #         label = False
                #     else:
                #         label = f"{self.names[c]} {conf:.2f}"
                #     annotator.box_label(xyxy, label, color=(255,0,0))       
                #     im0 = annotator.result()

        """
          generate det.txt
        """
        if self.evaluate:
            if len(det):
                for *xyxy, conf, cls in reversed(det):
                    # view(1,4) 增维, view(-1)降维, /gn 归一化
                    xywh = xyxy2ltwh(torch.tensor(xyxy).view(1,4)).view(-1).tolist()
                    print('%d,-1,%.2f,%.2f,%.2f,%.2f,%.2f,-1,-1,-1' % (
                        self.detframe, xywh[0], xywh[1], xywh[2], xywh[3], conf), file=self.det_f)
            else:
                print('%d,-1,-1,-1,-1,-1,-1,-1,-1,-1' % (self.detframe), file=self.det_f)

        """
            associate
        """
        for i in range(len(bbs.bounding_boxes)):
            bbox = bbs.bounding_boxes[i] 
            dets.append(np.array([bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax, bbox.probability]))
        dets = np.array(dets)
        
        trks = np.zeros((len(self.trks_dict), 5))
        for idx, trk in enumerate(trks):
            trk_pos = self.trks_dict[list(self.trks_dict.keys())[idx]]["trkbox"] # 上一帧的跟踪
            trk[:] = [trk_pos[0], trk_pos[1], trk_pos[0]+trk_pos[2], trk_pos[1]+trk_pos[3], self.trks_dict[list(self.trks_dict.keys())[idx]]["id"]]
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_thres)
        self.matched = matched
        self.unmatched_dets = unmatched_dets 
        self.unmatched_trks = unmatched_trks

        # 为新出现的检测框初始化跟踪器
        for n in self.unmatched_dets: 
            trk = cv2.TrackerCSRT_create()
            # trk = cv2.TrackerMOSSE_create()
            trk_age = 0
            if len(self.trks_dict):
                id = max(self.trks_dict.values(), key= lambda x:x['id'])['id'] +1
            else:
                id = 1
            my_roi = (int(dets[n][0]), int(dets[n][1]), int(dets[n][2] - dets[n][0]), int(dets[n][3] - dets[n][1])) # l, t, w, h
            trk_box = my_roi[:]
            # print(f'create track{id} {trk_box}')
            trk.init(im0, my_roi)
            self.trks_dict[f"track{id}"] = {"id":id, "tracker":trk, "age":trk_age, "trkbox":trk_box}

        # 跟踪器失配一次则age+1
        trks_bak = self.trks_dict.copy()
        unmatched_tbak = self.unmatched_trks.copy()
        for i,u in enumerate(sorted(unmatched_tbak, reverse=True)):
            trk = list(trks_bak.keys())[u]
            self.trks_dict[trk]["age"] +=1
            if self.trks_dict[trk]["age"] >= self.max_age:
                self.trks_dict.pop(trk)
                if len(self.unmatched_trks) >1:
                    todel = unmatched_tbak[i]
                    idx = np.where(self.unmatched_trks == todel)
                    self.unmatched_trks = np.delete(sorted(self.unmatched_trks, reverse=True), idx)
                else:
                    self.unmatched_trks = np.array([])

        # 现有全部跟踪器更新
        for trk in self.trks_dict:
            _, trk_box = self.trks_dict[trk]["tracker"].update(im0)
            # print(f'update {trk} trk box {trk_box}')
            self.trks_dict[trk].update({"trkbox":trk_box})

        trks_bak = self.trks_dict.copy()
        match_bak = self.matched.copy()

        for i,m in enumerate(match_bak):
            trk = list(trks_bak.keys())[m[1]]
            id = self.trks_dict[trk]["id"]
            self.trks_dict[trk]["age"] =0
            trk_box = self.trks_dict[trk]["trkbox"]
            
            """
             generate track.txt
            """
            if self.evaluate:
                if trk_box[0] and trk_box[1] != 0: 
                    print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (self.trkframe, id, trk_box[0], trk_box[1], trk_box[2], trk_box[3]), file=self.trk_f)
            # draw
            if self.view_image:
                p1 = (int(trk_box[0]), int(trk_box[1])) # x1, y1, x2, y2
                p2 = (int(trk_box[0] + trk_box[2]), int(trk_box[1] + trk_box[3]))
                cv2.rectangle(im0, p1, p2, (0, 255, 0), 2)
                cv2.putText(im0,'ID-{} '.format(id), (int(trk_box[0]), int(trk_box[1]) - 2), 0, 0.5,
                        [255, 255, 0], thickness=1, lineType=cv2.LINE_AA)

        self.detframe +=1
        self.trkframe +=1

        t2 = time.time()
        
        if self.view_image:    
            cv2.imshow('YoloRT', im0)
            cv2.waitKey(1) 

        # print(t2-t1)
        
    def preprocess(self, im):
        im = cv2.resize(im,(640, 480)) 
        im0 = im.copy()  
        im = np.array([letterbox(im, self.img_size, stride=self.stride, auto=self.pt)[0]])
        # Convert
        im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        im = np.ascontiguousarray(im)
        return im, im0 


    def mk_folder(self, root, train_name, seq_name, txt_type):
        new_train_pth = os.path.join(root, train_name)
        if not os.path.exists(new_train_pth):
            Path(new_train_pth).mkdir() ## 建立train文件夹

        if txt_type == "trk":
            trk_pth = os.path.join(new_train_pth, f"{train_name[:-6]}Track")
            if not os.path.exists(trk_pth):
                Path(trk_pth).mkdir() ## 建立Track文件夹

            data_pth = os.path.join(trk_pth, "data")
            if not os.path.exists(data_pth):
                Path(data_pth).mkdir() ## 建立data文件夹

            # trk_result_pth = os.path.join(data_pth, f"{seq_name}")
            # if not os.path.exists(trk_result_pth):
            #     Path(trk_result_pth).mkdir() ## 建立存放trk.txt

            return data_pth
        
        elif txt_type == "det":
            ## 构造xx-train文件夹
            det_result_pth = os.path.join(new_train_pth, f"{seq_name}")
            if not os.path.exists(det_result_pth):
                Path(det_result_pth).mkdir() ## 建立存放det.txt, gt.txt, seqinfo.ini等的文件夹
            det_pth = os.path.join(det_result_pth, "det")
            if not os.path.exists(det_pth):
                Path(det_pth).mkdir()
            gt_pth = os.path.join(det_result_pth, "gt")
            if not os.path.exists(gt_pth):
                Path(gt_pth).mkdir()

            ## 复制gt.txt和seqinfo.ini至对应的路径
            # all_gt_pth = "/home/fbh/2023_goal/test/Evaluate/trk_gt"
            all_gt_pth = "/home/fbh/2023_goal/kgd_sub/src/used_to_evaluate/trk_gt"
            shutil.copy2(os.path.join(all_gt_pth, seq_name, "gt.txt"), gt_pth)
            shutil.copy2(os.path.join(all_gt_pth, seq_name, "seqinfo.ini"), os.path.join(det_result_pth,"seqinfo.ini"))

            ## 构造seqmaps文件夹
            seqmaps_pth = os.path.join(root, "seqmaps")
            # seqmaps = ["grass", "gym", "npu", "pets", "river"]
            seqmaps = ["car2","car3","npu","war2","war3","war4"]
            seqmaps_file = os.path.join(seqmaps_pth, f"{train_name}.txt")
            with open(seqmaps_file, 'w') as f:
                for seq in seqmaps:
                    f.write(seq + '\n')

            return det_pth
    

    def gen_exp(self, root, seq_name, txt_type):

        existing_exp_f = [
            f for f in Path(root).iterdir() 
            if f.is_dir() and f.name.startswith("exp")
        ]
        
        max_num = max([int(folder.name[3:-6]) for folder in existing_exp_f], default=1)

        result_pth = ""
        train_name = ""
        if (self.mk_new_train):
            train_name = f"exp{max_num+1}-train"
            result_pth = self.mk_folder(root, train_name, seq_name, txt_type)
        else:
            train_name = f"exp{max_num}-train"
            result_pth = self.mk_folder(root, train_name, seq_name, txt_type)

        return result_pth

if __name__ == "__main__":

    rospy.init_node("yolov5", anonymous=True)
    detector = YOLOv5_MultiSOT()
    rospy.spin()
