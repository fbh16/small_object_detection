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
from utils.augmentations import letterbox
from models.common import DetectMultiBackend
from utils.plots import Annotator, colors
from utils.torch_utils import select_device
from utils.general import (check_img_size,non_max_suppression)


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

@torch.no_grad()
class YOLOv5_MultiSOT:
    def __init__(self):
        bs = 1  # batch size
        self.detframe = 1
        self.trkframe = 0
        self.max_age = 5
        self.max_det = 100
        self.iou_thres = 0.3  # track
        self.iou_thresh = 0.6 # det
        self.conf_thres = 0.6
        self.half = False
        self.evaluate = rospy.get_param("~evaluate")
        cudnn.benchmark = True # set True to speed up constant image size inference
        self.agnostic_nms = True
        self.trks_dict = {}
        self.matched = []
        self.unmatched_trks = []
        self.bridge = CvBridge()
        self.img_size = [480,640]
        weights = rospy.get_param("~weights", '/home/fbh/2023_goal/kgd/src/weights/sliced_visdrone/sliced_visdrone.engine')
        
        self.single_cls = rospy.get_param("~classes","")
        cls_lst=['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']
        if self.single_cls != "":
            # self.classes = self.single_cls.split(",")
            self.classes = self.single_cls
            print(cls_lst[self.classes])
        else:
            self.classes = None

        
        self.view_image = rospy.get_param("~view_image", True)
        self.hide_label = rospy.get_param("~hide_label", False)
        self.device = select_device(str(rospy.get_param("~device","0")))
        self.model = DetectMultiBackend(
                            weights, 
                            device=self.device, 
                            dnn=False, 
                            data=rospy.get_param("~data",'/home/fbh/2023_goal/kgd/src/yolov5_ros/src/yolov5/data/VisDrone.yaml')
                    )
        self.stride, self.names, self.pt, self.engine = (self.model.stride,self.model.names,self.model.pt,self.model.engine)
        self.img_size = check_img_size(self.img_size, s=self.stride)
        # self.half &= self.engine and self.device.type != "cpu"  # FP16 supported on limited backends with CUDA
        self.model.warmup(imgsz=(1 if self.pt else bs, 3, *self.img_size), half=self.half)  
        # topic
        input_image_type, input_image_topic, _ = get_topic_type(rospy.get_param("~input_image_topic", '/usb_cam/image_raw'), blocking = True)        
        self.img_sub = rospy.Subscriber(input_image_topic, Image, self.callback, queue_size=1)
        self.pred_pub = rospy.Publisher(rospy.get_param("~output_topic", "/yolo/box"), BoundingBoxes, queue_size=1)
        self.img_pub = rospy.Publisher(rospy.get_param("~output_img", "/yolo/img"), Image, queue_size=1)
        print('yolo is ready')

        self.mk_new_train = rospy.get_param("~mk_new_train")
        if self.evaluate:
            seq_name = rospy.get_param("~seq_name")
            det_root = "/home/fbh/2023_goal/test/Evaluate/TrackEval/data/gt/mot_challenge"
            det_result_pth = self.gen_exp(det_root, seq_name)
            print(det_result_pth)
            self.det_f = open(os.path.join(det_result_pth, f'det.txt'), 'w') 


    def callback(self, data):

        self.img_pub.publish(data)
        # img = np.frombuffer(data.data, dtype=np.uint8).reshape(data.width, data.height, -1) 
        img = self.bridge.imgmsg_to_cv2(data, "bgr8")
        im, im0 = self.preprocess(img) #im0.shape=()
        im = torch.from_numpy(im).to(self.device) 
        im = im.half() if self.model.fp16 else im.float()
        im /= 255 # 归一化
        if len(im.shape) == 3:
            im = im[None]
        t1 = time.time() 
        pred = self.model(im, augment=False, visualize=False) # 检测
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thresh, self.classes, self.agnostic_nms, max_det=self.max_det)
        # Process predictions 
        det = pred[0].cpu().numpy() # resize后的坐标
        bbs = BoundingBoxes()
        bbs.header = data.header
        bbs.image_header = data.header
        annotator = Annotator(im0, line_width=2.5, example=str(self.names)) 

        if len(det):
            det[:,:4] = tran2rawsz(det[:,:4], img.shape[1], img.shape[0])
            for *xyxy, conf, cls in reversed(det):
                bb = BoundingBox()
                c = int(cls)
                bb.Class = self.names[c]
                bb.probability = conf 
                bb.xmin = int(xyxy[0]) 
                bb.ymin = int(xyxy[1])
                bb.xmax = int(xyxy[2])
                bb.ymax = int(xyxy[3])
                bbs.bounding_boxes.append(bb)
                if self.view_image:  # Add bbox to image
                    if self.hide_label:
                        label = False
                    else:
                        label = f"{self.names[c]} {conf:.2f}"
                    annotator.box_label(xyxy, label, color=(255,0,0))       
                    im0 = annotator.result()
        self.pred_pub.publish(bbs) # 发的bbox在原图上的坐标
        
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
        self.detframe +=1
        print(self.detframe)

        t2 = time.time()

        if self.view_image:    
            cv2.imshow('YoloRT', im0)
            cv2.waitKey(1) 


    def preprocess(self, im):
        im0 = im.copy()
        im = cv2.resize(im,(640, 480))   
        im = np.array([letterbox(im, self.img_size, stride=self.stride, auto=self.pt)[0]])
        # Convert
        im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        im = np.ascontiguousarray(im)
        return im, im0 


    def mk_folder(self, root, train_name, seq_name):
        new_train_pth = os.path.join(root, train_name)
        if not os.path.exists(new_train_pth):
            Path(new_train_pth).mkdir() ## 建立train文件夹
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
        all_gt_pth = "/home/fbh/2023_goal/kgd_sub/src/used_to_evaluate/trk_gt"
        shutil.copy2(os.path.join(all_gt_pth, seq_name, "gt.txt"), gt_pth)
        shutil.copy2(os.path.join(all_gt_pth, seq_name, "seqinfo.ini"), os.path.join(det_result_pth,"seqinfo.ini"))

        ## 构造seqmaps文件夹
        seqmaps_pth = os.path.join(root, "seqmaps")
        # seqmaps = ["grass", "gym", "npu", "pets", "river"]
        seqmaps = ["grass", "gym", "river", "car3","car6","war2","war3","war4", ]
        seqmaps_file = os.path.join(seqmaps_pth, f"{train_name}.txt")
        with open(seqmaps_file, 'w') as f:
            for seq in seqmaps:
                f.write(seq + '\n')
        return det_pth


    def gen_exp(self, root, seq_name):

        existing_exp_f = [
            f for f in Path(root).iterdir() 
            if f.is_dir() and f.name.startswith("exp")
        ]
        max_num = max([int(folder.name[3:-6]) for folder in existing_exp_f], default=1)
        result_pth = ""
        train_name = ""
        if (self.mk_new_train):
            train_name = f"exp{max_num+1}-train"
            result_pth = self.mk_folder(root, train_name, seq_name)
        else:
            train_name = f"exp{max_num}-train"
            result_pth = self.mk_folder(root, train_name, seq_name)

        return result_pth


if __name__ == "__main__":
    # if len(sys.argv) > 1:
    #     ifeval = sys.argv[1].lower() == 'true' ## 是否生成det.txt
    # else:
    #     ifeval = False
    rospy.init_node("yolov5_gendet", anonymous=True)
    detector = YOLOv5_MultiSOT()
    rospy.spin()
