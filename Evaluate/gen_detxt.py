
import cv2
import numpy as np
import sys
from pathlib import Path
import torch
import os.path

import sys
sys.path.append('/home/fbh/A/Yolo/src/yolov5_ros/src/yolov5')
from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import (check_img_size, non_max_suppression, scale_coords)
from utils.torch_utils import select_device

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
    # print(y)
    return y

def resize(img_folder, save_pth, new_w, new_h):
    idx = 1
    for img in sorted(os.listdir(img_folder)):
        img = cv2.imread(img_folder + img)
        h, w = img.shape[0], img.shape[1]
        # print(w,h)
        h_rate = new_h / h
        w_rate = new_w / w
        img_processing = cv2.resize(img, (0, 0), fx=w_rate, fy=h_rate, interpolation=cv2.INTER_NEAREST)
        img_name = str(idx).zfill(6) + '.jpg'
        cv2.imwrite(save_pth + img_name, img_processing)
        idx += 1

def video2frames(source, frames_savepth):
    """
      如果source路径下是视频文件,则截取视频的每一帧图像,存入文件夹Temp,再利用图片序列进行检测
    """
    cap = cv2.VideoCapture(source)
    c = 1
    frameRate = 1  # 帧数截取间隔（每隔100帧截取一帧）
    
    while(True):
        ret, frame = cap.read()
        if ret:
            if(c % frameRate == 0):
                print("开始截取视频第：" + str(c) + " 帧")
                # 这里就可以做一些操作了：显示截取的帧图片、保存截取帧到本地
                cv2.imwrite(frames_savepth + str(c).zfill(6) + '.jpg', frame)  # 这里是将截取的图像保存在本地
            c += 1
            cv2.waitKey(0)
        else:
            print("所有帧都已经保存在", frames_savepth)
            break
    cap.release()
    source = frames_savepth # 将原本的视频路径 替换为 每帧图片序列的文件夹路径
    
    return source 

def detect(model, device, dataset, save_dir, det_f):
    for path, im, im0s, vid_cap, frame in dataset:
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0~255 -> 0.0~1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        # Inference
        pred = model(im, augment=False, visualize=False)
        # 'pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'
        pred = non_max_suppression(pred, conf_thres=0.3, iou_thres=0.45, classes= 0)  # 只检测人

        # Process predictions
        for i, det in enumerate(pred):  # per image
            im0 = im0s.copy()
            # gn: w h w h
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
            if len(det):    
                for *xyxy, conf, cls in reversed(det):
                    # view(1,4) 增维, view(-1)降维, /gn 归一化
                    xywh = xyxy2ltwh(torch.tensor(xyxy).view(1,4)).view(-1).tolist()
                    print('%d,-1,%.2f,%.2f,%.2f,%.2f,%.2f,-1,-1,-1' % (
                        frame, xywh[0], xywh[1], xywh[2], xywh[3], conf), file=det_f)
            else:
                print('%d,-1,-1,-1,-1,-1,-1,-1,-1,-1' % (frame), file=det_f)
    print(f'results save to {save_dir}')


if __name__ == '__main__':

    # weights = '/home/fbh/A/weights/sliced_visdrone/sliced_visdrone.pt'
    # weights = '/home/fbh/A/weights/v6.2/yolov5m.pt'
    # source = '/home/fbh/A/Dataset/MOT15/train/PETS09-S2L1/img1' # 评估MOT官方数据集
    # source = '/home/fbh/A/Dataset/frames/grassland/' # 评估DIY数据集
    # source = '/home/fbh/A/Dataset/MOT15/train/PETS09-S2L1/img1'
    # source = "/media/fbh/DATA/1Windows2Ubuntu/bag/datasets_327/river2.mp4" # 数据集为视频

    # if ".mp4" in source:
    #     frames_savepth = "/home/fbh/A/Dataset/Temp/" # TODO: 按帧截取视频所得的图片序列的存放位置
    #     source = video2frames(source, frames_savepth)

    # save_pth = '' # TODO
    # resize(source, save_pth, 640, 480) # resize orig frames to the size of 640-w and 480-h

    # device = select_device('0')
    # data = 'data/visdrone.yaml' # TODO
    # bs = 1 # batchsize

    # """""""""""""""""""""""""""""""""""""""""""""""""""
    #                     TODO: 改尺寸

    # """""""""""""""""""""""""""""""""""""""""""""""""""
    # imgsz =  (1920, 1080)

    # model = DetectMultiBackend(weights, device=device, dnn=False, data=data, fp16=False)
    # stride, names, pt = model.stride, model.names, model.pt
    # imgsz = check_img_size(imgsz, s = stride)  # check image size
    # dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    # model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup

    # """""""""""""""""""""""""""""""""""""""""""""""""""
    #                     TODO: 改名

    # """""""""""""""""""""""""""""""""""""""""""""""""""
    # save_dir = '/home/fbh/A/Evaluate/detector_results/'
    # det_f = open(save_dir + 'det_riverr_ori.txt', 'w') # name of txt file to save
    # detect(model, device, dataset, save_dir, det_f)
    video2frames('/media/fbh/DATA/1Windows2Ubuntu/bag/datasets_April/npu/nnppuu.mp4', '/home/fbh/A/Yolo/src/yolov5_ros/src/datasets/ChangAn_nwpu/val/images/')