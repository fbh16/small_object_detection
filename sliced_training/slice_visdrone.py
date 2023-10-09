import os
import cv2
import random
import numpy as np
from PIL import Image
from pathlib import Path
from shapely.geometry import Polygon


"""
  yolov5 bbx type: class_id center_x center_y width height (Normalized)
  sliced bbx type: xmin, ymin, xmax, ymax (Unnormalized)
"""

def slice_bbxs(image_height, image_width, slice_height, slice_width, overlap_height_ratio, overlap_width_ratio):
    slice_bboxes = []
    y_max = y_min = 0
    y_overlap = int(overlap_height_ratio * slice_height)
    x_overlap = int(overlap_width_ratio * slice_width)

    while y_max < image_height:
        x_min = x_max = 0
        y_max = y_min + slice_height
        while x_max < image_width:
            x_max = x_min + slice_width
            if y_max > image_height or x_max > image_width:
                xmax = min(image_width, x_max)
                ymax = min(image_height, y_max)
                xmin = max(0, xmax - slice_width)
                ymin = max(0, ymax - slice_height)
                slice_bboxes.append([xmin, ymin, xmax, ymax])
            else:
                slice_bboxes.append([x_min, y_min, x_max, y_max])
            x_min = x_max - x_overlap
        y_min = y_max - y_overlap
    return slice_bboxes


def bbx_in_slice(bbx, slice_bbx, img_w, img_h):
    cx, cy, w, h = bbx
    
    if not slice_bbx[0] < cx*img_w < slice_bbx[2]:
        return False
    if not slice_bbx[1] < cy*img_h < slice_bbx[3]:
        return False
    if not w*img_w < (slice_bbx[2]-slice_bbx[0]):
        return False
    if not h*img_h < (slice_bbx[3]-slice_bbx[1]):
        return False

    return True

def condition(new_bbx, ori_bbx, ori_img_w, ori_img_h, new_img_w, new_img_h):
    ## TODO
    ## 将new_img中的bbx再转换回ori_img,再计算IOU
    ##
    ori_S = (ori_bbx[2] * ori_bbx[3]) / (ori_img_w * ori_img_h)
    new_S = (new_bbx[2] * new_bbx[3]) / (new_img_w * new_img_h)
    # print(ori_S,' ',new_S)
    if new_S/ori_S > 0.5: ## 切片后bbx的面积小于原图的对应bbx面积的50%滤除
        return True
    return False


def validate(validate_num, cls_lst, sliced_imgs_pth, sliced_labels_pth, output_dir):
    
    color_dict = {
        cls_lst[0]: (0, 0, 255),
        cls_lst[1]: (255, 0, 0),
        cls_lst[2]: (0, 255, 0),
        cls_lst[3]: (255, 255, 0),
        cls_lst[4]: (255, 0, 255),
        cls_lst[5]: (0, 255, 255),
        cls_lst[6]: (75, 0, 130),
        cls_lst[7]: (0, 100, 0),
        cls_lst[8]: (139, 69, 19),
        cls_lst[9]: (148, 0, 211)
    }
    if len(color_dict) != len(cls_lst):
        print("bbxs' color is not enough, plz update the color dict!")

    # for img in os.listdir(sliced_imgs_pth):
    for img in os.listdir(sliced_imgs_pth)[:validate_num]:
        label_name = img[:-4]+'.txt'
        with open(os.path.join(sliced_labels_pth, label_name), 'r') as f:
            lines = f.readlines()

        im_show = cv2.imread(os.path.join(sliced_imgs_pth, img))
        
        for line in lines:
            values = line.strip().split() 
            class_index = int(values[0])
            center_x, center_y, width, height = map(float, values[1:])
            # calculate left-top and right-bottom coordinates
            x1 = int((center_x - width/2) * im_show.shape[1])
            y1 = int((center_y - height/2) * im_show.shape[0])
            x2 = int((center_x + width/2) * im_show.shape[1])
            y2 = int((center_y + height/2) * im_show.shape[0])
            # draw boxes
            # color = (0, 255, 0)  # BGR颜色，这里使用绿色
            label = cls_lst[class_index]
            color = color_dict[label]
            thickness = 2  # 框的线宽
            cv2.rectangle(im_show, (x1, y1), (x2, y2), color, thickness)
            # label = f"Class: {class_index}"
            cv2.putText(im_show, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
        
        result_pth = os.path.join(output_dir, 'validate')
        if not os.path.exists(result_pth):
            os.mkdir(result_pth) 

        cv2.imwrite(os.path.join(result_pth, img), im_show)


def slice_imgs (image, img_name, root, output_dir, 
                slice_h, slice_w, overlap_h_ratio, overlap_w_ratio, classes, val):
    
    img = Image.open(image)
    img_w, img_h = img.size

    sliced_imgs_pth = os.path.join(output_dir, 'sliced_images')
    if not os.path.exists(sliced_imgs_pth):
        os.mkdir(sliced_imgs_pth)

    slice_bboxes = slice_bbxs(img_h, img_w, slice_h, slice_w, overlap_h_ratio, overlap_w_ratio)

    image_pil_arr = np.asarray(img)

    n_ims = 0
    for slice_bbx in slice_bboxes:
        n_ims += 1
        tlx = slice_bbx[0] ## xmin
        tly = slice_bbx[1] ## ymin
        brx = slice_bbx[2] ## xmax
        bry = slice_bbx[3] ## ymax
        slice_img = image_pil_arr[tly:bry, tlx:brx]
        # slice_suffixes = "_".join(map(str, slice_bbx))
        # slice_img_name = f"{img_name[:-4]}_{slice_suffixes}.jpg"
        slice_img_name = f"{img_name[:-4]}_{n_ims}.jpg"
        export_img = Image.fromarray(slice_img)
        export_img.save(os.path.join(output_dir, 'sliced_images', slice_img_name))
    
    ###########
    ########
    label_name = img_name[:-4] + '.txt'
    with open(os.path.join(root, 'labels', label_name), "r") as label:
        label_content = label.readlines()    

    for piece_id, slice_bbx in enumerate(slice_bboxes,1):

        sliced_labels_pth = os.path.join(output_dir, 'sliced_labels')
        if not os.path.exists(sliced_labels_pth):
            os.mkdir(sliced_labels_pth)

        sliced_labels_name = f"{img_name[:-4]}_{piece_id}.txt"
        new_label = open(os.path.join(sliced_labels_pth, sliced_labels_name), "w")
        # print('*'*20)
        for line in label_content:
            line = line.strip().split()
            cls_id = line[0]
            ori_bbx = list(map(float, line[1:])) ## cx, cy, w, h (based on orig image and normalized)
            if bbx_in_slice(ori_bbx, slice_bbx, img_w, img_h):
                new_bbx = [
                    (ori_bbx[0]*img_w - slice_bbx[0])/slice_w,
                    (ori_bbx[1]*img_h - slice_bbx[1])/slice_h,
                    (ori_bbx[2]*img_w)/slice_w,
                    (ori_bbx[3]*img_h)/slice_h,
                ]
                new_bbx= [round(num, 4) for num in new_bbx]

                # if condition(new_bbx, ori_bbx, img_w, img_h, slice_w, slice_h):
                #     round_bbx = [round(num, 3) for num in new_bbx]

                new_bbx.insert(0, cls_id)
                # print(new_bbx)
                new_label.write(' '.join(map(str, new_bbx)) + '\n')
        new_label.close()
    
    print(f"results saved in {output_dir}")

    # gen_clstxt = (False, ['car', 'person', 'truck'])
    # if gen_cls[0]:    
    #     cls_list = gen_cls[1]
    #     gen_cls_txt(output_dir, cls_list)

    if val and len(classes)>0:
        ## 6 is the number of images to validate
        validate(6, classes, sliced_imgs_pth, sliced_labels_pth, output_dir)


def gen_cls_txt(pth, cls_list):

    with open(os.path.join(pth, 'sliced_labels', 'classes.txt'), "w") as f:
        for cls in cls_list:
            f.write(cls + '\n')


def gen_exp(root):
    if not os.path.exists(os.path.join(root,'runs')):
        os.makedirs(os.path.join(root,'runs'))

    existing_exp_f = [
        f for f in Path(os.path.join(root,'runs')).iterdir() 
        if f.is_dir() and f.name.startswith("exp")
    ]
    max_num = max([int(folder.name[3:]) for folder in existing_exp_f], default=0)
    new_exp_num = max_num + 1
    new_exp_name = f"exp{new_exp_num}"
    new_exp_pth = Path(os.path.join(root,'runs')) / new_exp_name
    new_exp_pth.mkdir()
    # print(new_exp_pth)
    return new_exp_pth


if __name__ == "__main__":    

    ## path to your folder
    root = '/home/fbh/2023_goal/kgd/src/Dataset'
    
    if not os.path.exists(root):
        os.mkdir(root)
    
    ## generate "runs/exp{i}" directory which saves result
    exp_pth = gen_exp(root)

    ## parameters to set
    slice_width = 640
    slice_height = 480
    overlap_h_ratio = 0.1
    overlap_w_ratio = 0.1
    val = True ## whether validate results  
    classes = ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']
    
    for d in 'VisDrone2019-DET-train', 'VisDrone2019-DET-val', 'VisDrone2019-DET-test-dev':
      root_dir = Path(root) / d
      ## slice images and labels
      for f in os.listdir(os.path.join(root_dir, 'images')):
          if f.endswith('.jpg'):
              img = os.path.join(root_dir, 'images', f)
              img_name = f
              slice_imgs (img, img_name, root_dir, exp_pth, 
                          slice_height, slice_width, overlap_h_ratio, overlap_w_ratio, classes, val)






