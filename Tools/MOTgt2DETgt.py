import numpy as np

def ltwh2xywh(str_ltwh, imw, imh):
    """
        inputs: ltwh, the lt is left-top of bbox
        output: xywh, the xy is centre of bbox
    """
    ltwh = []
    # imw = 1916 # 图像序列的尺寸
    # imh = 1080
    for i,v in enumerate(str_ltwh): 
        v = float(v)
        ltwh.append(v)
    # 格式转换和归一化
    xc = round(((ltwh[2] + 0.5*ltwh[4]) / imw), 6)
    yc = round(((ltwh[3] + 0.5*ltwh[5]) / imh), 6)
    bw = round((ltwh[4] / imw), 6)
    bh = round((ltwh[5] / imh), 6)
    return ltwh[:2] + [xc,yc,bw,bh]

def process_dettxt(file, imw, imh): 
    """
        input :  gt.txt: <frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<conf>,<x>,<y>,<z>
        output: a numpy array like [[1, -1, 500, 158, 530.98, 228.3], [frame_id, clsid, xc, yc, w, h]...]
    """
    det_gt = []
    f = open(file,"r")
    data = f.readlines()

    for _, line in enumerate(data):
        l = line.strip().split(",")
        # print(l)
        # l = ['795', '8', '217', '157', '25.613', '68.992', '1', '-6.279', '-0.33723', '0']
        ididxyxy = ltwh2xywh(l[:6], imw, imh)
        # print(ididxyxy)
        det_gt.append(ididxyxy)
    # print(det_gt)
    det_gt = np.array(det_gt)
    # print(det_gt)
    slice_index = []
    slice_lst = []
    # 取帧数发生变化时的索引
    for i,v in enumerate(det_gt[:,0]):
        if i < det_gt.shape[0]-1 and det_gt[i,0] != det_gt[i+1,0]:
            slice_index.append(i+1)
    slice_index.append(det_gt.shape[0]-1)
    slice_index.append(0)
    slice_index.sort()
    # 按不同帧数进行切片
    for i in range(len(slice_index)-1):
        slice_lst.append(det_gt[slice_index[i] : slice_index[i+1]])
    # 把det中漏掉的最后一行加到slice list的最后一个数组中
    last_det = np.expand_dims(det_gt[-1],axis=0)
    slice_lst[-1] = np.append(slice_lst[-1], last_det, axis=0)
    # print(slice_lst)
    return slice_lst

def processline(line):
    newline = []
    line.pop(0)
    line.pop(0)
    for k,num in enumerate(line):
        if '.' in num and k < len(line)-1:
            line.insert(k+1, ' ')
    for i,v in enumerate(line[:3]):
        if '.' in v:
            l = v[:-2]
            newline.append(l)
    for i,v in enumerate(line[3:]):
        newline.append(v)
    
    newline.insert(0, '0')
    newline.insert(1,' ')
    newline.insert(3, ' ')

    newline += '\n'
    print(newline)
    return newline

if __name__ == '__main__':

    #path = '/media/fbh/DATA/Windows2Ubuntu/bag/datasets_327/labels/grassland.txt'
    path = '/media/fbh/DATA/1Windows2Ubuntu/bag/datasets_April/410/gt/river_gt.txt'
    save_pth = '/media/fbh/DATA/1Windows2Ubuntu/bag/datasets_April/410/det_gt/river_det_gt/'
    imw = 1920
    imh = 1080
    lst = process_dettxt(path, imw, imh)
    """
        lst = [array([],[],...), array([],..),...]
    """
    print(lst)
    for i,v in enumerate(lst,1):
        # v[:,:2].astype(int)
        frame_data = v.astype(str).tolist()
        for _,line in enumerate(frame_data):
            # line += '\n'
            newline = processline(line)
            f = open(save_pth + str(i).zfill(6)+'.txt', 'a')
            f.writelines(newline)
    f.close()
    
