import os
"""
    使用darklabel标注的数据集生成的标签, 索引是从0开始的,用以下脚本将索引改为从1开始
"""

# darklabel生成的标签文件路径
# path = "/home/fbh/A/Evaluate/TrackEval/data/gt/mot_challenge/kcf-train/ChangAn_nwpu/gt/"
path = "/home/fbh/A/colo/src/pose_test/src/evaluate/gt/"
# 保存路径
# save_pth = "/home/fbh/A/Evaluate/TrackEval/data/gt/mot_challenge/kcf-train/ChangAn_nwpu/gt/"
save_pth = "/home/fbh/A/colo/src/pose_test/src/evaluate/gt/"

for txt in sorted(os.listdir(path)):
    f = open(path + txt, "r")
    data = f.readlines()
    save_f = open(save_pth + 'gt_new' +'.txt', 'a')
    for _, line in enumerate(data):
        l = line.strip().split(",")
        # l[0] = str(int(l[0]) + 1) # frame (1-based)
        l[1] = str(int(l[1]) + 1) # id (1-based)
        # l[7] = str(int(1))
        # l[-1] = str(int(-1))
        # l[-2] = str(int(-1))
        # l[-3] = str(int(1))
        # l[-4] = str(int(-1))
        # print(l)
        for i in range(len(l) + len(l) - 2):
            if i % 2 == 0:
                l.insert(i+1, ',')
        l += '\n'
        save_f.writelines(l)
    save_f.close()
