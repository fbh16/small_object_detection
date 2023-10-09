# 如何运行SORT跟踪器
New Terminal（启动检测节点）:

    cd kgd
    source devel/setup.bash
    roslaunch yolov5_ros detect.launch

New Terminal（启动跟踪节点）:

    cd kgd
    source devel/setup.bash
    roslaunch sort_ros sort.launch

New Terminal（当两个终端都准备好后）:

    rosbag play xx.bag

sort.launch文件参数：

**display**：是否可视化跟踪过程

**iou_threshold**：卡尔曼滤波器预测的预测框与检测框的交并比阈值
