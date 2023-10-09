# 如何启动yolov5检测节点与csrt(c++)跟踪节点

## 1. 启动yolov5检测节点
Terminal1:

    roslaunch yolov5_ros detect.launch

Terminal2:  

    rosbag play xx.bag

**或者**：

Terminal1：  

    roscore

Terminal2（注：要将yolo_gendet.py中的～weights和～data的路径对应正确）：  

    cd kgd
    source devel/setup.bash
    rosrun yolov5_ros yolo_gendet.py

Terminal3:  
    
    rosbag play xx.bag  


launch文件参数：  
**weights**：模型的绝对路径  
**data**： 数据集对应的yaml文件的相对路径  
**confidence_threshold** ：置信度阈值  
**iou_threshold**：交并比阈值  
**maximum_detections**：最大检测数量  
**classes**：单一类别检测，数字为对应类别的索引，可在yaml文件中查看  
**view_image**: 是否可视化检测过程  
**hide_label**: 是否隐藏检测框的标签  
**hide_conf**： 是否隐藏检测框的置信度  

## 2. 启动csrt跟踪器（c++）节点
New Terminal:

    cd kgd 
    source devel/setup.bash
    rosrun sot track

# 如何启动yolov5检测器与CSRT跟踪器（python）
    roslaunch yolov5_ros mot.launch 
