# 下载VisDrone2019数据集   
进入
 <https://github.com/VisDrone/VisDrone-Dataset> 
 下载  
 '**VisDrone2019-DET-train**',   
 '**VisDrone2019-DET-val**',   
 '**VisDrone2019-DET-test-dev**'。

# 运行VisDrone2yolo.py
**无需解压**三个数据集压缩包！  
代码运行完成后会将标签文件格式转为yolo格式，并在  '**VisDrone2019-DET-train**',   
'**VisDrone2019-DET-val**',  
'**VisDrone2019-DET-test-dev**'  
文件夹下增加**labels**文件夹，即yolo格式的标签。

# 运行slice_visdrone.py
修改**root**路径为三个VisDrone文件夹所在根目录绝对路径。代码运行结束后，会在根目录建立**runs**文件夹，该文件夹下有切片后的图片文件夹**sliced_images**及对应标签文件夹**sliced+labels**，还有用于验证图片与标签对应关系的**validate**文件夹。

# 用切片数据集训练YOLOv5检测模型
