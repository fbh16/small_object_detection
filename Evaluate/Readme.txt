usage of mot_benchmark

1.使用gen_detxt检测Dataset/MOT15/train/seq_to_evaluate/img1中的图片序列，生成det.txt，该txt存储在detector_results中。
2.使用sort.py对det.txt跟踪并生成跟踪结果SortTrack.txt，该文件存储在tracker_results中
	python3 sort.py --seq_path detecor_results/det.txt      (默认max_age=5, min_hits=4)
3.使用TrackEval/scripts中的run_mot_challenge.py对跟踪结果进行评估，在这之前，
先将（1）生成的det.txt复制到/home/fbh/A/Evaluate/TrackEval/data/gt/mot_challenge/MOT15-train/seq_to_eval/det/下，重命名为det.txt。
再将（2）生成的SortTrack.txt复制到/home/fbh/A/Evaluate/TrackEval/data/trackers/mot_challenge/MOT15-train/SortTrack/data/下，重命名为PETS09-S2L1.txt
最后执行命令： 
	python3 scripts/run_mot_challenge.py --BENCHMARK MOT15 --TRACKERS_TO_EVAL SortTrack  （可以保存结果至TXT）

对于自己制作的数据集：
1.用darklabel标定图片序列生成label文件
2.用frameplus1.py对darklabel生成的label进行调整，第一列（idx=0）和第二列（idx=1）索引从1开始，还有第8列（idx=7）加2（全为正1），至此该label文件就是sort的gt.txt
3.用所训练的模型，对图片序列进行检测（用gen_detxt.py），生成det.txt
4.将det.txt输入sort.py，生成跟踪的label文件result.txt(该txt可自定义命名，但名字必须和trackEval中的tracker文件名一致)
5.使用TrackEval/scripts中的run_mot_challenge.py对跟踪结果进行评估，在这之前，
先将gen_detxt生成的det.txt复制到/home/fbh/A/Evaluate/TrackEval/data/gt/mot_challenge/nwpu/seq_to_eval/det/下，重命名为det.txt。
再将darklabel标注生成的文件复制到/home/fbh/A/Evaluate/TrackEval/data/gt/mot_challenge/nwpu/seq_to_eval/gt下，重命名为gt.txt。
再将sort.py生成的SortTrack.txt复制到/home/fbh/A/Evaluate/TrackEval/data/trackers/mot_challenge/nwpu/SortTrack/data/下，重命名为nwpu.txt
最后执行命令： 
	python3 scripts/run_mot_challenge.py --BENCHMARK nwpu

