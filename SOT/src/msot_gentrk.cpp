#include <map>
#include <set>
#include <tuple>
#include <vector>
#include <cfloat>
#include <fstream>

#include <iostream>
#include <ros/ros.h>
#include <algorithm>
#include "Hungarian.h"
#include <cv_bridge/cv_bridge.h>

#include "opencv2/opencv.hpp"
#include "opencv2/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "sensor_msgs/Image.h"
#include "sensor_msgs/image_encodings.h"
#include <darknet_ros_msgs/BoundingBox.h>
#include "darknet_ros_msgs/BoundingBoxes.h"
#include <image_transport/image_transport.h>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
using namespace cv;
using namespace std;
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#include "rosbag/bag.h"
#include "rosbag/view.h"

double GetIOU(Rect2d bb_test, Rect2d bb_gt)
{
    float in = (bb_test & bb_gt).area();           // 交集
    float un = bb_test.area() + bb_gt.area() - in; // 并集
    if (un < DBL_EPSILON)                          // 避免分母为0
        return 0;
    return (double)(in / un);
}

tuple<vector<pair<int, int>>, set<int>, set<int>> associate_dets_to_trks(vector<Rect2d> dets,
                                                                         vector<Rect2d> trks, float iou_threshold = 0.3)
{
    int trkNum = trks.size();
    int detNum = dets.size();

    if (trks.size() == 0)
    {
        vector<pair<int, int>> match(0, make_pair(0, 0));
        set<int> unmtrk;
        set<int> unmdet;
        for (int i = 0; i < dets.size(); i++)
            unmdet.insert(i);
        tuple<vector<pair<int, int>>, set<int>, set<int>> result(match, unmdet, unmtrk);
        return result;
    }

    vector<vector<double>> iouMatrix;
    iouMatrix.resize(trkNum, vector<double>(detNum, 0));
    for (int t = 0; t < trkNum; t++)
    {
        for (int d = 0; d < detNum; d++)
        {
            iouMatrix[t][d] = 1 - GetIOU(trks[t], dets[d]);
        }
    }
    vector<int> assignment;
    HungarianAlgorithm HungAlgo;
    if (min(iouMatrix.size(), iouMatrix[0].size()) > 0)
    {
        HungAlgo.Solve(iouMatrix, assignment); // assignment[i] 表示第 i 个检测框的匹配结果
    }
    set<int> allDets;
    set<int> allTrks;
    set<int> mDets;
    set<int> mTrks;
    set<int> umDets;
    set<int> umTrks;
    
    for (int d = 0; d < dets.size(); d++)
        allDets.insert(d);

    for (int t = 0; t < trks.size(); t++)
        allTrks.insert(t);
    
    if (assignment.size() > 0) {
        for (int i = 0; i < trks.size(); i++)
        {
            if (assignment[i] != -1)
            {
                mDets.insert(assignment[i]);
                mTrks.insert(i);
            }
        }
    }
    set_difference (allDets.begin(), allDets.end(),
                    mDets.begin(), mDets.end(), insert_iterator<set<int>>(umDets, umDets.begin()));

    set_difference (allTrks.begin(), allTrks.end(),
                    mTrks.begin(), mTrks.end(), insert_iterator<set<int>>(umTrks, umTrks.begin()));

    vector<pair<int, int>> mPairs;
    for (int i = 0; i < assignment.size(); i++)
    {
        if (assignment[i] == -1)
            continue;

        if (1 - iouMatrix[i][assignment[i]] < iou_threshold)
        {
            umDets.insert(assignment[i]);
            umTrks.insert(i);
        } else {
            mPairs.push_back(make_pair(i, assignment[i])); // matches(trk, det)
        }
    }
    tuple<vector<pair<int, int>>, set<int>, set<int>> associateResult(mPairs, umDets, umTrks);

    return associateResult;
}

Rect2d adjustBbx(Rect2d bbox, cv_bridge::CvImagePtr cv_ptr)
{
    if (bbox.x <0) {
        bbox.width = bbox.width + bbox.x;
        bbox.x = 0;
    } 
    else if (bbox.y <0) {
        bbox.height = bbox.height + bbox.y;
        bbox.y = 0;
    } 
    else if ((bbox.x+bbox.width) > cv_ptr->image.cols)
    {
        bbox.width = cv_ptr->image.cols;
    }
    else if ((bbox.y+bbox.height) > cv_ptr->image.rows) 
    {
        bbox.height = cv_ptr->image.rows;
    }

    return bbox;
}

class SoTracker
{
public:
    SoTracker(const std::string& expPath, const std::string& bagName, bool viewIMG, bool Evaluate, int maxAge, int minHits);
    void callback(const sensor_msgs::Image::ConstPtr &img, const darknet_ros_msgs::BoundingBoxes::ConstPtr &msg);

private:
    ros::NodeHandle nh;
    image_transport::Publisher pub;
    int trkFrame = 1;
    int minHits; // 2
    int maxAge; // 7
    int thickness = 2;
    float iou_thresh = 0.3;
    bool initialized = false;
    struct trkinfo
    {
        int trkID;
        // Ptr<cv::TrackerCSRT> trk;
        Ptr<cv::TrackerKCF> trk;
        int trkAge;
        Rect2d trkROI;
    };
    map<string, trkinfo> trksMap;
    string expPath;
    string bagName;
    ofstream outFile;
    bool Evaluate;
    bool viewIMG;
    int globalMaxID = 1;
};

typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, darknet_ros_msgs::BoundingBoxes> SyncPolicy;

SoTracker::SoTracker(
        const std::string& exp_path, 
        const std::string& bag_name, 
        bool view_img, bool evaluate,
        int max_age, int min_hits
    ): 
        expPath(exp_path), 
        bagName(bag_name), 
        viewIMG(view_img), 
        Evaluate(evaluate),
        maxAge(max_age),
        minHits(min_hits) 
{
    message_filters::Subscriber<sensor_msgs::Image> img_sub(nh, "/yolo/img", 10);
    message_filters::Subscriber<darknet_ros_msgs::BoundingBoxes> yolo_sub(nh, "/yolo/bbx", 10);
    message_filters::Synchronizer<SyncPolicy> sync(SyncPolicy(10), img_sub, yolo_sub);
    sync.registerCallback(boost::bind(&SoTracker::callback, this, _1, _2));
    ros::spin();
}

void SoTracker::callback(const sensor_msgs::ImageConstPtr &img, const darknet_ros_msgs::BoundingBoxes::ConstPtr &msg)
{
    clock_t startTime = clock();

    if (!initialized && Evaluate) 
    {
        string filename = bagName+".txt";
        outFile.open(expPath + '/' + filename);
        initialized = true;
    }
    int bbs_num = msg->bounding_boxes.size();
    cv_bridge::CvImagePtr cv_ptr;
    cv_ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::BGR8);

    // cout << cv_ptr->image.rows << "*" << cv_ptr->image.cols << endl;
    
    vector<Rect2d> dets;
    for (size_t i = 0; i < bbs_num; i++)
    {
        const auto &bb_msg = msg->bounding_boxes[i];
        Rect2d det(bb_msg.xmin, bb_msg.ymin, bb_msg.xmax - bb_msg.xmin, bb_msg.ymax - bb_msg.ymin); // l,t,w,h
        dets.push_back(det);
    }

    vector<Rect2d> trks;
    for (auto map_it = trksMap.begin(); map_it != trksMap.end(); map_it++)
    {
        trks.push_back(map_it->second.trkROI);
    }
    
    // Hungarian & IOU match
    auto result = associate_dets_to_trks(dets, trks, iou_thresh);
    vector<pair<int, int>> mPairs = get<0>(result);
    set<int> umDets = get<1>(result);
    set<int> umTrks = get<2>(result);

    // Init new tracker for umDets.
    int trkID = 0;
    int max_id = 0;
    for (int ud : umDets)
    {
        // Ptr<TrackerCSRT> trk = TrackerCSRT::create();
        Ptr<TrackerKCF> trk = TrackerKCF::create();
        int trkAge = 0;
        if (!trksMap.empty())
        {
            for (const auto &pair : trksMap)
            {
                int current_id = pair.second.trkID;
                max_id = std::max(max_id, current_id);
            }
            // 找出整个序列中，当前为止的历史最大id：globalMaxID
            globalMaxID = std::max(globalMaxID, max_id);
            trkID = globalMaxID +1;
            cout << "global max ID: " << globalMaxID << "    " << " trkID: " << trkID <<endl;
        } else {
            trkID = 1;
        }
        Rect2d trkROI(dets[ud].x, dets[ud].y, dets[ud].width, dets[ud].height);
        trk->init(cv_ptr->image, trkROI);
        trkinfo trkInfo = {trkID, trk, trkAge, trkROI};
        trksMap.insert(pair<string, trkinfo>("trk" + to_string(trkID), trkInfo));
    }
    
    // Delete matching failed tracker.
    vector<int> trkid_list;
    set<int> umTrks_bak = umTrks;
    for (auto map_it = trksMap.begin(); map_it != trksMap.end(); ++map_it)
    {
        trkid_list.push_back(map_it->second.trkID);
    }
    for (auto utb = umTrks_bak.rbegin(); utb != umTrks_bak.rend(); utb++)
    {
        int id = trkid_list[*utb]; //*ut表示umTrks中的元素
        trksMap["trk" + to_string(id)].trkAge += 1;
        if (trksMap["trk" + to_string(id)].trkAge >= maxAge)
        {
            // 从map中删除过期的轨迹
            auto map_it = trksMap.find("trk" + to_string(id));
            int deltrk_idx = distance(trksMap.begin(), map_it);
            map_it = trksMap.erase(map_it); 
            // cout << "delete " << "trk" + to_string(id) << endl;
            // 删除umTrks中过期轨迹的索引
            if (umTrks.size() > 1) 
            {
                umTrks.erase(deltrk_idx);
            } else {
                umTrks.clear();
            }
            // 更新mPairs中轨迹的索引
            for (auto it = mPairs.begin(); it != mPairs.end(); it++)
            {
                int idx = distance(mPairs.begin(), it);
                if (it->first > deltrk_idx) {
                    mPairs[idx] = make_pair(it->first - 1, it->second);
                }
            }
        }
    }
    // Update current trackers.
    vector<int> mtrk_list; // 有匹配的trk在map中的索引
    vector<int> mdet_list;
    for (auto it = mPairs.begin(); it != mPairs.end(); it++) {
        mtrk_list.push_back(it->first);
        mdet_list.push_back(it->second);
    }
    for (auto it = trksMap.begin(); it != trksMap.end(); it++) 
    {   
        Rect2d bbox;
        // map中的索引
        int trkidx = distance(trksMap.begin(), it); 
        // 遍历map中存储的trk的索引(trkidx)是否能在mPairs中找到
        auto a = find(mtrk_list.begin(), mtrk_list.end(), trkidx); // *a == trkidx
        // trk有匹配时：用上帧预测框和当前帧检测框加权更新tracker；trk无匹配时：用上帧预测框更新tracker
        int trkId = it->second.trkID;
        if (a != mtrk_list.end()) 
        {   // trk有匹配
            int aa = distance(mtrk_list.begin(), a); 
            int detidx = mPairs[aa].second; 
            double l = dets[detidx].x;
            double t = dets[detidx].y;
            double w = dets[detidx].width;
            double h = dets[detidx].height;
            bbox = Rect2d(l, t, w, h); 
            if (Evaluate)
            {
                outFile << trkFrame << "," << trkId << ","
                    << std::fixed << std::setprecision(2)
                    << bbox.x << "," << bbox.y << ","
                    << bbox.width << "," << bbox.height
                    << ",1,-1,-1,-1" << std::endl;
            }
        }
        // else { // trk无匹配
        //     bbox = it->second.trkROI;
        //     if (Evaluate)
        //     {
        //         outFile << trkFrame << "," << trkId << ","
        //             << std::fixed << std::setprecision(2)
        //             << bbox.x << "," << bbox.y << ","
        //             << bbox.width << "," << bbox.height
        //             << ",1,-1,-1,-1" << std::endl;
        //     }
        // }
        
        // cout << "======================================================================" << endl;
        // cout << "qian  " << trkId << "->" << " age: " << it->second.trkAge << "  " << bbox.x << " " << bbox.y << " " << bbox.width << " " << bbox.height << endl;
        bbox = adjustBbx(bbox, cv_ptr);
        
        it->second.trk->update(cv_ptr->image, bbox);
         
        bbox = adjustBbx(bbox, cv_ptr);
        // cout << "hou  " << trkId << "->" << " age: " << it->second.trkAge << "  " << bbox.x << " " << bbox.y << " " << bbox.width << " " << bbox.height << endl;
        it->second.trkROI = bbox;
    }
    trkFrame++;

    // Draw & Show
    if (viewIMG)
    {
        vector<string> trk_list;
        Scalar boxRGB(0, 255, 0);
        Scalar txtRGB(255,255,0);
        for (auto t = trksMap.begin(); t != trksMap.end(); t++)
        {
            trk_list.push_back(t->first);
        }
        for (auto td = mPairs.begin(); td != mPairs.end(); td++)
        {
            trksMap[trk_list[td->first]].trkAge = 0;
            int id = trksMap[trk_list[td->first]].trkID;
            Rect2d box = trksMap[trk_list[td->first]].trkROI;
            Point p1(box.x, box.y);
            Point p2(box.x + box.width, box.y + box.height);
            cv::rectangle(cv_ptr->image, p1, p2, boxRGB, thickness);
            cv::putText(cv_ptr->image, "ID-"+std::to_string(id), Point(box.x,box.y-3), cv::FONT_HERSHEY_SIMPLEX, 0.6, txtRGB, 1, cv::LINE_AA);
        }
        cv::imshow("6", cv_ptr->image);
        cv::waitKey(1);
    }
    clock_t endTime = clock();
    // 计算运行时间
    double duration = static_cast<double>(endTime - startTime) / CLOCKS_PER_SEC;
    // 输出运行时间
    // std::cout << duration << std::endl;
}

int getMaxExpIndex(const std::string& root) {
    int maxIndex = 1;
    DIR* dir = opendir(root.c_str()); // 打开根目录

    if (dir != nullptr) {
        struct dirent* entry;
        while ((entry = readdir(dir)) != nullptr) {
            std::string filename = entry->d_name;
            if (filename.find("exp") == 0) { // 仅匹配以"exp"开头的文件夹
                int index = 1;
                try {
                    string str1 = filename.substr(3); // 解析索引部分
                    std::reverse(str1.begin(),str1.end());
                    string str2 = str1.substr(6);
                    std::reverse(str2.begin(),str2.end());
                    index = std::stoi(str2);
                } catch (const std::invalid_argument&) {
                    continue; // 转换失败，跳过
                }
                maxIndex = std::max(maxIndex, index);
            }
        }
        closedir(dir); // 关闭根目录
    } else {
        std::cerr << "Error opening directory: " << root << std::endl;
    }

    return maxIndex;
}

string createPassingFolder(const std::string& root, const std::string& trainName)
{
    struct stat info1, info2;
    std::string trainPath;
    std::string TrackName;
    std::string TrackPath;
    std::string trkTxtPath;

    trainPath = root + '/' + trainName;
    if (stat(trainPath.c_str(), &info1) != 0)
    {
        mkdir(trainPath.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    }
    TrackName = trainName.substr(0,trainName.length()-6)+"Track";
    TrackPath = trainPath + '/' + TrackName;
    if (stat(TrackPath.c_str(), &info2) != 0)
    {
        mkdir(TrackPath.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    }
    trkTxtPath = root+'/'+trainName+'/'+TrackName+'/'+"data"; //trk.txt文件存放路径
    
    return trkTxtPath;
}

string createExpFolder(const std::string& root, bool newTrain) 
{
    struct stat info;
    std::string trainName;
    std::string trkTxtPath;
    int exp_idx = getMaxExpIndex(root);  //找到该路径下train的最大索引
    if (newTrain)
    {
        trainName = "exp" + std::to_string(exp_idx + 1) + "-train"; //xx-train文件夹的路径
        trkTxtPath = createPassingFolder(root, trainName); //trk.txt文件存放路径  
    }
    else {
        trainName = "exp" + std::to_string(exp_idx) + "-train";
        trkTxtPath = createPassingFolder(root, trainName);;
    }

    if (stat(trkTxtPath.c_str(), &info) != 0)
    {
        mkdir(trkTxtPath.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    }

    return trkTxtPath;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "Multi_SOT");

    bool newTrain = false;
    std::string exp_pth;
    std::string bag_name;
    bool view_img = false;
    bool evaluate = false;
    int max_age, min_hits;

    ros::param::get("~evaluate", evaluate);
    ros::param::get("~view_image", view_img);
    ros::param::get("~bag_name", bag_name);
    ros::param::get("~create_new_train", newTrain);
    ros::param::get("~max_age", max_age);
    ros::param::get("~min_hits",min_hits);
    
    string root = "/home/fbh/2023_goal/test/Evaluate/TrackEval/data/trackers/mot_challenge";   
    
    exp_pth = createExpFolder(root, newTrain);
    cout << exp_pth << endl;
    SoTracker ST(exp_pth, bag_name, view_img, evaluate, max_age, min_hits); 

    return 0;
}