/**
 * @fileparamaters.cc
 * @author Dongying (yudong2817@sina.com)
 * @brief 
 * @version 1.0
 * @date 2022-03-25
 *
 * @copyright Copyright (c) 2022
 */

#include "paramaters.h"

int accumulate_window;
double surf_filter_length;
double corn_filter_length;
int windows_size;
int margi_size;
int filter_num;
int thread_num;
int scan2map_on;
int pub_skip;

template <typename T>
T readParam(ros::NodeHandle &n, std::string name){
    T ret;
    if(n.getParam(name, ret)){
        ROS_INFO_STREAM("Loaded  " << name << ": " << ans);
    }else
    {
        ROS_ERROR_STREAM("Failed to load  " << name);
    }
    return ret;
}

// 通过加载yaml文件读取参数
void readParameters(ros::NodeHandle &n){
    std::string config_file;
    config_file = getParam<std::string>(n, "config_file");
    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if(!fsSettings.isOpened()){
        std::cerr << "[ERROR]: Wrong path to settings. " << std::endl;
    }

    fsSettings["accumulate_window"] >> accumulate_window;
    fsSettings["surf_filter_length"] >> surf_filter_length;
    fsSettings["corn_filter_length"] >> corn_filter_length;
    windows_size = fsSettings["windows_size"];
    margi_size = fsSettings["margi_size"];
    filter_num = fsSettings["filter_num"];
    thread_num = fsSettings["thread_num"];
    scan2map_on = fsSettings["scan2map_on"];
    pub_skip = fsSettings["pub_skip"];

    // TODO 直接初始化参数赋值

    fsSettings.release();
}

// 从launch文件中加载参数赋值
void initialParameters(ros::NodeHandle &n){
    n.param<int>("accumulate_window",accumulate_window,3);
    n.param<double>("surf_filter_length", surf_filter_length, 0.4);
    n.param<double>("corn_filter_length", corn_filter_length, 0.2);
    n.param<double>("root_surf_voxel_size", voxel_size[0], 1);
    n.param<double>("root_corn_voxel_size", voxel_size[1], 1);
    n.param<int>("scan2map_on", scan2map_on, 10);
    n.param<int>("pub_skip", pub_skip, 1);    
}
