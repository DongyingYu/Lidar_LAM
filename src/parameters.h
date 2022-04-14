/**
 * @file parameters.h
 * @author Dongying (yudong2817@sina.com)
 * @brief 
 * @version 1.0
 * @date 2022-03-25
 *
 * @copyright Copyright (c) 2022
 */
#pragma once

#include <ros/ros.h>
#include <opencv2/highgui/highgui.hpp>

extern int accumulate_window;
extern double surf_filter_length;
extern double corn_filter_length;
extern int window_size;
extern int margi_size;
extern int filter_num;
extern int thread_num;
extern int scan2map_on;
extern int pub_skip;
// 网格大小以米为单位
extern double voxel_size[2]; // {surf, corn}


void readParameters(ros::NodeHandle &n);

void initialParameters(ros::NodeHandle &n);