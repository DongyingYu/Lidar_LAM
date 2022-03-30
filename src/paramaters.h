/**
 * @file paramaters.h
 * @author Dongying (yudong2817@sina.com)
 * @brief 
 * @version 1.0
 * @date 2022-03-25
 *
 * @copyright Copyright (c) 2022
 */
#pragma once

#include <ros/ros.h>

extern int accumulate_window = 1;
extern double surf_filter_length = 0.2;
extern double corn_filter_length = 0.0;
extern int window_size = 20;
extern int margi_size = 5;
extern int filter_num = 1;
extern int thread_num = 4;
extern int scan2map_on = 10;
extern int pub_skip = 5;
extern double voxel_size[2] = {1, 1}; // {surf, corn}


void readParameters(ros::NodeHandle &n);

void initialParameters(ros::NodeHandle &n);