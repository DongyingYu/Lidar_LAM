/**
 * @file basic_data.cc
 * @author Dongying (yudong2817@sina.com)
 * @brief 
 * @version 1.0
 * @date 2022-03-25
 *
 * @copyright Copyright (c) 2022
 */

#include "basic_data.h"

VoxelStrcuture::VoxelStrcuture(int64_t vx, int64_t vy, int64_t vz) : x_(vx), y_(vy), z_(vz) {}

SigmaVector::SigmaVector()
{
    sigma_vTv_.setZero();
    sigma_vi_.setZero();
    sigma_size_ = 0;
}

void SigmaVector::toZero()
{
    sigma_vTv_.setZero();
    sigma_vi_.setZero();
    sigma_size_ = 0;
}