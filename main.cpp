/* 
 * File:   main.cpp
 * Author: heshan
 *
 * Created on June 6, 2018, 12:03 AM
 */

#include <iostream>
#include <libiproc.h>
#include "CNN.hpp"

/*
 * 
 */
int main(int argc, char** argv) {
    
   
    std::tuple<int, int, int> dimensions = std::make_tuple(1,28,28);
    
    struct::ConvLayStruct CL1;
    CL1.filterSize = 5; // filter size: N x N
    CL1.filters = 3; // No of filters
    CL1.stride = 1;
    
    struct::PoolLayStruct PL1;
    PL1.poolH = 2; // pool size: N x N
    PL1.poolW = 2;
    
    struct::FCLayStruct FCL1;
    FCL1.outputs = 5; // neurons in fully connected layer
    FCL1.classes = 4; // target classes
    
    char layerOrder[] = {'C','P','F'};
    struct::ConvLayStruct CLs[] = {CL1};
    struct::PoolLayStruct PLs[] = {PL1};
    struct::FCLayStruct FCLs[] = {FCL1};
    
    
    struct::NetStruct netStruct;
    netStruct.layers = 3;
    netStruct.layerOrder = layerOrder;
    netStruct.CL = CLs;
    netStruct.PL = PLs;
    netStruct.FCL = FCLs;
    
    Eigen::MatrixXd inImgArr[1];
    Eigen::MatrixXd inImg(28,28);
    
    for(int i = 0; i < 28; i++){
        for(int j = 10; j < 28; j++){
            inImg(i,j) = 1;
        }
    }
    
    inImgArr[0] = inImg;
    
    CNN cn(dimensions, netStruct);
    cn.train(inImgArr);
    
    
//    cn.tmp();
    
    
    return 0;
}

