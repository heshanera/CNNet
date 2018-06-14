/* 
 * File:   main.cpp
 * Author: heshan
 *
 * Created on June 6, 2018, 12:03 AM
 */

#include <iostream>
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
    struct::ConvLayStruct CL2;
    CL2.filterSize = 4; // filter size: N x N
    CL2.filters = 2; // No of filters
    CL2.stride = 1;
    
    struct::PoolLayStruct PL1;
    PL1.poolH = 2; // pool size: N x N
    PL1.poolW = 2;
    struct::PoolLayStruct PL2;
    PL2.poolH = 2; // pool size: N x N
    PL2.poolW = 2;
    
    struct::FCLayStruct FCL1;
    FCL1.outputs = 10; // neurons in fully connected layer
    FCL1.classes = 4; // target classes
    struct::FCLayStruct FCL2;
    FCL2.outputs = 8; // neurons in fully connected layer
    FCL2.classes = 4; // target classes
    struct::FCLayStruct FCL3;
    FCL3.outputs = 5; // neurons in fully connected layer
    FCL3.classes = 4; // target classes
    
    char layerOrder[] = {'C','P','C','P','F','F','F'};
    struct::ConvLayStruct CLs[] = {CL1,CL2};
    struct::PoolLayStruct PLs[] = {PL1,PL2};
    struct::FCLayStruct FCLs[] = {FCL1,FCL2,FCL3};
    
    
    struct::NetStruct netStruct;
    netStruct.layers = 7;
    netStruct.layerOrder = layerOrder;
    netStruct.CL = CLs;
    netStruct.PL = PLs;
    netStruct.FCL = FCLs;
    
    Eigen::MatrixXd ** inImgArr;
    inImgArr = new Eigen::MatrixXd * [1]; 
    inImgArr[0] = new Eigen::MatrixXd[1]; 
    Eigen::MatrixXd inImg = Eigen::MatrixXd::Random(28,28);
    inImgArr[0][0] = inImg;
    
    Eigen::MatrixXd * inLblArr;
    inLblArr = new Eigen::MatrixXd[1];
    Eigen::MatrixXd inLbl = Eigen::MatrixXd::Zero(5,1);
    inLbl(2,0) = 1;
    inLblArr[0] = inLbl;
    
    int iterations = 20;
    int inputSize = 1;
    double learningRate = 0.1;
    
    CNN cn(dimensions, netStruct);
    cn.train(inImgArr, inLblArr, inputSize, iterations, learningRate);
    
    Eigen::MatrixXd tstImgArr[1];
    tstImgArr[0] = Eigen::MatrixXd::Random(28,28);
    
    std::cout<<cn.predict(tstImgArr)<<"\n";
    
    
//    cn.tmp();
    
    
    return 0;
}

