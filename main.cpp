/* 
 * File:   main.cpp
 * Author: heshan
 *
 * Created on June 6, 2018, 12:03 AM
 */

#include <iostream>

#include "CNN.hpp"
#include "FileProcessor.h"
#include "DataProcessor.h"

int net() {
    
    int width = 20;
    int height = 20;
    
    std::tuple<int, int, int> dimensions = std::make_tuple(1,height,width);
    
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
    FCL1.outputs = 30; // neurons in fully connected layer
    FCL1.classes = 4; // target classes
    struct::FCLayStruct FCL2;
    FCL2.outputs = 15; // neurons in fully connected layer
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
    Eigen::MatrixXd inImg = Eigen::MatrixXd::Random(height,width);
    inImgArr[0][0] = inImg;
    
    Eigen::MatrixXd * inLblArr;
    inLblArr = new Eigen::MatrixXd[1];
    Eigen::MatrixXd inLbl = Eigen::MatrixXd::Zero(5,1);
    inLbl(2,0) = 1;
    inLblArr[0] = inLbl;
    
    int iterations = 20;
    int inputSize = 1;
    double learningRate = 0.00001;
    
    CNN cn(dimensions, netStruct);
    cn.train(inImgArr, inLblArr, inputSize, iterations, learningRate);
    
    Eigen::MatrixXd tstImgArr[1];
    tstImgArr[0] = Eigen::MatrixXd::Random(height,width);
    
    std::cout<<cn.predict(tstImgArr)<<"\n";
    
    return 0;
}

/**
 * Time Series = { t, t+1, t+2, .... t+x} 
 * Input  = { {t, t+1, .. t+m} ...., {t+q+1, t+q+2, .. t+n} }
 */
int net2() {
    
    // Generating a convolutional network
    int width = 20;
    int height = 2;
    int iterations = 20;
    int inputSize = 20;
    int targetsC = 1;
    double learningRate = 1;
    
    std::string infiles[] = {"dummy2.txt","InternetTraff.txt","dailyMinimumTemperatures.txt"};
    std::string inFile = infiles[0];
    
    // network structure
    std::tuple<int, int, int> dimensions = std::make_tuple(1,height,width);
    
    struct::ConvLayStruct CL1;
    CL1.filterSize = 2; // filter size: N x N
    CL1.filters = 1; // No of filters
    CL1.stride = 1;
//    struct::ConvLayStruct CL2;
//    CL2.filterSize = 4; // filter size: N x N
//    CL2.filters = 3; // No of filters
//    CL2.stride = 1;
    
    struct::PoolLayStruct PL1;
    PL1.poolH = 1; // pool size: N x N
    PL1.poolW = 2;
//    struct::PoolLayStruct PL2;
//    PL2.poolH = 2; // pool size: N x N
//    PL2.poolW = 2;
    
    struct::FCLayStruct FCL1;
    FCL1.outputs = 60; // neurons in fully connected layer
//    FCL1.classes = 4; // target classes
    struct::FCLayStruct FCL2;
    FCL2.outputs = 10; // neurons in fully connected layer
//    FCL2.classes = 1; // target classes
    struct::FCLayStruct FCL3;
    FCL3.outputs = 1; // neurons in fully connected layer
//    FCL3.classes = 1; // target classes
    
    char layerOrder[] = {/*'C','P',*/'C','P','F','F','F'};
    struct::ConvLayStruct CLs[] = {CL1/*,CL2*/};
    struct::PoolLayStruct PLs[] = {PL1/*,PL2*/};
    struct::FCLayStruct FCLs[] = {FCL1,FCL2,FCL3};
    
    
    struct::NetStruct netStruct;
    netStruct.layers = 5;
    netStruct.layerOrder = layerOrder;
    netStruct.CL = CLs;
    netStruct.PL = PLs;
    netStruct.FCL = FCLs;
    
    // Generating input matrices
    Eigen::MatrixXd ** inMatArr;
    Eigen::MatrixXd * inLblArr;
    Eigen::MatrixXd inMat;
    Eigen::MatrixXd inLbl;
    inMatArr = new Eigen::MatrixXd * [inputSize];
    inLblArr = new Eigen::MatrixXd[inputSize];
    
    // Reading the file
    FileProcessor fp;
    DataProcessor dp;
    std::vector<double> timeSeries;
    timeSeries = fp.read("datasets/univariate/inputs/"+inFile,1);
    timeSeries =  dp.process(timeSeries,1);
        
    std::vector<double>::const_iterator start;
    std::vector<double>::const_iterator end;
    int inputMatSize = height * width;

    for (int i = 0; i < inputSize; i++) {
        
//        start = timeSeries.begin() + i;
//        end = timeSeries.begin() + i + inputMatSize + 1;
//        std::vector<double> inVec(start, end);
//        inVec =  dp.process(inVec,1);
        
        // inputs
        inMatArr[i] = new Eigen::MatrixXd[1]; // image depth
        inMat = Eigen::MatrixXd(height,width);
        for (int a = 0; a < height; a++) {
            for (int b = 0; b < width; b++) {
//                inMat(a,b) = inVec.at(( a * width ) + b);
                inMat(a,b) = timeSeries.at(i + ( a * width ) + b);
            }
        }
        inMatArr[i][0] = inMat;
        // labels
        inLbl = Eigen::MatrixXd::Zero(targetsC,1);
        for (int a = 0; a < targetsC; a++) {
//            inLbl(a,0) = timeSeries.at(((i + 1) * width) + a);
            inLbl(a,0) = timeSeries.at(i + (width*height));
//            inLbl(a,0) = inVec.at(width*height);
        }
        inLblArr[i] = inLbl;
    }
    
    // Generating the network
    CNN cn(dimensions, netStruct);
    // Training the network
    cn.train(inMatArr, inLblArr, inputSize, iterations, learningRate);
    
    // Predictions
    std::cout<<"\n Predictions: \n";
    Eigen::MatrixXd prediction;
    // Open the file to write the time series predictions
    std::ofstream out_file;
    out_file.open("datasets/univariate/predictions/"+inFile,std::ofstream::out | std::ofstream::trunc);
    Eigen::MatrixXd tstMatArr[1];
    double errorSq = 0, MSE;
    double expected;
    double val;
    int predSize = 1200;//timeSeries.size() - matSize; // training size 500 points
    for (int i = 0; i < predSize; i++) {
        
//        start = timeSeries.begin() + i;
//        end = timeSeries.begin() + i + inputMatSize + 1;
//        std::vector<double> inVec(start, end);
//        inVec =  dp.process(inVec,1);
        
        tstMatArr[0] = Eigen::MatrixXd::Zero(height,width);
        for (int a = 0; a < height; a++) {
            for (int b = 0; b < width; b++) {
                tstMatArr[0](a,b) = timeSeries.at(i + ( a * width ) + b);
//                tstMatArr[0](a,b) = inVec.at(( a * width ) + b);
            }
        }
        
        prediction = cn.predict(tstMatArr);
        std::cout<<prediction<<"\n"; 
        expected = timeSeries.at(i + (width*height));
        for (int i = 0; i < targetsC; i++) {
            val = prediction(i,0);
            errorSq += pow(val - expected,2);
            out_file<<dp.postProcess(val)<<"\n"; 
        }
    }
    MSE = errorSq/predSize;
    std::cout<<"\nMean Squared Error: "<<MSE<<"\n\n";
    
    return 0;
}

/**
 * Multivariate Time Series
 */
int net3() {
    
    // Generating a convolutional network
    int width = 5;
    int height = 1;
    int iterations = 20;
    int inputSize = 20;
    int targetsC = 1;
    double learningRate = 1;
    
    // network structure
    std::tuple<int, int, int> dimensions = std::make_tuple(1,height,width);
    
    struct::ConvLayStruct CL1;
    CL1.filterSize = 1; // filter size: N x N
    CL1.filters = 1; // No of filters
    CL1.stride = 1;
//    struct::ConvLayStruct CL2;
//    CL2.filterSize = 4; // filter size: N x N
//    CL2.filters = 3; // No of filters
//    CL2.stride = 1;
    
    struct::PoolLayStruct PL1;
    PL1.poolH = 1; // pool size: N x N
    PL1.poolW = 2;
//    struct::PoolLayStruct PL2;
//    PL2.poolH = 2; // pool size: N x N
//    PL2.poolW = 2;
    
    struct::FCLayStruct FCL1;
    FCL1.outputs = 60; // neurons in fully connected layer
//    FCL1.classes = 4; // target classes
    struct::FCLayStruct FCL2;
    FCL2.outputs = 10; // neurons in fully connected layer
//    FCL2.classes = 1; // target classes
    struct::FCLayStruct FCL3;
    FCL3.outputs = 1; // neurons in fully connected layer
//    FCL3.classes = 1; // target classes
    
    char layerOrder[] = {/*'C','P',*/'C','P','F','F','F'};
    struct::ConvLayStruct CLs[] = {CL1/*,CL2*/};
    struct::PoolLayStruct PLs[] = {PL1/*,PL2*/};
    struct::FCLayStruct FCLs[] = {FCL1,FCL2,FCL3};
    
    
    struct::NetStruct netStruct;
    netStruct.layers = 5;
    netStruct.layerOrder = layerOrder;
    netStruct.CL = CLs;
    netStruct.PL = PLs;
    netStruct.FCL = FCLs;
    
    // Generating input matrices
    Eigen::MatrixXd ** inMatArr;
    Eigen::MatrixXd * inLblArr;
    Eigen::MatrixXd inMat;
    Eigen::MatrixXd inLbl;
    inMatArr = new Eigen::MatrixXd * [inputSize];
    inLblArr = new Eigen::MatrixXd[inputSize];
    
    // Reading the file
    FileProcessor fp;
    DataProcessor dp;
    std::vector<double> * timeSeries;
    
    int colIndxs[] = {0,0,1,1,1,1,1};
    int targetValCol = 7;
    int lines = 5000;
    int inputVecSize = height*width;
    timeSeries = fp->readMultivariate("datasets/multivariate/input/occupancyData/datatraining.txt",lines,inputVecSize,colIndxs,targetValCol);
    
    std::vector<double>::const_iterator start;
    std::vector<double>::const_iterator end;
    int inputMatSize = height * width;

    for (int i = 0; i < inputSize; i++) {
        
//        start = timeSeries.begin() + i;
//        end = timeSeries.begin() + i + inputMatSize + 1;
//        std::vector<double> inVec(start, end);
//        inVec =  dp.process(inVec,1);
        
        // inputs
        inMatArr[i] = new Eigen::MatrixXd[1]; // image depth
        inMat = Eigen::MatrixXd(height,width);
        for (int a = 0; a < height; a++) {
            for (int b = 0; b < width; b++) {
//                inMat(a,b) = inVec.at(( a * width ) + b);
                inMat(a,b) = timeSeries[i].at((a * width ) + b);
            }
        }
        inMatArr[i][0] = inMat;
        // labels
        inLbl = Eigen::MatrixXd::Zero(targetsC,1);
        for (int a = 0; a < targetsC; a++) {
//            inLbl(a,0) = timeSeries.at(((i + 1) * width) + a);
            inLbl(a,0) = timeSeries.at(i + (width*height));
//            inLbl(a,0) = inVec.at(width*height);
        }
        inLblArr[i] = inLbl;
    }
    
    // Generating the network
    CNN cn(dimensions, netStruct);
    // Training the network
    cn.train(inMatArr, inLblArr, inputSize, iterations, learningRate);
    
    // Predictions
    std::cout<<"\n Predictions: \n";
    Eigen::MatrixXd prediction;
    // Open the file to write the time series predictions
    std::ofstream out_file;
    out_file.open("datasets/univariate/predictions/"+inFile,std::ofstream::out | std::ofstream::trunc);
    Eigen::MatrixXd tstMatArr[1];
    double errorSq = 0, MSE;
    double expected;
    double val;
    int predSize = 1200;//timeSeries.size() - matSize; // training size 500 points
    for (int i = 0; i < predSize; i++) {
        
//        start = timeSeries.begin() + i;
//        end = timeSeries.begin() + i + inputMatSize + 1;
//        std::vector<double> inVec(start, end);
//        inVec =  dp.process(inVec,1);
        
        tstMatArr[0] = Eigen::MatrixXd::Zero(height,width);
        for (int a = 0; a < height; a++) {
            for (int b = 0; b < width; b++) {
                tstMatArr[0](a,b) = timeSeries.at(i + ( a * width ) + b);
//                tstMatArr[0](a,b) = inVec.at(( a * width ) + b);
            }
        }
        
        prediction = cn.predict(tstMatArr);
        std::cout<<prediction<<"\n"; 
        expected = timeSeries.at(i + (width*height));
        for (int i = 0; i < targetsC; i++) {
            val = prediction(i,0);
            errorSq += pow(val - expected,2);
            out_file<<dp.postProcess(val)<<"\n"; 
        }
    }
    MSE = errorSq/predSize;
    std::cout<<"\nMean Squared Error: "<<MSE<<"\n\n";
    
    return 0;
}

int main() {
    
    net2();
    
    return 0;
}

