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
    int iterations = 1;
    int inputSize = 5000;
    int targetsC = 1;
    double learningRate = 1;
    
    // network structure
    std::tuple<int, int, int> dimensions = std::make_tuple(1,height,width);
    
    struct::ConvLayStruct CL1;
    CL1.filterSize = 1; // filter size: N x N
    CL1.filters = 1; // No of filters
    CL1.stride = 1;
    
    struct::PoolLayStruct PL1;
    PL1.poolH = 1; // pool size: N x N
    PL1.poolW = 1;
    
    struct::FCLayStruct FCL1;
    FCL1.outputs = 4; // neurons in fully connected layer

    struct::FCLayStruct FCL2;
    FCL2.outputs = 2; // neurons in fully connected layer

    struct::FCLayStruct FCL3;
    FCL3.outputs = 1; // neurons in fully connected layer
    
    char layerOrder[] = {'C','P','F','F','F'};
    struct::ConvLayStruct CLs[] = {CL1};
    struct::PoolLayStruct PLs[] = {PL1};
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
    int lines = inputSize;
    int inputVecSize = 5;
    timeSeries = fp.readMultivariate("datasets/multivariate/input/occupancyData/datatraining.txt",lines,inputVecSize,colIndxs,targetValCol);
    
    std::vector<double>::const_iterator first = timeSeries[lines].begin();
    std::vector<double>::const_iterator last = timeSeries[lines].begin() + inputSize;
    std::vector<double> targetVector(first, last);
    for (std::vector<double>::iterator it = targetVector.begin(); it != targetVector.end(); ++it) {
        if (*it == 0) *it = -1;
    } 

    for (int i = 0; i < inputSize; i++) {
        
        if (timeSeries[i].size() != 5) continue;
        
        // inputs
        timeSeries[i] = dp.process(timeSeries[i],0);
        inMatArr[i] = new Eigen::MatrixXd[1]; // image depth
        inMat = Eigen::MatrixXd(height,width);
        for (int a = 0; a < height; a++) {
            for (int b = 0; b < width; b++) {
                inMat(a,b) = timeSeries[i].at(b);
            }
        }
        
        inMatArr[i][0] = inMat;
        // labels
        inLbl = Eigen::MatrixXd::Zero(targetsC,1);
        // filling the target vector
        inLbl(0,0) = targetVector.at(i);
//        if (targetVector.at(i) == 1) inLbl(1,0) = -1;
//        else inLbl(1,0) = 1;
        inLblArr[i] = inLbl;
    }
    
    // Generating the network
    CNN cn(dimensions, netStruct);
    // Training the network
    cn.train(inMatArr, inLblArr, inputSize, iterations, learningRate);
    
    // Predictions
    std::cout<<"\n Predictions: \n";
    lines = 2000; // lines read from the files
    timeSeries = fp.readMultivariate("datasets/multivariate/input/occupancyData/datatest.txt",lines,inputVecSize,colIndxs,targetValCol);
    Eigen::MatrixXd prediction;
    Eigen::MatrixXd tstMatArr[1];
    int predSize = 2000;//timeSeries.size() - matSize; // training size 500 points
    double min = 0, max = 0;
    double result;
    std::vector<double> resultVec;
    for (int i = 0; i < predSize; i++) {
        tstMatArr[0] = Eigen::MatrixXd::Zero(height,width);
        timeSeries[i] = dp.process(timeSeries[i],0);
        for (int a = 0; a < height; a++) {
            for (int b = 0; b < width; b++) {
                tstMatArr[0](a,b) = timeSeries[i].at(b);
            }
        }
        
        prediction = cn.predict(tstMatArr);
        result = prediction(0,0);
//        std::cout<<prediction<<"\n\n";
        
//        if (prediction(0,0) > 0.5) result = 1;
        
        resultVec.push_back(result);
//        std::cout<<result<<"\n"; 

        if (i == 0){
            min = result;
            max = result;
        } else {
        
            if (min > result) min = result;
            if (max < result) max = result;
        }
    }
    
    std::cout<<"min: "<<min<<std::endl;
    std::cout<<"max: "<<max<<std::endl;
    
    double line = 0; //(min + max)/2;
    std::cout<<"margin: "<<line<<std::endl<<std::endl;
    
    
    int occu = 0, notoccu = 0;
    
    int corr = 0;
    int incorr = 0;
    
    int truePos = 0;
    int falsePos = 0;
    int trueNeg = 0;
    int falseNeg = 0;
    
    int corrNwMgn = 0;
    int incorrNwMgn = 0;
    
    // Open the file to write the time series predictions
    std::ofstream out_file;
    std::ofstream out_file2;
    out_file.open("datasets/multivariate/predictions/occupancyData/multiResults.txt",std::ofstream::out | std::ofstream::trunc);
    out_file2.open("datasets/multivariate/predictions/occupancyData/multiTargets.txt",std::ofstream::out | std::ofstream::trunc);
    
    for (int i = 0; i < predSize; i++) { 
        out_file<<timeSeries[lines].at(i)<<","<<resultVec.at(i)<<"\n";
        out_file2<<timeSeries[lines].at(i)<<",";
        if (timeSeries[lines].at(i) == 1) {
            out_file2<<1<<"\n";
        } else out_file2<<-1<<"\n";
        
        if ( (resultVec.at(i) > line) && (timeSeries[lines].at(i) == 1)) { 
            corr++;
            truePos++;
            occu++;
        } else if ( (resultVec.at(i) <= line) && (timeSeries[lines].at(i) == 0)) {
            corr++;
            trueNeg++;
            notoccu++;
        } else if ( (resultVec.at(i) <= line) && (timeSeries[lines].at(i) == 1)) { 
            incorr++; 
            falseNeg++;
            occu++;
        } else if ( (resultVec.at(i) > line) && (timeSeries[lines].at(i) == 0)) { 
            incorr++; 
            falsePos++;
            notoccu++;
        }
        //std::cout<<resultVec.at(i)<<" ------ "<<timeSeries[lines].at(i)<<"\n";
        
    }
    
    std::cout<<std::endl;
    
    std::cout<<"----------------------"<<std::endl;
    std::cout<<"Data "<<std::endl;
    std::cout<<"----------------------"<<std::endl;
    std::cout<<"Occupied: "<<occu<<std::endl;
    std::cout<<"NotOccupied: "<<notoccu<<std::endl<<std::endl;
    
    std::cout<<"----------------------"<<std::endl;
    std::cout<<"margin: "<<line<<std::endl;
    std::cout<<"----------------------"<<std::endl;
    std::cout<<"Correct predictions: "<<corr<<std::endl;
    std::cout<<"Incorrect predictions: "<<incorr<<std::endl<<std::endl;
    
    std::cout<<"True Positive: "<<truePos<<std::endl;
    std::cout<<"True Negative: "<<trueNeg<<std::endl;
    std::cout<<"False Positive: "<<falsePos<<std::endl;
    std::cout<<"False Negative: "<<falseNeg<<std::endl;
    
    std::cout<<std::endl<<"Accuracy: "<<(corr/(double)predSize)*100<<"%"<<std::endl<<std::endl;
    
    
    line = (min + max)/2;
    occu = 0;
    notoccu = 0;
    corr = 0;
    incorr = 0;
    truePos = 0;
    falsePos = 0;
    trueNeg = 0;
    falseNeg = 0;
    
    for (int i = 0; i < predSize; i++) {    
        if ( (resultVec.at(i) > line) && (timeSeries[lines].at(i) == 1)) { 
            corr++;
            truePos++;
            occu++;
        } else if ( (resultVec.at(i) <= line) && (timeSeries[lines].at(i) == 0)) {
            corr++;
            trueNeg++;
            notoccu++;
        } else if ( (resultVec.at(i) <= line) && (timeSeries[lines].at(i) == 1)) { 
            incorr++; 
            falseNeg++;
            occu++;
        } else if ( (resultVec.at(i) > line) && (timeSeries[lines].at(i) == 0)) { 
            incorr++; 
            falsePos++;
            notoccu++;
        }
        
        
        
        if (line > 0) {
            if ( (resultVec.at(i) <= line) && (resultVec.at(i) > 0)) {
                if (timeSeries[lines].at(i) == 0) {
                    corrNwMgn++;
                } else incorrNwMgn++;
            }
        } else {
            if ( (resultVec.at(i) > line) && (resultVec.at(i) < 0)) {
                if (timeSeries[lines].at(i) == 1) {
                    corrNwMgn++;
                } else incorrNwMgn++;
            }
        }
        
    }
    
    std::cout<<"----------------------"<<std::endl;
    std::cout<<"margin: "<<line<<std::endl;
    std::cout<<"----------------------"<<std::endl;
    std::cout<<"Correct predictions: "<<corr<<std::endl;
    std::cout<<"Incorrect predictions: "<<incorr<<std::endl<<std::endl;
    
    std::cout<<"True Positive: "<<truePos<<std::endl;
    std::cout<<"True Negative: "<<trueNeg<<std::endl;
    std::cout<<"False Positive: "<<falsePos<<std::endl;
    std::cout<<"False Negative: "<<falseNeg<<std::endl;
    
    std::cout<<std::endl<<"Accuracy: "<<(corr/(double)predSize)*100<<"%"<<std::endl<<std::endl;
    
    std::cout<<"----------------------"<<std::endl;
    std::cout<<"Within the new margin and 0"<<std::endl;
    std::cout<<"----------------------"<<std::endl;
    std::cout<<"Correct: "<<corrNwMgn<<std::endl;
    std::cout<<"Incorrect: "<<incorrNwMgn<<std::endl<<std::endl<<std::endl;
    
    return 0;
    
}

/**
 * Multivariate Time Series
 */
int net4() {
    
    // Generating a convolutional network
    int width = 5;
    int height = 5;
    int iterations = 1;
    int inputSize = 1000;
    int targetsC = 5;
    double learningRate = 1;
    
    // network structure
    std::tuple<int, int, int> dimensions = std::make_tuple(1,height,width);
    
    struct::ConvLayStruct CL1;
    CL1.filterSize = 1; // filter size: N x N
    CL1.filters = 1; // No of filters
    CL1.stride = 1;
    
    struct::PoolLayStruct PL1;
    PL1.poolH = 1; // pool size: N x N
    PL1.poolW = 1;
    
    struct::FCLayStruct FCL1;
    FCL1.outputs = 10; // neurons in fully connected layer

    struct::FCLayStruct FCL2;
    FCL2.outputs = 10; // neurons in fully connected layer

    struct::FCLayStruct FCL3;
    FCL3.outputs = 5; // neurons in fully connected layer
    
    char layerOrder[] = {'C','P','F','F','F'};
    struct::ConvLayStruct CLs[] = {CL1};
    struct::PoolLayStruct PLs[] = {PL1};
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
    int lines = inputSize;
    int inputVecSize = 5;
    timeSeries = fp.readMultivariate("datasets/multivariate/input/occupancyData/datatraining.txt",lines,inputVecSize,colIndxs,targetValCol);
    
    std::vector<double>::const_iterator first = timeSeries[lines].begin();
    std::vector<double>::const_iterator last = timeSeries[lines].begin() + inputSize;
    std::vector<double> targetVector(first, last);
    for (std::vector<double>::iterator it = targetVector.begin(); it != targetVector.end(); ++it) {
        if (*it == 0) *it = -1;
    } 

    for (int i = 0; i < inputSize; i++) {
        
        inMat = Eigen::MatrixXd(height,width);
        
        inMatArr[i] = new Eigen::MatrixXd[1]; // image depth
        for(int j = 0; j < height; j++) {
            // inputs
            timeSeries[(i*5) +j] = dp.process(timeSeries[(i*5) +j],0);
            for (int b = 0; b < width; b++) {
                inMat(j,b) = timeSeries[(i*5) +j].at(b);
            }
        }
        inMatArr[i][0] = inMat;
        // labels
        inLbl = Eigen::MatrixXd::Zero(targetsC,1);
        // filling the target vector
        for(int j = 0; j < targetsC; j++) {
            inLbl(j,0) = targetVector.at(i+j);
        }        
        inLblArr[i] = inLbl;
    }

    // Generating the network
    CNN cn(dimensions, netStruct);
    // Training the network
    cn.train(inMatArr, inLblArr, inputSize, iterations, learningRate);
    
    // Predictions
    std::cout<<"\n Predictions: \n";
    lines = 100; // lines read from the files
    timeSeries = fp.readMultivariate("datasets/multivariate/input/occupancyData/datatest.txt",lines,inputVecSize,colIndxs,targetValCol);
    Eigen::MatrixXd prediction;
    Eigen::MatrixXd tstMatArr[1];
    int predSize = 2000;//timeSeries.size() - matSize; // training size 500 points
    double min = 0, max = 0;
    double result;
    std::vector<double> resultVec;
    for (int i = 0; i < predSize; i++) {
        tstMatArr[0] = Eigen::MatrixXd::Zero(height,width);
        timeSeries[i] = dp.process(timeSeries[i],0);
        for (int a = 0; a < height; a++) {
            for (int b = 0; b < width; b++) {
                tstMatArr[0](a,b) = timeSeries[i].at(b);
            }
        }
        
        prediction = cn.predict(tstMatArr);
        result = prediction(0,0);
        
        std::cout<<prediction(0,0)<<"\n";
        std::cout<<prediction(1,0)<<"\n";
        std::cout<<prediction(2,0)<<"\n";
        std::cout<<prediction(3,0)<<"\n\n";
        
//        std::cout<<prediction<<"\n\n";
        
//        if (prediction(0,0) > 0.5) result = 1;
        
        resultVec.push_back(result);
//        std::cout<<result<<"\n"; 

        if (i == 0){
            min = result;
            max = result;
        } else {
        
            if (min > result) min = result;
            if (max < result) max = result;
        }
    }
    
    std::cout<<"min: "<<min<<std::endl;
    std::cout<<"max: "<<max<<std::endl;
    
    double line = 0; //(min + max)/2;
    std::cout<<"margin: "<<line<<std::endl<<std::endl;
    
    
    int occu = 0, notoccu = 0;
    
    int corr = 0;
    int incorr = 0;
    
    int truePos = 0;
    int falsePos = 0;
    int trueNeg = 0;
    int falseNeg = 0;
    
    int corrNwMgn = 0;
    int incorrNwMgn = 0;
    
    // Open the file to write the time series predictions
    std::ofstream out_file;
    std::ofstream out_file2;
    out_file.open("datasets/multivariate/predictions/occupancyData/multiResults.txt",std::ofstream::out | std::ofstream::trunc);
    out_file2.open("datasets/multivariate/predictions/occupancyData/multiTargets.txt",std::ofstream::out | std::ofstream::trunc);
    
    for (int i = 0; i < predSize; i++) { 
        out_file<<timeSeries[lines].at(i)<<","<<resultVec.at(i)<<"\n";
        out_file2<<timeSeries[lines].at(i)<<",";
        if (timeSeries[lines].at(i) == 1) {
            out_file2<<1<<"\n";
        } else out_file2<<-1<<"\n";
        
        if ( (resultVec.at(i) > line) && (timeSeries[lines].at(i) == 1)) { 
            corr++;
            truePos++;
            occu++;
        } else if ( (resultVec.at(i) <= line) && (timeSeries[lines].at(i) == 0)) {
            corr++;
            trueNeg++;
            notoccu++;
        } else if ( (resultVec.at(i) <= line) && (timeSeries[lines].at(i) == 1)) { 
            incorr++; 
            falseNeg++;
            occu++;
        } else if ( (resultVec.at(i) > line) && (timeSeries[lines].at(i) == 0)) { 
            incorr++; 
            falsePos++;
            notoccu++;
        }
        //std::cout<<resultVec.at(i)<<" ------ "<<timeSeries[lines].at(i)<<"\n";
        
    }
    
    std::cout<<std::endl;
    
    std::cout<<"----------------------"<<std::endl;
    std::cout<<"Data "<<std::endl;
    std::cout<<"----------------------"<<std::endl;
    std::cout<<"Occupied: "<<occu<<std::endl;
    std::cout<<"NotOccupied: "<<notoccu<<std::endl<<std::endl;
    
    std::cout<<"----------------------"<<std::endl;
    std::cout<<"margin: "<<line<<std::endl;
    std::cout<<"----------------------"<<std::endl;
    std::cout<<"Correct predictions: "<<corr<<std::endl;
    std::cout<<"Incorrect predictions: "<<incorr<<std::endl<<std::endl;
    
    std::cout<<"True Positive: "<<truePos<<std::endl;
    std::cout<<"True Negative: "<<trueNeg<<std::endl;
    std::cout<<"False Positive: "<<falsePos<<std::endl;
    std::cout<<"False Negative: "<<falseNeg<<std::endl;
    
    std::cout<<std::endl<<"Accuracy: "<<(corr/(double)predSize)*100<<"%"<<std::endl<<std::endl;
    
    
    line = (min + max)/2;
    occu = 0;
    notoccu = 0;
    corr = 0;
    incorr = 0;
    truePos = 0;
    falsePos = 0;
    trueNeg = 0;
    falseNeg = 0;
    
    for (int i = 0; i < predSize; i++) {    
        if ( (resultVec.at(i) > line) && (timeSeries[lines].at(i) == 1)) { 
            corr++;
            truePos++;
            occu++;
        } else if ( (resultVec.at(i) <= line) && (timeSeries[lines].at(i) == 0)) {
            corr++;
            trueNeg++;
            notoccu++;
        } else if ( (resultVec.at(i) <= line) && (timeSeries[lines].at(i) == 1)) { 
            incorr++; 
            falseNeg++;
            occu++;
        } else if ( (resultVec.at(i) > line) && (timeSeries[lines].at(i) == 0)) { 
            incorr++; 
            falsePos++;
            notoccu++;
        }
        
        
        
        if (line > 0) {
            if ( (resultVec.at(i) <= line) && (resultVec.at(i) > 0)) {
                if (timeSeries[lines].at(i) == 0) {
                    corrNwMgn++;
                } else incorrNwMgn++;
            }
        } else {
            if ( (resultVec.at(i) > line) && (resultVec.at(i) < 0)) {
                if (timeSeries[lines].at(i) == 1) {
                    corrNwMgn++;
                } else incorrNwMgn++;
            }
        }
        
    }
    
    std::cout<<"----------------------"<<std::endl;
    std::cout<<"margin: "<<line<<std::endl;
    std::cout<<"----------------------"<<std::endl;
    std::cout<<"Correct predictions: "<<corr<<std::endl;
    std::cout<<"Incorrect predictions: "<<incorr<<std::endl<<std::endl;
    
    std::cout<<"True Positive: "<<truePos<<std::endl;
    std::cout<<"True Negative: "<<trueNeg<<std::endl;
    std::cout<<"False Positive: "<<falsePos<<std::endl;
    std::cout<<"False Negative: "<<falseNeg<<std::endl;
    
    std::cout<<std::endl<<"Accuracy: "<<(corr/(double)predSize)*100<<"%"<<std::endl<<std::endl;
    
    std::cout<<"----------------------"<<std::endl;
    std::cout<<"Within the new margin and 0"<<std::endl;
    std::cout<<"----------------------"<<std::endl;
    std::cout<<"Correct: "<<corrNwMgn<<std::endl;
    std::cout<<"Incorrect: "<<incorrNwMgn<<std::endl<<std::endl<<std::endl;
    
    return 0;
    
}

int main() {
    
    net4();
    
    return 0;
}

