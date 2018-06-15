/* 
 * File:   FCLayer.cpp
 * Author: heshan
 * 
 * Created on June 7, 2018, 8:46 PM
 */

#include "FCLayer.hpp"

FCLayer::FCLayer() { }

FCLayer::FCLayer(std::tuple<int, int, int> dimensions, int outputs) {
    this->depth = std::get<0>(dimensions);
    this->height = std::get<1>(dimensions);
    this->width = std::get<2>(dimensions);
    this->outputs = outputs;
    initMat();
}

FCLayer::FCLayer(const FCLayer& orig) { }

FCLayer::~FCLayer() { }

int FCLayer::initMat() {
    
    output = new Eigen::MatrixXd[1];
    activatedOut = new Eigen::MatrixXd[1];
    
    Eigen::MatrixXd outMat = Eigen::MatrixXd::Zero(outputs,1);
    output[0] = outMat;
    activatedOut[0] = outMat;

//    output = Eigen::MatrixXd(outputs,1); 
    bias = Eigen::MatrixXd::Random(outputs,1); 
    bias*0.01;
    
    weights = new Eigen::MatrixXd * [outputs];
    for(int i = 0; i < outputs; i++) {
        weights[i] = new Eigen::MatrixXd[depth];
        for(int j = 0; j < depth; j++) {
            weights[i][j] = Eigen::MatrixXd::Random(height,width);
            weights[i][j]*0.01;
        }
    }
    return 0;
}

Eigen::MatrixXd * FCLayer::forward(Eigen::MatrixXd * input) {

    // reshaping the weight matrix
    Eigen::MatrixXd weightsVecMat(outputs,depth*height*width);
    for (int i = 0; i < outputs; i++) {
        Eigen::RowVectorXd weightsVec;
        Eigen::RowVectorXd weightsTmpVec;
        for (int j = 0; j < depth; j++) {
            weightsTmpVec = weightsVec;
            weightsVec = Eigen::RowVectorXd((j+1)*height*width);
            Eigen::Map<Eigen::RowVectorXd> weightsV(weights[i][j].data(), weights[i][j].size());
            weightsVec<<weightsTmpVec, weightsV;
        }
        weightsVecMat.row(i) = weightsVec;
    }
    
    // reshaping the input matrix
    Eigen::MatrixXd inputVecMat(depth*height*width,1);
    Eigen::RowVectorXd inputArr;
    Eigen::RowVectorXd inputTmpVec;
    for (int i = 0; i < depth; i++) {
        inputTmpVec = inputArr;
        inputArr = Eigen::RowVectorXd((i+1)*height*width);
        Eigen::Map<Eigen::RowVectorXd> inputV(input[i].data(), input[i].size());
        inputArr<<inputTmpVec, inputV;
    }
    inputVecMat.col(0) = inputArr;
    output[0] = (weightsVecMat*inputVecMat) + bias;
    activatedOut[0] = Activation::sigmoid(output[0]);
    
//    std::cout<<output<<"\n";
//    for(int i = 0; i < outputs; i++) {
//        for(int j = 0; j < depth; j++) {
//            std::cout<<weights[i][j]<<"\n**** "<<j<<" ****\n";
//        }
//        std::cout<<"\n------------------------ "<<i<<" ---------\n";
//    }
    
    return activatedOut;
}

std::tuple<int, int, int> FCLayer::getOutputDims() {
    return std::make_tuple(1, outputs, 1);
}
