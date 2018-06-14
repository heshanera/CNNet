/* 
 * File:   Activation.cpp
 * Author: heshan
 * 
 * Created on June 7, 2018, 11:17 AM
 */

#include "Activation.hpp"
#include <iostream>

Activation::Activation() { }

Activation::Activation(const Activation& orig) { }

Activation::~Activation() { }

double Activation::sigmoid(double x) {
    return 1.0/(1.0 + std::exp(-x));
}
    
Eigen::MatrixXd Activation::sigmoid(Eigen::MatrixXd mat){
    for(int x = 0; x < mat.cols(); x++){
        for(int y = 0; y < mat.rows(); y++){
            mat(y,x) = 1.0/(1.0 + std::exp(-mat(y,x)));
        }
    }
    return mat;
}

double Activation::sigmoidDeriv(double x) {
    return sigmoid(x) * (1-sigmoid(x));
}
    
Eigen::MatrixXd Activation::sigmoidDeriv(Eigen::MatrixXd mat){
    for(int x = 0; x < mat.cols(); x++){
        for(int y = 0; y < mat.rows(); y++){
            mat(y,x) = (sigmoid(mat(y,x)) * (1-mat(y,x)));
        }
    }
    return mat;
}

Eigen::MatrixXd Activation::maxPoolDelta(
    double poolOutVal, 
    double deltaVal, 
    Eigen::MatrixXd poolBlock,
    int poolDim1,
    int poolDim2
) {
    
    for (int i = 0; i < poolDim1; i++) {
        for (int j = 0; j < poolDim2; j++) {
            if ( poolBlock(i,j) < poolOutVal ) poolBlock(i,j) = 0;
            else poolBlock(i,j) = deltaVal;
        }
    }
    
    return poolBlock;
}

