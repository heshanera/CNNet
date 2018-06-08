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
    

