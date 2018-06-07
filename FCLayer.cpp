/* 
 * File:   FCLayer.cpp
 * Author: heshan
 * 
 * Created on June 7, 2018, 8:46 PM
 */

#include "FCLayer.hpp"

FCLayer::FCLayer(int depth, int height, int width, int outputs) {
    this->depth = depth;
    this->height = height;
    this->width = width;
    this->outputs = outputs;
    initMat();
}

FCLayer::FCLayer(const FCLayer& orig) { }

FCLayer::~FCLayer() { }

int FCLayer::initMat() {
    target = Eigen::RowVectorXd::Ones(outputs);
    bias = Eigen::RowVectorXd::Ones(outputs);
    
    weights = new Eigen::MatrixXd * [outputs];
    for(int i = 0; i < outputs; i++) {
        weights[i] = new Eigen::MatrixXd[depth];
        for(int j = 0; j < depth; j++) {
            weights[i][j] = Eigen::MatrixXd(height,width);
        }
    }
    return 0;
}

Eigen::RowVectorXd FCLayer::forward() {

    return (Eigen::RowVectorXd::Ones(outputs));
}

