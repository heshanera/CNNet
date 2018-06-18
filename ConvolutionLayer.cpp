/* 
 * File:   ConvolutionLayer.cpp
 * Author: heshan
 * 
 * Created on June 6, 2018, 6:26 PM
 */

#include "ConvolutionLayer.hpp"

ConvolutionLayer::ConvolutionLayer() { }

ConvolutionLayer::ConvolutionLayer(std::tuple<int, int, int> dimensions, int filterSize, int stride, int noOfFilters, int padding = 0) { 
    
    this->depth = std::get<0>(dimensions);
    this->height = std::get<1>(dimensions);
    this->width = std::get<2>(dimensions);
    this->filterSize = filterSize;
    this->stride = stride;
    this->noOfFilters = noOfFilters;
    this->padding = padding;
    initMat();
}

ConvolutionLayer::ConvolutionLayer(const ConvolutionLayer& orig) { }

ConvolutionLayer::~ConvolutionLayer() { }

int ConvolutionLayer::initMat() {

    // generating filter weight matrix array
    // depth * filterSize * filterSize * noOfFilters
    
    filters = new Eigen::MatrixXd * [depth];
    for(int i = 0; i < depth; i++) {
        filters[i] = new Eigen::MatrixXd[noOfFilters];
    }
    
    double max, min, diff;
    min = -0.01;
    max = 0.01;
    diff = max - min;
    
//    srand(time(NULL));  // if not random numbers are generated in the same order
    for (int i = 0; i < depth; i++) {
        for (int j = 0; j < noOfFilters; j++) {
            Eigen::MatrixXd filter(filterSize,filterSize);
            for (int x = 0; x < filterSize; x++) {
                for (int y = 0; y < filterSize; y++) {
                    filter(x,y) = min + ((double)rand() / RAND_MAX) * diff;
                }
            }
            filters[i][j] = filter;
        }    
    }
    
    // generating bias values
    bias = new double[noOfFilters];
    for (int i = 0; i < noOfFilters; i++) {
        bias[i] = (min + ((double)rand() / RAND_MAX) * diff);
    }
    
    this->outHeight = (int)((height - filterSize + 2*padding)/stride) + 1; // rows
    this->outWidth = (int)((width - filterSize + 2*padding)/stride) + 1; // columns
   
    output = new Eigen::MatrixXd[noOfFilters*depth];
    activatedOut = new Eigen::MatrixXd[noOfFilters*depth];
    for (int i = 0; i < noOfFilters*depth; i++) {
        Eigen::MatrixXd outVal = Eigen::MatrixXd::Zero(outHeight,outWidth);
        output[i] = outVal;
        activatedOut[i] = outVal;
    }
    return 0;
}


Eigen::MatrixXd *  ConvolutionLayer::convolute(Eigen::MatrixXd * input) {

    for (int j = 0; j < depth; j++) {
        for (int i = 0; i < noOfFilters; i++) {
            Eigen::MatrixXd filter = filters[j /*depth*/][i /*filter*/];
            Eigen::Map<Eigen::RowVectorXd> filterV(filter.data(), filter.size());

            for (int x = 0; x < outWidth; x+=stride) {
                for (int y = 0; y < outHeight; y+=stride) {
                    Eigen::MatrixXd inputBlock = input[j /*depth*/].block(y,x,filterSize,filterSize);
                    Eigen::Map<Eigen::RowVectorXd> inputBlockV(inputBlock.data(), inputBlock.size());                   
                    output[(j*noOfFilters) + i](y,x) = (inputBlockV.dot(filterV) + bias[i]);   
                    activatedOut[(j*noOfFilters) + i](y,x) = Activation::sigmoid(output[(j*noOfFilters) + i](y,x));
                }
            }
        }    
    }

//    for (int i = 0; i < noOfFilters; i++) {
//        std::cout << "Here is the matrix m:\n" << output[i] << std::endl;
//        std::cout <<"---------------------------------\n";
//    }    
    
    return activatedOut;
}

std::tuple<int, int, int> ConvolutionLayer::getOutputDims() {
    return std::make_tuple((noOfFilters*depth), outHeight, outWidth);
}
