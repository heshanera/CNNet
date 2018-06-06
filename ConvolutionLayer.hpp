/* 
 * File:   ConvolutionLayer.hpp
 * Author: heshan
 *
 * Created on June 6, 2018, 6:26 PM
 */

#ifndef CONVOLUTIONLAYER_HPP
#define CONVOLUTIONLAYER_HPP

#include <Eigen>
#include <iostream>

class ConvolutionLayer {
public:
    /**
     * initialize weight matrices and the bias values for the convolutional layer
     * 
     * @param depth: depth of the input matrix
     * @param height: height of the input matrix
     * @param width: width of the input matrix
     * @param filterSize: size of the filter N, (N x N) 
     * @param stride
     * @param noOfFilters: no of filters
     * @param padding
     */
    ConvolutionLayer(int depth, int height, int width, int filterSize, int stride, int noOfFilters, int padding);
    ConvolutionLayer(const ConvolutionLayer& orig);
    virtual ~ConvolutionLayer();
private:
    int depth, height, width, filterSize, stride, noOfFilters, padding = 0;
    int outHeight, outWidth;
    Eigen::MatrixXd ** filters;
    Eigen::VectorXd bias;
    Eigen::MatrixXd * zValues;
    Eigen::MatrixXd * output;
    
};

#endif /* CONVOLUTIONLAYER_HPP */

