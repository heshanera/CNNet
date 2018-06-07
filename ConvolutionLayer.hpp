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
#include "Activation.hpp"

class ConvolutionLayer {
public:
    /**
     * Initialize weight matrices and the bias values for the convolutional layer
     * 
     * @param depth: depth of the input matrix
     * @param height: height of the input matrix
     * @param width: width of the input matrix
     * @param filterSize: size of the filter N, (N x N) 
     * @param displace: displacement units of the filter
     * @param noOfFilters: no of filters
     * @param padding
     */
    ConvolutionLayer(int depth, int height, int width, int filterSize, int displace, int noOfFilters, int padding);
    /**
     * 
     * @param orig
     */
    ConvolutionLayer(const ConvolutionLayer& orig);
    /**
     */
    virtual ~ConvolutionLayer();
    
    /**
     * Apply the convolution operation to the image using the generated filters
     * 
     * @param input: input image or matrix
     * @return an array of convoluted images (array size = no of filters)
     */
    Eigen::MatrixXd * convolute(Eigen::MatrixXd * input);
    
private:
    int depth, height, width, filterSize, displace, noOfFilters, padding = 0;
    int outHeight, outWidth;
    double * bias;
    Eigen::MatrixXd ** filters;
    Eigen::MatrixXd * output;
    
};

#endif /* CONVOLUTIONLAYER_HPP */

