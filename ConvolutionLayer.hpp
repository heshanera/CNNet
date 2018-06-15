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
    ConvolutionLayer();
    /**
     * Constructor
     * 
     * @param dimensions: dimensions of the input matrix (depth, height, width)
     * @param filterSize: size of the filter N, (N x N) 
     * @param stride: displacement units of the filter
     * @param noOfFilters: no of filters
     * @param padding
     */
    ConvolutionLayer(std::tuple<int, int, int> dimensions, int filterSize, int stride, int noOfFilters, int padding);
    /**
     * 
     * @param orig
     */
    ConvolutionLayer(const ConvolutionLayer& orig);
    /**
     */
    virtual ~ConvolutionLayer();
    /**
     * Initialize weight matrices and the bias values for the convolutional layer
     * 
     * @return 0
     */
    int initMat();
    /**
     * Apply the convolution operation to the image using the generated filters
     * 
     * @param input: input image or matrix
     * @return an array of convoluted images (array size = no of filters)
     */
    Eigen::MatrixXd * convolute(Eigen::MatrixXd * input);
    /**
     * Return the dimension of the layer output
     * 
     * @return a tuple
     */
    std::tuple<int, int, int> getOutputDims();
    
private:
    int height, width, padding = 0;
    
public:
    int stride, filterSize;
    int depth, noOfFilters;
    int outHeight, outWidth;
    double * bias;
    Eigen::MatrixXd ** filters;
    Eigen::MatrixXd * output;
    Eigen::MatrixXd * activatedOut;
    
};

#endif /* CONVOLUTIONLAYER_HPP */

