/* 
 * File:   PoolLayer.hpp
 * Author: heshan
 *
 * Created on June 7, 2018, 2:05 PM
 */

#ifndef POOLLAYER_HPP
#define POOLLAYER_HPP

#include <iostream>
#include <Eigen>

class PoolLayer {
public:
    PoolLayer();
    /**
     * Constructor
     * 
     * @param dimensions: dimensions of the input matrix (depth, height, width)
     * @param poolW: pool matrix width
     * @param poolH: pool matrix height
     */
    PoolLayer(std::tuple<int, int, int> dimensions, int poolW, int poolH);
    /**
     * 
     * @param orig
     */
    PoolLayer(const PoolLayer& orig);
    /**
     * 
     */
    virtual ~PoolLayer();
    /**
     * Initialize output matrix
     * 
     * @return 0
     */
    int initMat();
    /**
     * 
     * @param input: input image or matrix
     * @return 
     */
    Eigen::MatrixXd * pool(Eigen::MatrixXd * input);
    /**
     * Return the dimension of the layer output
     * 
     * @return a tuple
     */
    std::tuple<int, int, int> getOutputDims();
private:
    
public:
    int height, width;
    int depth, outHeight, outWidth, poolW, poolH;
    Eigen::MatrixXd ** maxIndices;
    Eigen::MatrixXd * output;
    

};

#endif /* POOLLAYER_HPP */

