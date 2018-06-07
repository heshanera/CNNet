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
    /**
     * Constructor
     * 
     * @param depth: depth of the input matrix
     * @param height: height of the input matrix
     * @param width: width of the input matrix
     * @param poolW: pool matrix width
     * @param poolH: pool matrix height
     */
    PoolLayer(int depth, int height, int width, int poolW, int poolH);
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
private:
    int depth, height, width, poolW, poolH;
    int outHeight, outWidth;
    Eigen::MatrixXd * output;
    Eigen::MatrixXd ** maxIndices;

};

#endif /* POOLLAYER_HPP */

