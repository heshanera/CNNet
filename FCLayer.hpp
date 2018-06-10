/* 
 * File:   FCLayer.hpp
 * Author: heshan
 *
 * Created on June 7, 2018, 8:46 PM
 */

#ifndef FCLAYER_HPP
#define FCLAYER_HPP

#include <iostream>
#include <Eigen>
#include "Activation.hpp"

class FCLayer {
public:
    FCLayer();
    /**
     * Constructor
     * 
     * @param dimensions: dimensions of the input matrix (depth, height, width)
     * @param outputs: no of outputs
     */
    FCLayer(std::tuple<int, int, int> dimensions, int outputs);
    /**
     * 
     * @param orig
     */
    FCLayer(const FCLayer& orig);
    /**
     * 
     */
    virtual ~FCLayer();
    /**
     * Initialize weight matrix and bias values
     * 
     * @return 0
     */
    int initMat();
    /**
     * 
     * @param input: input matrix 
     * @return 
     */
    Eigen::MatrixXd * forward(Eigen::MatrixXd * input);
    /**
     * Return the dimension of the layer output
     * 
     * @return a tuple
     */
    std::tuple<int, int, int> getOutputDims();
private:
public:
    int height, width, depth,outputs;
    Eigen::MatrixXd ** weights;
    Eigen::MatrixXd bias;
    Eigen::MatrixXd * output; // outputs before activation
    Eigen::MatrixXd * activatedOut;
    
    
};

#endif /* FCLAYER_HPP */

