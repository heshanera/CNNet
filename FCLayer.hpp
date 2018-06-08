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
    /**
     * Constructor
     * 
     * @param depth: depth of the input matrix
     * @param height: height of the input matrix
     * @param width: width of the input matrix
     * @param outputs: no of outputs
     */
    FCLayer(int depth, int height, int width, int outputs);
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
    Eigen::RowVectorXd forward(Eigen::MatrixXd * input);
private:
    int depth, height, width, outputs;
    Eigen::MatrixXd ** weights;
    Eigen::MatrixXd bias;
    Eigen::MatrixXd output;
    
};

#endif /* FCLAYER_HPP */

