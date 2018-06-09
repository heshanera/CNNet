/* 
 * File:   CNN.hpp
 * Author: heshan
 *
 * Created on June 9, 2018, 10:17 AM
 */

#ifndef CNN_HPP
#define CNN_HPP

#include "ConvolutionLayer.hpp"
#include "PoolLayer.hpp"
#include "FCLayer.hpp"

struct NetLayers{
    int numCL = 0;
    int numPL = 0;
    int numFCL = 0;
    ConvolutionLayer * CL;
    PoolLayer * PL;
    FCLayer * FCL;
};

class CNN {
public:
    /**
     * Initialize network layer to given structure
     * 
     * @param dimensions: dimension of the input matrix (depth, height, width)
     * @param netStruct: structure of the network layers with order and parameters
     */
    CNN(std::tuple<int, int, int> dimensions, struct NetStruct netStruct);
    CNN(const CNN& orig);
    virtual ~CNN();
    
    int forward(Eigen::MatrixXd * input);
    int backprop();
    
    int train(Eigen::MatrixXd * input);
    
    int tmp();
private:
    int layers;
    char * layerOrder;
    struct::NetLayers netLayers;
};

struct ConvLayStruct {
    int filterSize;
    int stride;
    int filters;
};

struct PoolLayStruct {
    int poolH;
    int poolW;
};

struct FCLayStruct {
    int outputs;
    int classes;
};

struct NetStruct {
    int layers;
    char * layerOrder;
    struct::ConvLayStruct * CL;
    struct::PoolLayStruct * PL;
    struct::FCLayStruct * FCL;
};



#endif /* CNN_HPP */

