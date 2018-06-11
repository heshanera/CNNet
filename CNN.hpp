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
    int backprop(Eigen::MatrixXd * input, Eigen::MatrixXd label);
    int train(Eigen::MatrixXd ** inputs, Eigen::MatrixXd * labels);
    
    /**
     * Back propagation from final layer to previous fully connected layer
     * 
     * @param prevDelta: previous delta values 
     * @param prevActivOut: previous layer activations
     * @return tuple of deltaW and new previous delta value
     */
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> backPropgateLayer(
        Eigen::MatrixXd prevDelta, 
        Eigen::MatrixXd * prevActivOut
    );
    /**
     * Back propagation from fully connected layer to fully connected layer
     * 
     * @param prevDelta: previous delta values 
     * @param prevWeight: previous layer weights
     * @param prevActivOut: previous layer activations
     * @param preOut
     * @return tuple of deltaW and new delta value
     */
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> backPropgateLayer(
        Eigen::MatrixXd prevDelta, 
        Eigen::MatrixXd ** prevWeight,
        Eigen::MatrixXd * prevActivOut,
        Eigen::MatrixXd * preOut
    );
    /**
     * Back propagation from fully connected layer to the pooling layer
     * 
     * @param prevDelta: previous delta values 
     * @param prevWeight: previous layer weights
     * @param prevActivOut: previous layer activations
     * @param preOut
     * @return 
     */
    Eigen::MatrixXd backPropgateToPool(
        Eigen::MatrixXd prevDelta, 
        Eigen::MatrixXd ** prevWeight,
        Eigen::MatrixXd * prevActivOut,
        Eigen::MatrixXd * preOut
    );
    /**
     * Back propagate from pool layer to the convolutional layer
     * 
     * @param prevDelta: previous delta values 
     * @param prevWeight: previous layer weights
     * @param prevActivOut: previous layer activations
     * @param maxIndices: indices of the max positions
     * @param poolH: pool window width
     * @param poolW: pool window width
     * @param output
     * @return 
     */
    Eigen::MatrixXd backPropgateToConv(
        Eigen::MatrixXd prevDelta, 
        Eigen::MatrixXd ** prevWeight,
        Eigen::MatrixXd * prevActivOut,
        Eigen::MatrixXd ** maxIndices,
        int poolH, int poolW,
        Eigen::MatrixXd * output
    );
    
    
private:
    int layers;
    char * layerOrder;
    struct::NetLayers netLayers;
    Eigen::MatrixXd forwardOut;
    // for back propagation 
    Eigen::MatrixXd ** weights;
    Eigen::MatrixXd * output;
    Eigen::MatrixXd * activatedOut;
    // back propagation in fully connected layers
    int depth, outputs;
    // back propagation in pooling layers
    int poolDepth, outHeight, outWidth;
    Eigen::MatrixXd * poolDeltaW;
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

