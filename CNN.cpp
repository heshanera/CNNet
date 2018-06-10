/* 
 * File:   CNN.cpp
 * Author: heshan
 * 
 * Created on June 9, 2018, 10:17 AM
 */

#include "CNN.hpp"

CNN::CNN(std::tuple<int, int, int> dimensions, struct NetStruct netStruct) { 
    
    int padding = 0; 
    int CLpos,PLpos,FCLpos;
    this->layerOrder = netStruct.layerOrder;
    this->layers = netStruct.layers;
    
    for (int i = 0; i < layers; i++) {
        switch (layerOrder[i]) {
            case 'C':
                netLayers.numCL++;
                break;
            case 'P':
                netLayers.numPL++;
                break;
            case 'F':
                netLayers.numFCL++;
                break;    
        }
    }
    netLayers.CL = new ConvolutionLayer[netLayers.numCL];
    netLayers.PL = new PoolLayer[netLayers.numPL];
    netLayers.FCL = new FCLayer[netLayers.numFCL];
    
    CLpos = 0,PLpos = 0,FCLpos = 0;
    for (int i = 0; i < layers; i++) {

        switch (layerOrder[i]) {
            case 'C':
            {    
                std::cout<<"Initializing Convolution Layer"<<CLpos+1<<"...\n";
                netLayers.CL[CLpos] = ConvolutionLayer(
                    dimensions,
                    netStruct.CL[CLpos].filterSize, 
                    netStruct.CL[CLpos].stride, 
                    netStruct.CL[CLpos].filters, 
                    padding
                );
                dimensions = netLayers.CL[CLpos].getOutputDims();
                CLpos++;
                break;
            }
            case 'P':
            {    
                std::cout<<"Initializing Pool Layer"<<PLpos+1<<"...\n";
                netLayers.PL[PLpos] = PoolLayer(
                    dimensions, 
                    netStruct.PL[PLpos].poolW, 
                    netStruct.PL[PLpos].poolH
                );
                dimensions = netLayers.PL[PLpos].getOutputDims();
                PLpos++;
                break;
            }
            case 'F':
            {    
                std::cout<<"Initializing Fully Connected Layer"<<FCLpos+1<<"...\n";
                netLayers.FCL[FCLpos] = FCLayer(
                    dimensions, 
                    netStruct.FCL[FCLpos].outputs
                );
                dimensions = netLayers.FCL[FCLpos].getOutputDims();
                FCLpos++;
                break;    
            }    
        }
    }

}

CNN::CNN(const CNN& orig) { }

CNN::~CNN() { }

int CNN::forward(Eigen::MatrixXd * layerInput) {

    int CLpos = 0,PLpos = 0,FCLpos = 0;
    for (int i = 0; i < layers; i++) {

        switch (layerOrder[i]) {
            case 'C':
            {    
                std::cout<<"Forward Pass: Convolution Layer"<<CLpos+1<<"...\n";
                layerInput = netLayers.CL[CLpos].convolute(layerInput);
                /////////////////////////////////////////////////////////////
                                
                // Write convoluted images 
                /*
                std::tuple<int, int, int> dimensions = netLayers.CL[CLpos].getOutputDims();
                for (int i = 0; i < std::get<0>(dimensions); i++) {
                    std::cout<<layerInput[i]<<std::endl<<std::endl;
                    //writeImg(layerInput[i]);
                } 
                */
                /////////////////////////////////////////////////////////////
                CLpos++;
                break;
            }
            case 'P':
            {    
                std::cout<<"Forward Pass: Pool Layer"<<PLpos+1<<"...\n";
                layerInput = netLayers.PL[PLpos].pool(layerInput);
                /////////////////////////////////////////////////////////////
                // Write pooled results
                /*       
                std::tuple<int, int, int> dimensions = netLayers.CL[CLpos].getOutputDims();
                for (int i = 0; i < std::get<0>(dimensions); i++) {
                    std::cout<<layerInput[i]<<std::endl<<std::endl;
                    //writeImg(layerInput[i]);
                } 
                */
                /////////////////////////////////////////////////////////////
                PLpos++;
                break;
            }
            case 'F':
            {    
                std::cout<<"Forward Pass: Fully Connected Layer"<<FCLpos+1<<"...\n";
                layerInput = netLayers.FCL[FCLpos].forward(layerInput);
                FCLpos++;
                break;    
            }    
        }
    }
    forwardOut = layerInput[0];
    return 0;
}

int CNN::backprop(Eigen::MatrixXd * input, Eigen::MatrixXd label) {

    double learningRate = 0.1;
    
    Eigen::MatrixXd outDeriv = netLayers.FCL[netLayers.numFCL-1].output[0];
    outDeriv = Activation::sigmoidDeriv(outDeriv);
    Eigen::MatrixXd delta = (forwardOut-label).array() * outDeriv.array();
    
    weights = netLayers.FCL[netLayers.numFCL-1].weights;
    output = netLayers.FCL[netLayers.numFCL-1].output;
    activatedOut = netLayers.FCL[netLayers.numFCL-1].activatedOut;
    
    int layer = (layers-1);
    int CLpos = netLayers.numCL-1;
    int PLpos = netLayers.numPL-1;
    int FCLpos = netLayers.numFCL - 1;
    
    int CLpos2 = netLayers.numCL-1;
    int PLpos2 = netLayers.numPL-1;
    int FCLpos2 = netLayers.numFCL - 2;
    
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> res;
    ////// final layer /////////////////////////
//    res = backPropgateLayer(delta, activatedOut);
//    delta = std::get<1>(res);
//    weights = netLayers.FCL[FCLpos].weights;
    ///////////////////////////////////////////
    
    for (layer; layer >= 0; layer--) {
        
        // Previous layer
        switch (layerOrder[layer-1]) {
            case 'C':
            {    
//                activatedOut = netLayers.CL[CLpos2].activatedOut;
//                CLpos2--;
//                break;
            }
            case 'P':
            {    
//                activatedOut = netLayers.PL[CLpos2].activatedOut;
//                PLpos2--;
//                break;
            }
            case 'F':
            {    
                activatedOut = netLayers.FCL[FCLpos2].activatedOut;
                FCLpos2--;
                break;    
            }    
        }
        
        
        switch (layerOrder[layer]) {
            case 'C':
            {    
                std::cout<<"Backward Pass from Convolution Layer"<<CLpos+1<<"...\n";
                CLpos--;
                break;
            }
            case 'P':
            {    
                std::cout<<"Backward Pass from Pool Layer"<<PLpos+1<<"...\n";
                PLpos--;
                break;
            }
            case 'F':
            {
                std::cout<<"Backward Pass from Fully Connected Layer"<<FCLpos+1<<"...\n";
                if ( layerOrder[layer-1] == 'F' ) {
                    Eigen::MatrixXd deltaW;

                    if ( layer == (layers-1)) {
                        res = backPropgateLayer(delta, activatedOut);
                        deltaW = std::get<0>(res);
                        delta = std::get<1>(res);

                    } else {
                        output = netLayers.FCL[FCLpos].output;
                        res = backPropgateLayer(delta, weights, activatedOut, output);

                        deltaW = std::get<0>(res);
                        delta = std::get<1>(res);

                        for(int i = 0; i < outputs; i++) {
                            for(int j = 0; j < depth; j++) {

                            }
                        }
                    }
                    weights = netLayers.FCL[FCLpos].weights;
                    outputs = netLayers.FCL[FCLpos].outputs;
                    depth = netLayers.FCL[FCLpos].depth;

                    // adjusting the weight matrix
                    int h = netLayers.FCL[FCLpos].height;
                    int w = netLayers.FCL[FCLpos].width;
                    for(int i = 0; i < outputs; i++) {
                        for(int j = 0; j < depth; j++) {
                            for(int a = 0; a < w; a++) {
                                for(int b = 0; b < h; b++) {
                                    weights[i][j](b,a) -= (deltaW(a,b) * learningRate);
                                }
                            }
                        }
                    }
                    
                    // adjusting the bia values
                    netLayers.FCL[FCLpos].bias -= delta;
                    
                    FCLpos--;
                } else if ( layerOrder[layer-1] != 'F' ) {
                    std::cout<<"\n\n******\n3d to 1d\n*******\n\n";
                }   
                break;    
            }    
        }
        
        if ( layerOrder[layer] != 'P' ) {
             
        }
    }
    return 0;
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> CNN::backPropgateLayer(
        Eigen::MatrixXd prevDelta,
        Eigen::MatrixXd * prevActivOut
) {
    
    Eigen::MatrixXd deltaW = prevDelta*prevActivOut[0].transpose();
    return std::make_tuple(deltaW,prevDelta);
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> CNN::backPropgateLayer(
        Eigen::MatrixXd prevDelta, 
        Eigen::MatrixXd ** prevWeight,
        Eigen::MatrixXd * prevActivOut,
        Eigen::MatrixXd * preOut
) {
    preOut[0] = Activation::sigmoidDeriv(preOut[0]);
    Eigen::MatrixXd weightMat;
    for(int i = 0; i < outputs; i++) {
        for(int j = 0; j < depth; j++) {
            Eigen::Map<Eigen::VectorXd> prevWeightV(prevWeight[i][j].data(), prevWeight[i][j].size());
            weightMat.conservativeResize(prevWeightV.size(), weightMat.cols()+1);
            weightMat.col(weightMat.cols()-1) = prevWeightV;
        }
    }
    prevDelta = (weightMat*prevDelta).array() * preOut[0].array();
    Eigen::MatrixXd deltaW = prevDelta*prevActivOut[0].transpose();
    return std::make_tuple(deltaW,prevDelta);
}

int CNN::train(Eigen::MatrixXd ** inputs, Eigen::MatrixXd * labels) {
    int noOfInputs = 1;
    for (int i = 0; i < noOfInputs; i++) {
        forward(inputs[i]);
        backprop(inputs[i],labels[i]);
    }
    
    return 0;
}
