/* 
 * File:   CNN.cpp
 * Author: heshan
 * 
 * Created on June 9, 2018, 10:17 AM
 */

#include "CNN.hpp"
#include "IProc/IProc.h"

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
    
    for (int i = 0; i < layers; i++) {
        CLpos = 0,PLpos = 0,FCLpos = 0;

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
                FCLpos++;
                break;    
            }    
        }
    }

}

CNN::CNN(const CNN& orig) { }

CNN::~CNN() { }

int CNN::forward(Eigen::MatrixXd * layerInput) {

    int CLpos,PLpos,FCLpos;
    for (int i = 0; i < layers; i++) {
        CLpos = 0,PLpos = 0,FCLpos = 0;

        switch (layerOrder[i]) {
            case 'C':
            {    
                std::cout<<"Forward Pass: Convolution Layer"<<CLpos+1<<"...\n";
                layerInput = netLayers.CL[CLpos].convolute(layerInput);
                /////////////////////////////////////////////////////////////
                /**                
                for (int i = 0; i < 3; i++) {
                    writeImg(layerInput[i]);
                }
                **/ 
                /////////////////////////////////////////////////////////////
                CLpos++;
                break;
            }
            case 'P':
            {    
                std::cout<<"Forward Pass: Pool Layer"<<PLpos+1<<"...\n";
                
                PLpos++;
                break;
            }
            case 'F':
            {    
                std::cout<<"Forward Pass:  Connected Layer"<<FCLpos+1<<"...\n";
                
                FCLpos++;
                break;    
            }    
        }
        
    }
    
    return 0;
}

int CNN::backprop() {

    return 0;
}

int CNN::train(Eigen::MatrixXd * inputArr) {
    forward(inputArr);
    return 0;
}


int CNN::tmp() {
    ///////////////////////////////////////////////////////////////
    
    std::tuple<int, int, int> dimensions = std::make_tuple(1,20,20);
    
    int filterSize = 5;
    int stride = 1; //
    int noOfFilters = 5;
    int padding = 0; //
    
    ConvolutionLayer cl(dimensions, filterSize, stride, noOfFilters, padding);
    
    Eigen::MatrixXd input[1];
    Eigen::MatrixXd img = Eigen::MatrixXd::Ones(20,20);
    input[0] = img;
    
//    Eigen::MatrixXd convolutedOut[]  = cl.convolute(input);
  
    ///////////////////////////////////////////////////////////////
    
    int poolW = 2;
    int poolH = 2;
    
    dimensions = cl.getOutputDims();
    
    PoolLayer pl(dimensions, poolW, poolH);
    Eigen::MatrixXd * inputFcl = pl.pool(cl.convolute(input));
    
    ///////////////////////////////////////////////////////////////
    
    int outputs = 2;
    dimensions = pl.getOutputDims();
    
    FCLayer fcl(dimensions, outputs);
    fcl.forward(inputFcl);
    
    ///////////////////////////////////////////////////////////////
    
    
    return 0;
}
