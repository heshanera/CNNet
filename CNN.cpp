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
//                std::cout<<"Forward Pass: Convolution Layer"<<CLpos+1<<"...\n";
                layerInput = netLayers.CL[CLpos].convolute(layerInput);
//                std::cout<<layerInput[0]<<"*******\n\n";
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
//                std::cout<<"Forward Pass: Pool Layer"<<PLpos+1<<"...\n";
                layerInput = netLayers.PL[PLpos].pool(layerInput);
//                std::cout<<layerInput[0]<<"*******\n\n";
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
//                std::cout<<"Forward Pass: Fully Connected Layer"<<FCLpos+1<<"...\n";
                layerInput = netLayers.FCL[FCLpos].forward(layerInput);
//                std::cout<<layerInput[0]<<"*******\n\n";
                FCLpos++;
                if (FCLpos == netLayers.numFCL) {
                    classPredicts = layerInput[0];
//                    std::cout<<classPredicts<<"\n******\n";
                }
                break;    
            }    
        }
    }
    forwardOut = layerInput[0];
    return 0;
}

int CNN::backprop(Eigen::MatrixXd * input, Eigen::MatrixXd label) {

    Eigen::MatrixXd outDeriv = netLayers.FCL[netLayers.numFCL-1].output[0];
    outDeriv = Activation::sigmoidDeriv(outDeriv);
    Eigen::MatrixXd delta = (forwardOut-label).array() * outDeriv.array();
    
//    std::cout<<forwardOut<<"\n";
//    std::cout<<label<<"\n";
//    std::cout<<outDeriv<<"\n";    
    std::cout<<delta<<"\n\n";
    
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
    
    for (layer; layer >= 0; layer--) {
        
        // Previous layer
        if (layer-1 < 0) { 
            activatedOut = input;
            poolDepth = 1;
        }    
        else {
            switch (layerOrder[layer-1]) {
                case 'C':
                {    
                    activatedOut = netLayers.CL[CLpos2].activatedOut;
                    convDepth = netLayers.CL[CLpos2].depth;
                    outHeight = netLayers.CL[CLpos2].outHeight;
                    outWidth = netLayers.CL[CLpos2].outWidth;
                    
                    CLpos2--;
                    break;
                }
                case 'P':
                {    
                    activatedOut = netLayers.PL[PLpos2].output;
                    poolDepth = netLayers.PL[PLpos2].depth;
                    outHeight = netLayers.PL[PLpos2].outHeight;
                    outWidth = netLayers.PL[PLpos2].outWidth;
                    PLpos2--;
                    break;
                }
                case 'F':
                {    
                    activatedOut = netLayers.FCL[FCLpos2].activatedOut;
                    FCLpos2--;
                    break;    
                }    
            }
        }
        
        switch (layerOrder[layer]) {
            case 'C':
            {    
//                std::cout<<"Backward Pass from Convolution Layer"<<CLpos+1<<"...\n";
                
                int stride = netLayers.CL[CLpos].stride;
                int noOfFilters = netLayers.CL[CLpos].noOfFilters;
                int filterSize = netLayers.CL[CLpos].filterSize;
//                Eigen::MatrixXd biasDelta;
                delta = backPropgateToFilters(weights,stride,filterSize,activatedOut);
                
                outputs = poolDepth;
                depth = noOfFilters;
                weights = new Eigen::MatrixXd * [poolDepth];
//                delta = Eigen::MatrixXd(poolDepth,1);
                for (int i = 0; i < poolDepth; i++) {
                    weights[i] = new Eigen::MatrixXd[noOfFilters];
                    for (int j = 0; j < noOfFilters; j++) {
                        weights[i][j] = netLayers.CL[CLpos].filters[i][j];
                        netLayers.CL[CLpos].filters[i][j] += (filterDelta[i]*learningRate);
                    }    
                }
                
                for (int i = 0; i < noOfFilters; i++) {
                    for (int j = 0; j < poolDepth; j++) {
                        netLayers.CL[CLpos].bias[i] += (delta(j,0) * learningRate);
                    }
                } 
                
                CLpos--;
                break;
            }
            case 'P':
            {    
//                std::cout<<"Backward Pass from Pool Layer"<<PLpos+1<<"...\n";
                
                outHeightC = netLayers.PL[PLpos].outHeight;
                outWidthC = netLayers.PL[PLpos].outWidth;
                poolDepth = netLayers.PL[PLpos].depth;
                int poolH = netLayers.PL[PLpos].poolH;
                int poolW = netLayers.PL[PLpos].poolW;
                output = netLayers.PL[PLpos].output;
                Eigen::MatrixXd ** maxIndices = netLayers.PL[PLpos].maxIndices;
                // delta values are stored in variable: delta2
                backPropgateToConv(delta, weights, activatedOut, 
                        maxIndices, poolH, poolW, output);
                
//                if ( PLpos == 0 ) {std::exit(0); }
                PLpos--;
                break;
            }
            case 'F':
            {
//                std::cout<<"Backward Pass from Fully Connected Layer"<<FCLpos+1<<"...\n";
                if ( layerOrder[layer-1] == 'F' ) {
                    Eigen::MatrixXd deltaW;
                    if ( layer == (layers-1)) {
                        // error values in each iteration
//                        std::cout<<(delta.sum() / delta.size())<<"\n";
                        
                        ///////////////////////////////////////////////////////////
                        res = backPropgateLayer(delta, activatedOut);
                        deltaW = std::get<0>(res);
                        delta = std::get<1>(res);
                    } else {
                        output = netLayers.FCL[FCLpos].output;
                        res = backPropgateLayer(delta, weights, activatedOut, output);
                        deltaW = std::get<0>(res);
                        delta = std::get<1>(res);
                    }
                    
                    outputs = netLayers.FCL[FCLpos].outputs;
                    depth = netLayers.FCL[FCLpos].depth;
                    weights = new Eigen::MatrixXd * [outputs];
                    for(int i = 0; i < outputs; i++) {
                        weights[i] = new Eigen::MatrixXd[depth];
                        for(int j = 0; j < depth; j++) {
                            weights[i][j] = netLayers.FCL[FCLpos].weights[i][j];
                        }
                    }

                    // adjusting the weight matrix
                    int h = netLayers.FCL[FCLpos].height;
                    int w = netLayers.FCL[FCLpos].width;
                    for(int i = 0; i < outputs; i++) {
                        for(int j = 0; j < depth; j++) {
                            for(int a = 0; a < w; a++) {
                                for(int b = 0; b < h; b++) {
                                    netLayers.FCL[FCLpos].weights[i][j](b,a) -= 
                                            (deltaW(a,b) * learningRate);
                                }
                            }
                        }
                    }
                    // adjusting the bias values
                    netLayers.FCL[FCLpos].bias -= delta;
                    
                } else if ( layerOrder[layer-1] != 'F' ) {
                    
                    output = netLayers.FCL[FCLpos].output;
                    delta = backPropgateToPool(delta, weights, activatedOut, output);
                    
                    outputs = netLayers.FCL[FCLpos].outputs;
                    depth = netLayers.FCL[FCLpos].depth;
                    weights = new Eigen::MatrixXd * [outputs];
                    for(int i = 0; i < outputs; i++) {
                        weights[i] = new Eigen::MatrixXd[depth];
                        for(int j = 0; j < depth; j++) {
                            weights[i][j] = netLayers.FCL[FCLpos].weights[i][j];
                        }
                    }
                    // adjusting the weight matrix
                    for(int i = 0; i < outputs; i++) {
                        for(int j = 0; j < depth; j++) {
                            Eigen::Map<Eigen::MatrixXd> weightLayer(
                                netLayers.FCL[FCLpos].weights[i][j].data(), outHeight, outWidth);
                            weights[i][j] = weightLayer;
                            netLayers.FCL[FCLpos].weights[i][j] -= (poolDeltaW[(i*poolDepth)+j] * learningRate); 
                        }
                    }
                    // adjusting the bias values
                    netLayers.FCL[FCLpos].bias -= delta;
                }   
                FCLpos--;
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

Eigen::MatrixXd CNN::backPropgateToPool(
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
    poolDeltaW = new Eigen::MatrixXd[prevDelta.rows() * poolDepth];
    for(int i = 0; i < prevDelta.rows(); i++) {
        for(int j = 0; j < poolDepth; j++) {
            poolDeltaW[(i*poolDepth)+j] = prevActivOut[j]*prevDelta(i,0);
//            std::cout<<"index: "<<(i*poolDepth)+j<<"\n";
//            std::cout<<poolDeltaW[(i*poolDepth)+j]<<"\n\n****************\n";
        }
    }
//    std::cout<<"poolDepth: "<<poolDepth<<"\n";
//    std::cout<<"outHeight: "<<outHeight<<"\n";
//    std::cout<<"outWidth: "<<outWidth<<"\n";
    return prevDelta;
}

int CNN::backPropgateToConv(
        Eigen::MatrixXd prevDelta, 
        Eigen::MatrixXd ** prevWeight, 
        Eigen::MatrixXd * prevActivOut, 
        Eigen::MatrixXd ** maxIndices, 
        int poolH, int poolW, 
        Eigen::MatrixXd * output
) {

    // shape of the pool output (outer layer - pool): poolDepth, outHeightC, outWidthC
    // previous weight (prevWeight) shape: outputs, depth, outHeight, outWidth
    
    Eigen::MatrixXd activePoolOut[poolDepth];
    for (int i = 0; i < poolDepth; i++) {
        activePoolOut[i] = Activation::sigmoidDeriv(output[i]);  
    }
    
    int WMatSize = outHeightC*outWidthC; 
    int scale = (WMatSize/prevWeight[0][0].size());
    int r = prevWeight[0][0].rows();
    int c = prevWeight[0][0].cols();
    if (prevWeight[0][0].size() != WMatSize) {    
        Eigen::MatrixXd ** tmpPrevWeight;
        tmpPrevWeight = new Eigen::MatrixXd * [outputs];
        for (int i = 0; i < outputs; i++) {
            tmpPrevWeight[i] = new Eigen::MatrixXd[depth];
            for (int j = 0; j < depth; j++) {
                tmpPrevWeight[i][j] = Eigen::MatrixXd::Zero(outHeightC,outWidthC);
                r = prevWeight[0][0].rows();
                for(int a = 0; a < outHeightC; a+=r) {
                    c = prevWeight[0][0].cols();
                    for(int b = 0; b < outWidthC; b+=c) {
                        if ((b+c) > outWidthC) {
                            c = outWidthC - b;
                        }               
                        if ((a+r) > outHeightC) {
                            r = outHeightC - a;
                        }
                        tmpPrevWeight[i][j].block(a,b,r,c) = prevWeight[i][j].block(0,0,r,c)/scale;       
                    }    
                }
            }
        }
        prevWeight = tmpPrevWeight;
    }
    
    Eigen::MatrixXd tmpDelta = Eigen::MatrixXd::Zero(poolDepth,(outHeightC*outWidthC));
//    std::cout<<"cols: "<<outHeightC*outWidthC<<"\n";
//    std::cout<<"previous w size: "<<prevWeight[0][0].size()<<"\n";
//    std::cout<<"outputs: "<<outputs<<"\n";
//    std::cout<<"depth: "<<depth<<"\n";
    for (int i = 0; i < outputs; i++) {
        for (int j = 0; j < depth; j++) {
            Eigen::Map<Eigen::RowVectorXd> deltaRowV(prevWeight[i][j].data(), prevWeight[i][j].size());
            deltaRowV = deltaRowV*prevDelta(i,0);
            tmpDelta.row(j).array() += deltaRowV.array();
        }
    }
    
    for (int i = 0; i < depth; i++) {
        Eigen::Map<Eigen::RowVectorXd> activePoolOutRowV(activePoolOut[i].data(), activePoolOut[i].size());
        tmpDelta.row(i) = tmpDelta.row(i).array() * activePoolOutRowV.array();
    }
    
    // reshaping the pool output
    Eigen::MatrixXd reshapedPoolOut(poolDepth, (outHeightC*outWidthC));
    for (int i = 0; i < poolDepth; i++) {
        Eigen::Map<Eigen::RowVectorXd> outputRowV(output[i].data(), output[i].size());
        reshapedPoolOut.row(i) = outputRowV;
    }
    
//    Eigen::MatrixXd newDelta[convDepth];
    delta2 = new Eigen::MatrixXd[convDepth];
    for (int i = 0; i < convDepth; i++) {
        delta2[i] = Eigen::MatrixXd::Zero(outHeight, outWidth);
    }
    
    // reshaping the maxIndices
    Eigen::MatrixXd reshapedMaxInd[poolDepth];
    for (int  i = 0; i < poolDepth; i++) {
        reshapedMaxInd[i] = Eigen::MatrixXd::Zero((outHeightC*outWidthC),2);
        for (int j = 0; j < outWidthC; j++) {
            for (int a = 0; a < outHeightC; a++) {
                reshapedMaxInd[i].row((j*outHeightC)+a) = maxIndices[i][j].row(a);   
            }
        }
    }
    
    Eigen::MatrixXd poolBlock;
    int row, slide;
    for (int i = 0; i < convDepth; i++) {
        row = 0; slide = 0;
        for (int j = 0; j < outWidthC; j++) {
            
            poolBlock = prevActivOut[i].block(row,slide,poolH,poolW);
            poolBlock = Activation::maxPoolDelta(reshapedPoolOut(i,j),tmpDelta(i,j),poolBlock,poolH,poolW);
            delta2[i].block(row, slide, poolH, poolW) = poolBlock;

            slide += poolW;
            if (slide >= outWidth) {
                slide = 0;
                row += poolH;
            }
        }
    }
//    std::cout<<newDelta[2]<<"\n\n";
//    std::cout<<"pool Depth: "<<poolDepth<<"\n";
//    std::cout<<"pool out height: "<<outHeightC<<"\n";
//    std::cout<<"pool out width: "<<outWidthC<<"\n";
//    
//    std::cout<<"Outputs: "<<outputs<<"\n";
//    std::cout<<"depth: "<<depth<<"\n";
    
//    delta2 = newDelta;
    return 0;
}

Eigen::MatrixXd CNN::backPropgateToFilters(
        Eigen::MatrixXd ** prevWeight, 
        int stride, int filterSize,
        Eigen::MatrixXd * prevActivOut
) {
    
    int totalDeltas = delta2[0].rows() * delta2[0].cols();
//    int filterSize = prevWeight[0][0].rows();
    
    Eigen::MatrixXd biasDelta = Eigen::MatrixXd::Zero(poolDepth, 1);
    filterDelta = new Eigen::MatrixXd[poolDepth];
    for(int i = 0; i < poolDepth; i++) {
        filterDelta[i] = Eigen::MatrixXd::Zero(filterSize,filterSize);
    }
    
    // reshaping the delta values matrix array
    Eigen::MatrixXd reshapedDelta = Eigen::MatrixXd::Zero(poolDepth,totalDeltas);
    
    for (int i = 0; i < poolDepth; i++) {
        Eigen::Map<Eigen::RowVectorXd> deltaRowV(delta2[i].data(), delta2[i].size());
        reshapedDelta.row(i) = deltaRowV;
    } 
    
    int row, slide;
    Eigen::MatrixXd inMat;
    for (int i = 0; i < poolDepth; i++) {
        row = 0; slide = 0;
        for (int j = 0; j < totalDeltas; j++) {
            inMat = prevActivOut[i].block(row,slide,filterSize,filterSize);
            filterDelta[i] += (inMat*reshapedDelta(i,j));
                   
            slide += stride;
            if ((slide + filterSize - stride) >= prevActivOut[0].cols()) {
                slide = 0;
                row+=stride;
            }
        }
        biasDelta(i,0) += delta2[i].sum();
    }
    
    return biasDelta;
}

int CNN::train(
    Eigen::MatrixXd ** inputs, 
    Eigen::MatrixXd * labels,
    int inputSize, 
    int iterations, 
    double learningRate
) {
    bpError = Eigen::VectorXd(0);
    this->learningRate = learningRate;
    for (int i = 0; i < inputSize; i++) {
        for (int j = 0; j < iterations; j++) {
            forward(inputs[i]);
            backprop(inputs[i],labels[i]);
        }    
    }
    std::cout<<bpError<<"\n";
    return 0;
}

Eigen::MatrixXd CNN::predict(Eigen::MatrixXd * input) {
    forward(input);
//    for (int i = 0; i < classPredicts.rows(); i++) {
//        if (classPredicts(i,0) > 0.5 ) classPredicts(i,0) = 1;
//        else classPredicts(i,0) = 0;
//    }
    return classPredicts;
}

