/* 
 * File:   PoolLayer.cpp
 * Author: heshan
 * 
 * Created on June 7, 2018, 2:05 PM
 */

#include "PoolLayer.hpp"

PoolLayer::PoolLayer() { }

PoolLayer::PoolLayer(std::tuple<int, int, int> dimensions, int poolW, int poolH) {

    this->depth = std::get<0>(dimensions);
    this->height = std::get<1>(dimensions);
    this->width = std::get<2>(dimensions);
    this->poolW = poolW;
    this->poolH = poolH;
    this->outHeight = ((height - poolH)/poolH) + 1;
    this->outWidth = ((width - poolW)/poolW) + 1;        
    initMat();
}

PoolLayer::PoolLayer(const PoolLayer& orig) { }

PoolLayer::~PoolLayer() { }

int PoolLayer::initMat() {
    
    output = new Eigen::MatrixXd[depth];
    for (int i = 0; i < depth; i++) {
        Eigen::MatrixXd outMat = Eigen::MatrixXd::Zero(outHeight,outWidth);
        output[i] = outMat;
    }
    
    maxIndices = new Eigen::MatrixXd * [depth];
    for (int i = 0; i < depth; i++) {
        maxIndices[i] = new Eigen::MatrixXd[outWidth];
        for (int j = 0; j < outWidth; j++) {
            maxIndices[i][j] = Eigen::MatrixXd::Zero(outHeight,2);
        }
    }
    return 0;
}

Eigen::MatrixXd * PoolLayer::pool(Eigen::MatrixXd * input) {

    int outX, outY;
    std::ptrdiff_t a, b;
    for (int i = 0; i < depth; i++) {
        outX = 0;
        for (int x = 0; x < width; x+=poolW) {
            outY = 0;
            for (int y = 0; y < height; y+=poolH) {
                if ((outX < outWidth) & (outY < outHeight)) {
                    Eigen::MatrixXd poolBlock;
                    if (((x + poolW) > width) && ((y + poolH) > height))
                        poolBlock = input[i /*depth*/].block(y,x,height-y,width-x);
                    else if ((x + poolW) > width)
                        poolBlock = input[i /*depth*/].block(y,x,poolH,width-x);
                    else if ((y + poolH) > height)
                        poolBlock = input[i /*depth*/].block(y,x,height-y,poolW);
                    else 
                        poolBlock = input[i /*depth*/].block(y,x,poolH,poolW);
                    output[i](outY,outX) = poolBlock.maxCoeff(&a,&b);
                    maxIndices[i][outX](outY,0) = a+y; 
                    maxIndices[i][outX](outY,1) = b+x;       
                    outY++;
                }    
            }
            outX++;
        }
    }

//    for (int i = 0; i < depth; i++) {
//        std::cout<<"\n------- Depth: "<<i<<" -------\n\n";
//        std::cout<<output[i]<<"\n--------------\n\n";
//    }
//    
//    std::cout<<"\n-------*******************************************-------\n\n";
//    
//    for (int i = 0; i < depth; i++) {
//        std::cout<<"\n------- Depth: "<<i<<" -------\n\n";
//        for (int j = 0; j < outHeight; j++) {
//            std::cout<<maxIndices[i][j]<<"\n--------------\n\n";
//        }
//    }

    return output;
}

std::tuple<int, int, int> PoolLayer::getOutputDims() {
    return std::make_tuple(depth, outHeight, outWidth);
}
