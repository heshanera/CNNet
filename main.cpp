/* 
 * File:   main.cpp
 * Author: heshan
 *
 * Created on June 6, 2018, 12:03 AM
 */

#include <iostream>
#include "ConvolutionLayer.hpp"

using namespace std;

/*
 * 
 */
int main(int argc, char** argv) {

    
    int depth = 1;
    int height = 20;
    int width = 20; 
    int filterSize = 5;
    int stride = 1; //
    int noOfFilters = 10;
    int padding = 0; //
    
    ConvolutionLayer cl( depth, height, width, filterSize, stride, noOfFilters, padding);
    
    Eigen::MatrixXd input[1];
    Eigen::MatrixXd img = Eigen::MatrixXd::Ones(20,20);
    input[0] = img;
    
    cl.convolute(input);
    
    return 0;
}

