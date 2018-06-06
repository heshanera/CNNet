/* 
 * File:   ConvolutionLayer.cpp
 * Author: heshan
 * 
 * Created on June 6, 2018, 6:26 PM
 */

#include "ConvolutionLayer.hpp"

ConvolutionLayer::ConvolutionLayer(int depth, int height, int width, int filterSize, int stride, int noOfFilters, int padding = 0) { 
    
    this->depth = depth;
    this->height = height;
    this->width = width;
    this->filterSize = filterSize;
    this->stride = stride;
    this->noOfFilters = noOfFilters;
    this->padding = padding;
    
    // generating filter weight matrix array
    // depth * filterSize * filterSize * noOfFilters
    
    filters = new Eigen::MatrixXd * [noOfFilters];
    for(int i = 0; i < noOfFilters; i++) {
        filters[i] = new Eigen::MatrixXd[depth];
    }
    
    double max, min, diff;
    min = -0.01;
    max = 0.01;
    diff = max - min;
    
    //srand(time(NULL));  // if not random numbers are generated in the same order
    for (int i = 0; i < noOfFilters; i++) {
        for (int j = 0; j < depth; j++) {
            Eigen::MatrixXd filter(filterSize,filterSize);
            for (int x = 0; x < filterSize; x++) {
                for (int y = 0; y < filterSize; y++) {
                    filter(x,y) = min + ((double)rand() / RAND_MAX) * diff;
                }
            }
            filters[i][j] = filter;
        }    
    }
    
    // generating bias values
    Eigen::VectorXd bias(noOfFilters);
    for (int i = 0; i < noOfFilters; i++) {
        bias(i) = min + ((double)rand() / RAND_MAX) * diff;
    }
    
    this->outHeight = (int)((height - filterSize + 2*padding)/stride) + 1; // rows
    this->outWidth = (int)((width - filterSize + 2*padding)/stride) + 1; // columns
   
    zValues = new Eigen::MatrixXd[noOfFilters];
    output = new Eigen::MatrixXd[noOfFilters];
    for (int i = 0; i < noOfFilters; i++) {
        Eigen::MatrixXd zVal(outHeight,outWidth);
        Eigen::MatrixXd outVal(outHeight,outWidth);
        for (int x = 0; x < outWidth; x++) {
            for (int y = 0; y < outHeight; y++) {
                zVal(x,y) = 0;
                outVal(x,y) = 0;
            }
        }
        zValues[i] = zVal;
        output[i] = outVal;
    }
    
    
//    for (int i = 0; i < noOfFilters; i++) {
//        for (int j = 0; j < depth; j++) {
//            std::cout << "Here is the matrix m:\n" << filters[i][j] << std::endl;
//        }    
//        std::cout<<"----------------------------\n\n";
//                
//    }
//    std::cout << "Here is the vector v:\n" << bias << std::endl;
//    for (int i = 0; i < noOfFilters; i++) {
//        std::cout << "Here is the matrix m:\n" << zValues[i] << std::endl;
//    }
//    for (int i = 0; i < noOfFilters; i++) {
//        std::cout << "Here is the matrix m:\n" << output[i] << std::endl;
//    }

}

ConvolutionLayer::ConvolutionLayer(const ConvolutionLayer& orig) { }

ConvolutionLayer::~ConvolutionLayer() { }



/*
 
 class ConvLayer(object):

    def __init__(self, input_shape, filter_size, stride, num_filters, padding = 0):
        self.depth, self.height_in, self.width_in = input_shape
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.num_filters = num_filters

        self.weights = np.random.randn(self.num_filters, self.depth, self.filter_size, self.filter_size)
        self.biases = np.random.rand(self.num_filters,1)

        self.output_dim1 = (self.height_in - self.filter_size + 2*self.padding)//self.stride + 1        # num of rows
        self.output_dim2 =  (self.width_in - self.filter_size + 2*self.padding)//self.stride + 1         # num of cols
        
        
        self.z_values = np.zeros((self.num_filters, self.output_dim1, self.output_dim2))
        self.output = np.zeros((self.num_filters, self.output_dim1, self.output_dim2))


    def convolve(self, input_neurons):
        '''
        Pass in the actual input data and do the convolution.
        Returns: sigmoid activation matrix after convolution 
        '''

        # roll out activations
        self.z_values = self.z_values.reshape((self.num_filters, self.output_dim1 * self.output_dim2))
        self.output = self.output.reshape((self.num_filters, self.output_dim1 * self.output_dim2))
        
        act_length1d =  self.output.shape[1]

        for j in range(self.num_filters):
            slide = 0
            row = 0

            for i in range(act_length1d):  # loop til the output array is filled up -> one dimensional (600)

                # ACTIVATIONS -> loop through each conv block horizontally
                self.z_values[j][i] = np.sum(input_neurons[:,row:self.filter_size+row, slide:self.filter_size + slide] * self.weights[j]) + self.biases[j]
                self.output[j][i] = sigmoid(self.z_values[j][i])
                slide += self.stride

                if (self.filter_size + slide)-self.stride >= self.width_in:    # wrap indices at the end of each row
                    slide = 0
                    row += self.stride

        self.z_values = self.output.reshape((self.num_filters, self.output_dim1, self.output_dim2))
        self.output = self.output.reshape((self.num_filters, self.output_dim1, self.output_dim2))

 
 
 
 */
