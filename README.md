## CNNet

[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://github.com/heshanera/LSTMnet/blob/master/LICENSE)&nbsp;&nbsp;
[![language](https://img.shields.io/badge/language-c%2B%2B-red.svg)](https://github.com/heshanera/IProc) &nbsp;&nbsp;

## Convolutional Neural Network

CNN is implemented in C++. Eigen library is used for matrix manipulations. Convolution layers, Activation layers, Pooling layers and Fully connected layers are available in the network. The number of Convolution layers, and Pooling layers and number of layers in the fully connected MLP can be adjusted accordingly. sigmoid function is used as the activation. and non overlapping max pooling is used in the pooling layers. Network is trained using gradient decent method. weights and the bias values for the filter matrices and the fully connected layers are initialized randomly.

## Network Architecture

![structure](https://github.com/heshanera/CNNet/blob/master/imgs/networkStructure/CNN_architecture.png)

## Creating A Network


###### initializing the layers

```
int width = 20;
int height = 20;

std::tuple<int, int, int> dimensions = std::make_tuple(1,height,width);

struct::ConvLayStruct CL1;
CL1.filterSize = 5; // filter size: N x N
CL1.filters = 3; // No of filters
CL1.stride = 1;
struct::ConvLayStruct CL2;
CL2.filterSize = 4; // filter size: N x N
CL2.filters = 2; // No of filters
CL2.stride = 1;

struct::PoolLayStruct PL1; // pool size: N x M
PL1.poolH = 2; // N
PL1.poolW = 2; // M
struct::PoolLayStruct PL2; // pool size: N x M
PL2.poolH = 2; // N
PL2.poolW = 2; // M

struct::FCLayStruct FCL1;
FCL1.outputs = 30; // neurons in the layer
struct::FCLayStruct FCL2;
FCL2.outputs = 15; // neurons in the layer
struct::FCLayStruct FCL3;
FCL3.outputs = 5; // neurons in the layer

char layerOrder[] = {'C','P','C','P','F','F','F'};
struct::ConvLayStruct CLs[] = {CL1,CL2};
struct::PoolLayStruct PLs[] = {PL1,PL2};
struct::FCLayStruct FCLs[] = {FCL1,FCL2,FCL3};

struct::NetStruct netStruct;
netStruct.layers = 7;
netStruct.layerOrder = layerOrder;
netStruct.CL = CLs;
netStruct.PL = PLs;
netStruct.FCL = FCLs;
```

###### Initializing the network
```
CNN cnn(dimensions, netStruct);
```

###### Training
```
Eigen::MatrixXd ** inImgArr;
Eigen::MatrixXd * inLblArr;
int iterations = 20;
int inputSize = 1;
double learningRate = 0.00001;
cnn.train(inImgArr, inLblArr, inputSize, iterations, learningRate);
```

###### Testing
```
double result;
Eigen::MatrixXd input[1];
result = cnn.predict(input);
```

## Predictions ( Normalized values )

![prediction](https://github.com/heshanera/CNNet/blob/master/imgs/timeSeries/Sea%20Level%20Pressure.png)
*The sea level pressure dataset for Darwin from the Climate Prediction Center*<br><br><br>

![prediction](https://github.com/heshanera/CNNet/blob/master/imgs/timeSeries/Daily%20Minimum%20Temperature.png)
*Daily minimum temperatures in Melbourne, Australia, 1981-1990*<br><br><br>
