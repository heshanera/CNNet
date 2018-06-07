/* 
 * File:   Activation.cpp
 * Author: heshan
 * 
 * Created on June 7, 2018, 11:17 AM
 */

#include "Activation.hpp"

Activation::Activation() { }

Activation::Activation(const Activation& orig) { }

Activation::~Activation() { }

double Activation::sigmoid(double x) {
    return 1.0/(1.0 + std::exp(-x));
}
    
double Activation::sigmoidDeriv(double x) {
    return sigmoid(x) * (1-sigmoid(x));
}
    

