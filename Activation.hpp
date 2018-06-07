/* 
 * File:   Activation.hpp
 * Author: heshan
 *
 * Created on June 7, 2018, 11:17 AM
 */

#ifndef ACTIVATION_HPP
#define ACTIVATION_HPP

#include <cmath>

class Activation {
public:
    Activation();
    Activation(const Activation& orig);
    virtual ~Activation();
    
    static double sigmoid(double);
    static double sigmoidDeriv(double);
    
private:

};

#endif /* ACTIVATION_HPP */

