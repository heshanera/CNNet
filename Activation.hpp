/* 
 * File:   Activation.hpp
 * Author: heshan
 *
 * Created on June 7, 2018, 11:17 AM
 */

#ifndef ACTIVATION_HPP
#define ACTIVATION_HPP

#include <cmath>
#include <Eigen>

class Activation {
public:
    Activation();
    Activation(const Activation& orig);
    virtual ~Activation();
    
    static double sigmoid(double);
    static Eigen::MatrixXd sigmoid(Eigen::MatrixXd);
    static double sigmoidDeriv(double);
    
private:

};

#endif /* ACTIVATION_HPP */

