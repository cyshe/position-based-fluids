#include "forces.h"
#include "calculate_lambda.h"
#include "calculate_delta_p.h"
#include <Eigen/Core>
#include <iostream>


void forces(
    Eigen::MatrixXd & X,
    Eigen::MatrixXd & N,
    const int numofparticles
    ){
    
    double h = 0.2;
    double rho_0 = 1000;
    


    Eigen::VectorXd lambda;
    Eigen::MatrixXd delta_p;

    delta_p.resize(numofparticles, 3);


    lambda.resize(numofparticles);
    lambda.setZero();
    calculate_lambda(X, N, lambda, h, rho_0);
    std::cout << "lambda "<< lambda(158) << std::endl;
    calculate_delta_p(delta_p, X, N, lambda, rho_0, numofparticles, h, 0.001);
    
    std::cout << "delta_p "<< delta_p(158)<< std::endl;

    X = X + delta_p;

    // std::cout << "+delta p" << X.row(0) << std::endl;
    

}