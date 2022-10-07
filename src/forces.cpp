#include "forces.h"
#include "calculate_lambda.h"
#include "calculate_delta_p.h"
#include <Eigen/Core>
#include <iostream>

template<int DIM>
void forces(
    Eigen::MatrixXd & X,
    Eigen::MatrixXi & N,
    const int numofparticles
    ){
    
    double h = 0.2;
    double rho_0 = 1000;
    


    Eigen::VectorXd lambda(numofparticles);
    Eigen::MatrixXd delta_p(numofparticles, DIM);
    lambda.setZero();
    calculate_lambda<DIM>(X, N, lambda, h, rho_0);
    std::cout << "lambda "<< lambda(158) << std::endl;
    calculate_delta_p<DIM>(delta_p, X, N, lambda, rho_0, numofparticles, h, 0.001); //k correlates to splashiness
    std::cout << "delta_p "<< delta_p(158)<< std::endl;

    X = X + delta_p;

    // std::cout << "+delta p" << X.row(0) << std::endl; 

}

template void forces<3>(
    Eigen::MatrixXd & X,
    Eigen::MatrixXi & N,
    const int numofparticles
    );
/*
template void forces<2>(
    Eigen::MatrixXd & X,
    Eigen::MatrixXi & N,
    const int numofparticles
    );
*/