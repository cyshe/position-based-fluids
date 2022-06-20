#include "forces.h"
#include "calculate_lambda.h"
#include "calculate_delta_p.h"
#include <Eigen/Core>


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
    calculate_delta_p(delta_p, X, N, lambda, rho_0, numofparticles, h);
    X = X + delta_p;


}