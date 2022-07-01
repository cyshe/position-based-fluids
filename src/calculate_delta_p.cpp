#include "calculate_delta_p.h"
#include <Eigen/Core>
#include <iostream>

void calculate_delta_p(
    Eigen::MatrixXd & delta_p,
    const Eigen::MatrixXd & X,
    const Eigen::MatrixXd & N,
    const Eigen::VectorXd & lambda,
    const double rho_0,
    const int numofparticles,
    const double h
    ){
    
    
    for (int i = 0; i < numofparticles; i ++){
        Eigen::Vector3d sum;
        sum.setZero();

        for (int j = 0; j < numofparticles; j++){
            if ((N(i,j) == 1) && i != j){
                Eigen::Vector3d r;
                r  = X.row(i) - X.row(j);
                //<< X(i, 0) - X(j,0), X(i, 1) - X(j,1), X(i, 2) - X(j,2);
                //std::cout << "r " << r << std::endl;

                sum += (lambda(i) + lambda(j)) * (45 * pow(h - r.norm(), 2)/3.14/pow(h,6))/rho_0 * r;
            }
        }
        delta_p.row(i) = sum;
    }

}