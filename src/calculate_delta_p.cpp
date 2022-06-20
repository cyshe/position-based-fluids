#include "calculate_delta_p.h"
#include <Eigen/Core>

void calculate_delta_p(
    Eigen::MatrixXd & delta_p,
    const Eigen::MatrixXd & X,
    const Eigen::MatrixXd & N,
    const Eigen::VectorXd & lambda,
    const double rho_0,
    const int numofparticles,
    const double h
    ){
    
    
    for (int i =0; i < numofparticles; i ++){
        Eigen::Vector3d sum;
        sum.setZero();

        for (int j = 0; j < numofparticles; j++){
            if (N(i,j) == 1){
                Eigen::Vector3d r;
                r  << X(i, 0) - X(j,0), X(i, 1) - X(j,1), X(i, 2) - X(j,2);

                sum += (lambda(i) + lambda(j)) * (45 * pow(h - r.norm(), 2)/3.14/pow(h,6))/rho_0 * r;
            }
        }
        delta_p(i,0) = sum(0);
        delta_p(i,1) = sum(1);
        delta_p(i,2) = sum(2);
    }
}