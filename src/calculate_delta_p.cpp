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

        for (int j = 0; j < numofparticles; j++){
            if (N(i,j) == 1){
                double r = X(i).norm()
                (lambda(i) + lambda(j)) * (45 * (h - r  ) 
            }
        }
    }
}