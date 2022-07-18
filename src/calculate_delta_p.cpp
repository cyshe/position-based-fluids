#include "calculate_delta_p.h"
#include <Eigen/Core>
#include <iostream>

#define _USE_MATH_DEFINES
#include <math.h>

void calculate_delta_p(
    Eigen::MatrixXd & delta_p,
    const Eigen::MatrixXd & X,
    const Eigen::MatrixXd & N,
    const Eigen::VectorXd & lambda,
    const double rho_0,
    const int numofparticles,
    const double h,
    const double k
    ){
    
    
    for (int i = 0; i < numofparticles; i ++){
        Eigen::Vector3d sum;
        sum.setZero();

        for (int j = 0; j < numofparticles; j++){
            if ((X.row(i) -X.row(j)).norm() <= h && i != j && N(i,j) == 1){    //N(i,j) == 1
                Eigen::Vector3d r;
                double s_corr;
                r  = X.row(i) - X.row(j);
                //std::cout << "r " << r << std::endl;
                
                if (r.norm() < h){
                    s_corr = -k * pow((pow(h,2) - pow(r.norm(), 2))/(pow(h,2) - pow(0.2*h,2)), 12); 
                }
                else{
                    s_corr = 0.0;
                }

                sum += (lambda(i) + lambda(j) + s_corr) * 45 * pow(h - r.norm(), 2)/ (M_PI * pow(h,6)) * r/r.norm();
                
            }
        }
        delta_p.row(i) = sum / rho_0;
        //std::cout << sum  << std::endl;
    }

}