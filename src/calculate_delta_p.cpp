#include "calculate_delta_p.h"
#include <Eigen/Core>
#include <iostream>

#define _USE_MATH_DEFINES
#include <math.h>


template<int DIM>
void calculate_delta_p(
    Eigen::MatrixXd & delta_p,
    const Eigen::MatrixXd & X,
    const Eigen::MatrixXi & N,
    const Eigen::VectorXd & lambda,
    const double rho_0,
    const int numofparticles,
    const double h,
    const double k
    ){
    
    const double W_fac = 315.0 / 64.0 / M_PI / std::pow(h,9);
    const double gW_fac = -45.0 / M_PI / std::pow(h,6);

    for (int i = 0; i < numofparticles; i ++){
        Eigen::Matrix<double, 1, DIM> sum;
        sum.setZero();

        for (int it = 0; it < N.cols(); it++){
            int j = N(i, it);

            double r = (X.row(i) -X.row(j)).norm();
            if (r < h && r > 0) {    //N(i,j) == 1
                Eigen::Matrix<double, 1, DIM> diff = X.row(i) - X.row(j);
                double s_corr = 0.0;
                
                if (k > 0.0) {
                    double dq = 0.2 * h;
                    double W = W_fac * std::pow(h*h - r*r, 3);
                    double W0 = W_fac * std::pow(h*h - dq*dq, 3);
                    s_corr = -k * std::pow(W/W0, 4);
                }

                sum += gW_fac * (lambda(i) + lambda(j)+ s_corr) * pow(h - r, 2) * diff / r;                
            }
        }
        delta_p.row(i) = sum / rho_0;
        //std::cout << sum  << std::endl;
    }

}