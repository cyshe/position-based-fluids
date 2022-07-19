#include "calculate_lambda.h"
#include <iostream>
#include <Eigen/Core>

#define _USE_MATH_DEFINES
#include <math.h>

void calculate_lambda(
    const Eigen::MatrixXd & x,
    const Eigen::MatrixXd & N,
    Eigen::VectorXd & lambda,
    const double h,
    const double rho_0
    ){
    //C_i = rho_i/rho_0 -1
    //rho_i = Sigma m_j * W(p_i-p_j,h)
    double c;
    double grad_c;
    for (int i = 0; i < x.rows(); i++){
        c = C(x, N, rho_0, i, h);
        grad_c = grad_C_squared(x, N, rho_0, i, h);
        if (grad_c + 1000 == 0){
            lambda(i) = -c/(grad_c + 999);
        }
        else {
            lambda(i) = -c/(grad_c + 1000);
        }
            
        //std::cout << "c " << c  << " grad_c "<< grad_c << std::endl;
    }
    
}

double W(const Eigen::Vector3d r, const double h){ //poly 6 kernel
    double l = r.norm();
    if (l <= h){
        return 315 * pow(pow(h, 2) - pow(l,2), 3)/ (64 * M_PI * pow(h,9));
    }
    else{
        return 0.0;
    }
}

double C(const Eigen::MatrixXd x, 
    const Eigen::MatrixXd N,
    const double rho_0,
    const int i,
    const double h){

    double rho_i = 0.0;

    for (int j = 0; j < x.rows(); j++){
        if ((x.row(i) -x.row(j)).norm() <= h && i != j){ // 
            //std::cout << "$" << std::endl;
            rho_i += 0.037* W((x.row(i) - x.row(j)), h);  //0.037 * is mass of each particle
        }
    }
    return rho_i/rho_0 - 1.0;
}

double grad_C_squared(const Eigen::MatrixXd x, 
    const Eigen::MatrixXd N,
    const double rho_0,
    const int i,
    const double h){
    double sum_k = 0;
    for (int k = 0; k < x.rows(); k++){
        Eigen::Vector3d grad_c;
        grad_c.setZero();
        if (k == i){
            for (int j = 0; j < x.rows(); j++){
                if ((x.row(i) -x.row(j)).norm() <= h && i != j){ //  && N(i,j) == 1 
                    grad_c += 45 * pow(h - ((x.row(i) -x.row(j)).norm()), 2) / (M_PI * pow(h, 6)) * (x.row(i) -x.row(j)) /(x.row(i) -x.row(j)).norm();
                }
            }
            grad_c = grad_c/rho_0;
            sum_k += pow(grad_c.norm(), 2);
        }
        else{
            if ((x.row(i) -x.row(k)).norm() <= h && i != k){ // && N(i,k) == 1
                grad_c += 45 * pow(h - (x.row(i) - x.row(k)).norm(), 2) / (M_PI * pow(h, 6)) * (x.row(i) -x.row(k))/(x.row(i) -x.row(k)).norm();
            }
            grad_c = grad_c/rho_0;
            sum_k += pow(grad_c.norm(), 2);
        }
    }
    return sum_k;
}