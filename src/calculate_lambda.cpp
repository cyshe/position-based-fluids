#include "calculate_lambda.h"
#include <iostream>
#include <Eigen/Core>

#define _USE_MATH_DEFINES
#include <math.h>

void calculate_lambda(
    const Eigen::MatrixXd & x,
    const Eigen::MatrixXi & N,
    Eigen::VectorXd & lambda,
    const double h,
    const double rho_0
    ){
    //C_i = rho_i/rho_0 -1
    //rho_i = Sigma m_j * W(p_i-p_j,h)
    double c;
    double grad_c;
    double epsilon = 100;
    for (int i = 0; i < x.rows(); i++){
        c = C(x, N, rho_0, i, h);
        grad_c = grad_C_squared(x, N, rho_0, i, h);
        lambda(i) = -c / (grad_c + epsilon);
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
    const Eigen::MatrixXi N,
    const double rho_0,
    const int i,
    const double h){

    double rho_i = 0.0;
    double mass = 1.0; //0.037 * is mass of each particle
/*
    for (int j = 0; j < x.rows(); j++){
        if ((x.row(i) -x.row(j)).norm() <= h && i != j){ // 
            //std::cout << "$" << std::endl;
            rho_i += 0.037* W((x.row(i) - x.row(j)), h);  //0.037 * is mass of each particle
        }
    }*/
    for (int it = 0; it < N.cols(); it++){
        int j = N(i, it); 
        if ((x.row(i) -x.row(j)).norm()<= h && (x.row(i) -x.row(j)).norm() > 0){
            rho_i += mass* W((x.row(i) - x.row(j)), h);  
        }
    }
    return rho_i/rho_0 - 1.0;
}

double grad_C_squared(const Eigen::MatrixXd x, 
    const Eigen::MatrixXi N,
    const double rho_0,
    const int i,
    const double h){
    double sum_k = 0;

    double fac = -45.0 / M_PI / std::pow(h,6);
    Eigen::RowVector3d grad_i(0.0,0.0,0.0);

    for (int ij = 0; ij < N.cols(); ij++) {
        int j = N(i, ij);

        double r = (x.row(i) -x.row(j)).norm();

        if (r < h && r > 0) {
            Eigen::RowVector3d grad_j = (fac * std::pow(h-r,2)
                * (x.row(i) - x.row(j)) / r) / rho_0;

            grad_i += grad_j;
            sum_k += grad_j.squaredNorm();
        }
    }
    sum_k += grad_i.squaredNorm();
    return sum_k;
}