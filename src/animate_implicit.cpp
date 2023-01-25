#include "animate_implicit.h"
#include "calculate_lambda.h"
#include <igl/signed_distance.h>
#include <Eigen/Core>
#include <iostream>

#define _USE_MATH_DEFINES
#include <math.h>

using namespace Eigen;

template <>
void animate_implicit<2>(
    MatrixXd & X, 
    MatrixXd & V,
    MatrixXd & J,
    MatrixXi & N,
    const Eigen::Matrix<double, 2, 1> & low_bound,
    const Eigen::Matrix<double, 2, 1> & up_bound,
    const int numofparticles,
    const int iters, 
    const double dt
    ){
    int n = numofparticles;
    MatrixXd X_new;
    double kappa = 1000;//100000;
    double rho_0 = 1; //define later
    double m = 1;
    double h = 0.1;
    double vol = m/rho_0;
    X_new.resize(n, 2);

    MatrixXd f_ext(n, 2);
    f_ext.setZero();
    f_ext.col(1).setConstant(-9.8);

    
    MatrixXd A, M, B, H, V_b, b, x_hat, Jx, X_flat, f_ext_flat, V_flat;
    A.resize(4 * n, 4 * n);
    M.resize(2 * n, 2 * n);
    B.resize(n, 2 * n);
    H.resize(n, n);
    V_b.resize(n, n);
    b.resize(4 * n, n);
    x_hat.resize(2 * n, 1);
    Jx.resize(n, 1); 
    X_flat.resize(2 * n, 1); 
    V_flat.resize(2 * n, 1); 
    f_ext_flat.resize(2 * n, 1); 

    A.setZero();
    M.setZero();
    B.setZero();
    H.setZero();
    V_b.setZero();
    b.setZero();
    x_hat.setZero();
      
    // Flatten position matrices
    for (int i = 0; i < n; i++) {
        X_flat(2 * i) = X(i, 0);
        X_flat(2 * i + 1) = X(i, 1);
        V_flat(2 * i) = V(i, 0);
        V_flat(2 * i + 1) = V(i, 1);
        f_ext_flat(2 * i) = f_ext(i, 0);
        f_ext_flat(2 * i + 1) = f_ext(i, 1);
        
    }
    
    //M diagonal of particle masses
    M = MatrixXd::Identity(2 * n, 2 * n) * m;

    //B
    double fac = 10/7/M_PI/h/rho_0;

    for (int i = 0; i < n; i ++){
        for (int j = 0; j < n; j++){
            double q = (X.row(j) - X.row(i)).norm();
            MatrixXd dc_dx;
            dc_dx.resize(1, 2);
            dc_dx.setZero();
            
            if (q <= 1 && q > 0){
                dc_dx = - fac * m * (3 - 9 * q/4) * X.row(j);
            }
            else if (q > 1 && q <= 2){
                dc_dx = fac * m * 0.75 * (2 - q *q) * X.row(j);
            }
            B(i, 2*j) = dc_dx(0);
            B(i, 2*j+1) = dc_dx(1);

        }
    }

    //V_block diagonal of particle volumes
    V_b = MatrixXd::Identity(n, n) * vol;
    
    //H
    H = kappa * h * h * V_b;
    
    
    std::cout << "inverse" << std::endl;
    // x hat
    x_hat = X_flat + h * V_flat + h * h * MatrixXd::Identity(2*n, 2*n) / m  * f_ext_flat;

    // Jx rho(x)/rho_0
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            double r = (X.row(j) - X.row(i)).norm();
            double sig = 10/7/M_PI/h;

            if (r <= 1 && r > 0){
                Jx(i) += m * (1 - 1.5 * r * r *(1 - 0.5 *r)) * sig;
            }
            else if (r > 1 && r <= 2){
                Jx(i) += m * (2-r)*(2-r)*(2-r) * sig /4;
            }
        }
    }

    std::cout << "block" << std::endl;
    A.block(0, 0, 2 * n, 2 * n) = M;
    A.block(0, 3 * n, 2 * n, n) = B.transpose();
    A.block(2 * n, 2 * n, n, n) = H;
    A.block(n, 3*n, n, n) = V_b;
    A.block(3*n, 0, n, 2 * n) = B;
    A.block(3*n, n, n, n) = V_b;
    
    b.block(0, 0, 2*n, 1) = M * (X_flat - x_hat);
    b.block(2 * n, 0, n, 1) = kappa * h * h * (J - MatrixXd::Constant(n, 1, 1));
    b.block(3 * n, 0, n, 1) = J - Jx;
    
    // TODO: set up Ax = b
    // 
    //solve matrix multiplication
    MatrixXd sol;
    sol.resize(4 * n, 1);
    
    std::cout << "start solve" << std::endl;
    sol = A.colPivHouseholderQr().solve(b);

    std::cout << "solve" << std::endl;
    for (int i=0 ; i < n; i++){
        X_new(i, 0) = sol(2 *i, 0);
        X_new(i, 1) = sol(2 *i + 1, 0); 
    }

    J = sol.block(2*n, 0, n, 1);
    V = (X_new - X)/h;
    X = X_new;
    std::cout << J << std::endl;
    // boundary detection
    
    for (int i = 0; i < numofparticles; i++){
        for (int j = 0; j < 2; ++j) {
            if (X(i,j) < low_bound(j)) {
                V(i, j) = abs(V(i, j));
                X(i,j) = low_bound(j);
            }
            if (X(i,j) > up_bound(j)) {
                V(i, j) = -1* abs(V(i, j));
                X(i,j) = up_bound(j);
            }
        }
    }
    
    return;
}

template <>
void animate_implicit<3>(
    MatrixXd & X, 
    MatrixXd & V, 
    MatrixXd & J,
    MatrixXi & N,
    const Eigen::Matrix<double, 3, 1> & low_bound,
    const Eigen::Matrix<double, 3, 1> & up_bound,
    const int numofparticles,
    const int iters, 
    const double dt
    ){}
