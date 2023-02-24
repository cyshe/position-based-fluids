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
    VectorXd & J,
    MatrixXi & N,
    const Eigen::Matrix<double, 2, 1> & low_bound,
    const Eigen::Matrix<double, 2, 1> & up_bound,
    const int numofparticles,
    const int iters, 
    const double dt
    ){
    int n = numofparticles;
    double kappa = 10000;//100000;
    double rho_0 = 1; //define later
    double m = 1;
    double h = 0.1; // h for particle distance
    double vol = m/rho_0;
    
    
    MatrixXd f_ext(n, 2);
    f_ext.setZero();
    f_ext.col(1).setConstant(-9.8);
    
    VectorXd x_hat;
    x_hat.resize(2 * n);
    x_hat.setZero();
    

    double e0;

    //variables for matrix solve
    MatrixXd A, M, B, H, V_b, V_b_inv;
    A.resize(2 * n, 2 * n);
    M.resize(2 * n, 2 * n);
    B.resize(n, 2 * n);
    H.resize(n, n);
    V_b.resize(n, n);
    V_b_inv.resize(n, n);

    VectorXd b, Jx;
    b.resize(2 * n);
    Jx.resize(n); 
    
    A.setZero();
    M.setZero();
    H.setZero();
    V_b.setZero();
    V_b_inv.setZero();
    b.setZero();
    
    MatrixXd X_curr;
    X_curr.resize(n, 2);
    X_curr.setZero();
    
    VectorXd J_curr;
    J_curr.resize(n, 1);
    J_curr.setZero();

    VectorXd X_flat, V_flat, f_ext_flat;
    X_flat.resize(2 * n); 
    V_flat.resize(2 * n); 
    f_ext_flat.resize(2 * n); 
    X_flat.setZero();
    V_flat.setZero();
    f_ext_flat.setZero();

    // Flatten position matrices and copy matrix into curr
    for (int i = 0; i < n; i++) {
        X_flat(2 * i) = X(i, 0);
        X_flat(2 * i + 1) = X(i, 1);
        V_flat(2 * i) = V(i, 0);
        V_flat(2 * i + 1) = V(i, 1);
        f_ext_flat(2 * i) = f_ext(i, 0);
        f_ext_flat(2 * i + 1) = f_ext(i, 1);

        X_curr(i, 0) = X(i, 0);
        X_curr(i, 1) = X(i, 1);
        J_curr(i) = J(i);
    }

    // x hat
    x_hat = X_flat + dt * V_flat + dt * dt * f_ext_flat;
    
    //M diagonal of particle masses
    M = MatrixXd::Identity(2 * n, 2 * n) * m;

   
    //V_block diagonal of particle volumes
    V_b = MatrixXd::Identity(n, n) * vol;
    V_b_inv = MatrixXd::Identity(n, n) / vol;

    //H
    H = kappa * dt * dt * V_b;

    
    for (int it = 0; it < iters; it++) {
        // Flatten current position matrix
        for (int i = 0; i < n; i++) {
            X_flat(2 * i) = X_curr(i, 0);
            X_flat(2 * i + 1) = X_curr(i, 1);
        }
        
        //e0 = ((X_flat - x_hat).transpose() * M * (X_flat - x_hat) + 0.5 * kappa * dt * dt * (J_curr - MatrixXd::Constant(n, 1, 1)).transpose() * (J_curr - MatrixXd::Constant(n, 1, 1)))(0, 0);

        //B
        B.setZero();
        double fac = 10/7/M_PI/h/h/rho_0;
        for (int i = 0; i < n; i ++){
            for (int j = 0; j < n; j++){
                double q = (X_curr.row(j) - X_curr.row(i)).norm();
                MatrixXd dc_dx;
                dc_dx.resize(1, 2);
                dc_dx.setZero();
            
                if (q <= 1 && q > 0){
                    dc_dx = fac * m * (3*q - 9 * q * q/4) * (X_curr.row(j) - X_curr.row(i))/q;
                }
                else if (q > 1 && q <= 2){
                    dc_dx = fac * m * 0.75 * (2 - q) * (2 - q)  *q* (X_curr.row(j) - X_curr.row(i))/q;
                }
                B(i, 2*j) = dc_dx(0);
                B(i, 2*j+1) = dc_dx(1);

            }
        }
    

        // Jx rho(x)/rho_0
        Jx.setZero();
        for (int i = 0; i < n; i++){
            for (int j = 0; j < n; j++){
                double r = (X_curr.row(j) - X_curr.row(i)).norm();
                double sig = 10/7/M_PI/h/h/rho_0;

                if (r <= 1 && r > 0){
                    Jx(i) += m * (1 - 1.5 * r * r *(1 - 0.5 *r)) * sig;
                }
                else if (r > 1 && r <= 2){
                    Jx(i) += m * (2-r)*(2-r)*(2-r) * sig /4;
                }
            }
        }

        std::cout << "block" << std::endl;
        A = M + B.transpose() * V_b_inv * H * V_b_inv *B;
        b = -M * (X_flat - x_hat) + B.transpose()* V_b_inv* kappa * dt * dt * (J_curr - MatrixXd::Constant(n, 1, 1)) - V_b_inv*H*V_b_inv*(J_curr-Jx);
        //A.block(0, 0, 2 * n, 2 * n) = M;
        //A.block(0, 3 * n, 2 * n, n) = B.transpose();
        //A.block(2 * n, 2 * n, n, n) = H;
        //A.block(n, 3*n, n, n) = V_b;
        //A.block(3*n, 0, n, 2 * n) = B;
        //A.block(3*n, n, n, n) = V_b;
        
        //b.block(0, 0, 2*n, 1) = M * (X_flat - x_hat);
        //b.block(2 * n, 0, n, 1) = kappa * dt * dt * (J - MatrixXd::Constant(n, 1, 1));
        //b.block(3 * n, 0, n, 1) = J - Jx;
        // TODO: set up Ax = b
        // 
        //solve matrix multiplication
        MatrixXd sol;
        sol.resize(2 * n, 1);
    
        std::cout << "start solve" << std::endl;
        sol = A.colPivHouseholderQr().solve(b);

        VectorXd X_new_flat, delta_X, J_new, delta_J, lambda;
        X_new_flat.resize(n*2);
        delta_X.resize(n*2);
        J_new.resize(n);
        delta_J.resize(n);
        lambda.resize(n);

        X_new_flat.setZero();
        delta_X.setZero();
        J_new.setZero();
        delta_J.setZero();
        lambda.setZero();

        std::cout << "solved" << std::endl;
        delta_X = sol;
        delta_J = V_b_inv * (-(J_curr-Jx) - (B * sol));
        lambda = -V_b_inv * kappa * dt * dt * (J_curr - MatrixXd::Constant(n, 1, 1)) \
        + V_b_inv * H * V_b_inv * (J_curr - Jx) +  V_b_inv * H * V_b_inv * B * sol;

        //do line search
        double alpha = 1.0;
        
        e0 = 0.5 * (X_flat - x_hat).transpose() * M * (X_flat - x_hat)
        + 0.5 * kappa * dt * dt * (J_curr - MatrixXd::Constant(n, 1, 1)).squaredNorm()
        + lambda.dot(J_curr - Jx);

        double e_new = e0 + 1;
        double e1, e2, e3;
        while (e_new > e0 && alpha > 1e-6){ 
            std::cout << "alpha: " << alpha << std::endl;
            X_new_flat = X_flat + delta_X * alpha;
            J_new = J_curr + delta_J * alpha;
            //calculate energy
            
            e1 = 0.5 * (X_new_flat - x_hat).transpose() * M * (X_new_flat - x_hat);
            e2 = 0.5 * kappa * dt * dt * (J_new - MatrixXd::Constant(n, 1, 1)).squaredNorm(); 
            e3 = lambda.dot(J_new - Jx);
            alpha = alpha/2;
            e_new = e1 + e2 + e3;
        }

        
        // std::cout << X_new_flat - X_flat << std::endl;
        std::cout << "alpha: " << alpha << std::endl;
        // std::cout << "e_new: " << e_new << std::endl;
        // std::cout << "e0: " << e0 << std::endl;
        //std::cout << "e1: " << e1 << std::endl;
        //std::cout << "e2: " << e2 << std::endl;
        // std::cout << "e3: " << e3 << std::endl;
        std::cout << (X_new_flat - X_flat).norm() << std::endl;

        for (int i = 0; i < n; i++) {
            X_curr(i, 0) = X_new_flat(2*i);
            X_curr(i, 1) = X_new_flat(2*i+1);
            J_curr(i) = J_new(i);
        }
    }        


    V = (X_curr-X)/dt;
    X = X_curr;
    J = J_curr;     
        
    //std::cout << X << std::endl;
    // boundary detection
    /*
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
 */   
    return;
}

template <>
void animate_implicit<3>(
    MatrixXd & X, 
    MatrixXd & V, 
    VectorXd & J,
    MatrixXi & N,
    const Eigen::Matrix<double, 3, 1> & low_bound,
    const Eigen::Matrix<double, 3, 1> & up_bound,
    const int numofparticles,
    const int iters, 
    const double dt
    ){}
