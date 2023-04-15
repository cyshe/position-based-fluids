#include "animate_implicit.h"
#include "calculate_lambda.h"
#include <igl/signed_distance.h>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <finitediff.hpp>
#include <cassert>
#include <iostream>
//#include "CompactNSearch"


#define _USE_MATH_DEFINES
#include <math.h>

using namespace Eigen;

namespace {
}

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
    double kappa = 100;//100000;
    double rho_0 = 100; //define later
    double m = 1;
    double h = 1; // h for particle distance
    double vol = 1;//m/rho_0;
    double n_corr = 4; //n for s_corr in tensile instability term

    const double kappa_dt_sqr = kappa * dt * dt;
    const double dt_sqr = dt * dt;
    MatrixXd f_ext(n, 2);
    f_ext.setZero();
    f_ext.col(1).setConstant(-9.8);
    
    VectorXd x_hat = VectorXd::Zero(2 * n);
    
    double e0;

    // Sparse matrices
    SparseMatrix<double> A, M, B, H, V_b, V_b_inv, H_inv;
    A.resize(2 * n, 2 * n);
    M.resize(2 * n, 2 * n);
    B.resize(n, 2 * n);
    H.resize(n, n);
    H_inv.resize(n, n);
    V_b.resize(n, n);
    V_b_inv.resize(n, n);

    MatrixXd d2sc_dx2;
    d2sc_dx2.resize(2 * n, 2 * n);

    // Vectors
    VectorXd b = VectorXd::Zero(2 * n);
    VectorXd Jx = VectorXd::Zero(n);
    MatrixXd X_curr = MatrixXd::Zero(n, 2);
    VectorXd J_curr = VectorXd::Zero(n);
    VectorXd X_flat = VectorXd::Zero(2 * n);
    VectorXd V_flat = VectorXd::Zero(2 * n);
    VectorXd f_ext_flat = VectorXd::Zero(2 * n);
    VectorXd dscorr_dx = VectorXd::Zero(2 * n);

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
    x_hat = X_flat + dt * V_flat + dt_sqr * f_ext_flat;
    
    // Diagonal of particle masses
    M.setIdentity();
    M *= m;
   
    // V_block diagonal of particle volumes
    V_b.setIdentity();
    V_b *= vol;
    V_b_inv.setIdentity();
    V_b_inv /= vol;

    // Hessian
    H.setIdentity();
    H *= kappa_dt_sqr * vol;
    H_inv.setIdentity();
    H_inv /= kappa_dt_sqr * vol;

    SimplicialLLT<SparseMatrix<double>> solver;

    // Newton solver
    for (int it = 0; it < iters; it++) {
        // Flatten current position matrix
        for (int i = 0; i < n; i++) {
            X_flat(2 * i) = X_curr(i, 0);
            X_flat(2 * i + 1) = X_curr(i, 1);
        }

        // Assemble B matrix
        double fac = 10/7/M_PI/h/h/rho_0;

        std::vector<Triplet<double>> B_triplets;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                RowVector2d diff = X_curr.row(j) - X_curr.row(i);
                double r = diff.norm() / h;
                double deriv = cubic_bspline_derivative(r, m*fac);

                if (deriv != 0.0) {
                    // dci_dxj
                    // Negating because constraint is c(J,x) = J - J(x) 
                    RowVector2d dc_dx = -(deriv * diff / r / h);
                    B_triplets.push_back(Triplet<double>(i, 2 * j, dc_dx(0)));
                    B_triplets.push_back(Triplet<double>(i, 2 * j + 1, dc_dx(1)));
                }

                // RowVector2d dc_dx;
                // bool is_neighbor = false;
                // if (q <= 1 && q > 0){
                //     dc_dx = fac * m * (3*q - 9 * q * q/4) * diff/q;
                //     is_neighbor = true;
                // }
                // else if (q > 1 && q <= 2){
                //     dc_dx = fac * m * 0.75 * (2 - q) * (2 - q) * q * diff / q;
                //     is_neighbor = true;
                // }
            }
        }
        B.setFromTriplets(B_triplets.begin(), B_triplets.end());
        
        auto Jx_func = [&](const VectorXd& x, VectorXd& Jx) {
            Jx.setZero();
            for (int i = 0; i < n; i++){
                for (int j = 0; j < n; j++){
                    auto& xi = x.segment<2>(2 * i);
                    auto& xj = x.segment<2>(2 * j);
                    double r = (xj - xi).norm()/h;
                    Jx(i) += cubic_bspline(r, m*fac);
                }
            }
        };

        Jx_func(X_flat, Jx);

        std::cout << "Jx = " << Jx(5) << " " << Jx(16) << " " << Jx(24)  << std::endl;
        
        dscorr_dx.setZero();
        d2sc_dx2.setZero();

        for (int i = 0; i < n; i++){
            for (int j = 0; j < n; j++){
                auto& xi = X_flat.segment<2>(2 * i);
                auto& xj = X_flat.segment<2>(2 * j);
                double r = (xj - xi).norm()/h;
                if (r > 0){
                    dscorr_dx(2*i) += pow(cubic_bspline(r, m*fac), (n_corr-1)) *
                        cubic_bspline_derivative(r, m*fac) * (xi(0) - xj(0)) / (r*h + 0.0001); 
                
                    dscorr_dx(2*i+1) += pow(cubic_bspline(r, m*fac), (n_corr-1)) *
                        cubic_bspline_derivative(r, m*fac) * (xi(1) - xj(1)) / (r*h + 0.0001); 

                    d2sc_dx2(2*i, 2*j) += (n_corr-1) * pow(cubic_bspline(r, m*fac), (n_corr-2)) *
                        pow(cubic_bspline_derivative(r, m*fac),2) * (xi(0) - xj(0)) * (xi(0) - xj(0))/ (r*h*r*h + 0.0001) -
                        pow(cubic_bspline(r, m*fac), (n_corr-1)) *
                        cubic_bspline_derivative(r, m*fac) / (r*h + 0.0001);

                    d2sc_dx2(2*i, 2*j+1) += (n_corr-1) * pow(cubic_bspline(r, m*fac), (n_corr-2)) *
                        pow(cubic_bspline_derivative(r, m*fac),2) * (xi(0) - xj(0)) * (xj(1) - xi(1)) / (r*h*r*h + 0.0001);

                    d2sc_dx2(2*i+1, 2*j) += (n_corr-1) * pow(cubic_bspline(r, m*fac), (n_corr-2)) *
                        pow(cubic_bspline_derivative(r, m*fac),2) * (xi(1) - xj(1)) * (xj(0) - xi(0)) / (r*r*h*h + 0.0001);

                    d2sc_dx2(2*i+1, 2*j+1) += (n_corr-1) * pow(cubic_bspline(r, m*fac), (n_corr-2)) *
                        pow(cubic_bspline_derivative(r, m*fac),2) * (xi(1) - xj(1)) * (xi(1) - xj(1)) / (r*r*h*h + 0.0001) -
                        pow(cubic_bspline(r, m*fac), (n_corr-1)) *
                        cubic_bspline_derivative(r, m*fac) / (r*h + 0.0001);
                }

                if (0 < r < 1){
                    d2sc_dx2(2*i, 2*j) += pow(cubic_bspline(r, m*fac), (n_corr-1)) 
                        * ((m*fac) * (-3 + 9 * r/2)) * (xj(0) - xi(0)) * (xi(0) - xj(0)) / (r*r*h*h + 0.0001);
                    
                    d2sc_dx2(2*i, 2*j+1) += pow(cubic_bspline(r, m*fac), (n_corr-1))
                        * ((m*fac) * (-3 + 9 * r/2)) * (xi(0) - xj(0)) * (xj(1) - xi(1)) / (r*r*h*h + 0.0001);

                    d2sc_dx2(2*i+1, 2*j) += pow(cubic_bspline(r, m*fac), (n_corr-1))
                        * ((m*fac) * (-3 + 9 * r/2)) * (xi(1) - xj(1)) * (xj(0) - xi(0)) / (r*r*h*h + 0.0001);
                    
                    d2sc_dx2(2*i+1, 2*j+1) += pow(cubic_bspline(r, m*fac), (n_corr-1))
                        * ((m*fac) * (-3 + 9 * r/2)) * (xj(1) - xi(1)) * (xi(1) - xj(1)) / (r*r*h*h + 0.0001);
                
                }
                else if (r < 2){
                    d2sc_dx2(2*i, 2*j) += pow(cubic_bspline(r, m*fac), (n_corr-1)) 
                        * ((m*fac) * (3/2) * (2-r)) * (xj(0) - xi(0)) * (xi(0) - xj(0)) / (r*r*h*h + 0.0001);
                    d2sc_dx2(2*i, 2*j+1) += pow(cubic_bspline(r, m*fac), (n_corr-1)) 
                        * ((m*fac) * (3/2) * (2-r)) * (xi(0) - xj(0)) * (xj(1) - xi(1)) / (r*r*h*h + 0.0001);
                    d2sc_dx2(2*i+1, 2*j) += pow(cubic_bspline(r, m*fac), (n_corr-1)) 
                        * ((m*fac) * (3/2) * (2-r)) * (xi(1) - xj(1)) * (xj(0) - xi(0)) / (r*r*h*h + 0.0001);
                    d2sc_dx2(2*i+1, 2*j+1) += pow(cubic_bspline(r, m*fac), (n_corr-1)) 
                        * ((m*fac) * (3/2) * (2-r)) * (xj(1) - xi(1)) * (xi(1) - xj(1)) / (r*r*h*h + 0.0001);
                }
                
            }
        }
        dscorr_dx *= -0.1/pow(cubic_bspline(0.2, m*fac), n_corr) * n_corr * 2;
        d2sc_dx2 *= -0.1/pow(cubic_bspline(0.2, m*fac), n_corr) * n_corr * 2;
    
        // Check finite difference
        fd::AccuracyOrder accuracy = fd::SECOND;
        
        const auto scorr = [&](const Eigen::VectorXd& x) -> double {
            double sc = 0; 
            for (int i = 0; i < n; i++){
                for (int j = 0; j < n; j++){
                    auto& xi = x.segment<2>(2 * i);
                    auto& xj = x.segment<2>(2 * j); 
                    double r = (xj - xi).norm()/h;
                    sc += -0.1 * pow(cubic_bspline(r, m*fac)/cubic_bspline(0.2, m*fac), n_corr);
                }
            }
            return sc;
        };

        Eigen::VectorXd fdscorr_dx;
        fd::finite_gradient(X_flat, scorr, fdscorr_dx, accuracy);
        std::cout << "Norm Gradient: " << (fdscorr_dx-dscorr_dx).norm() << std::endl;
        std::cout << "finite diff: " << fd::compare_gradient(fdscorr_dx, dscorr_dx, 1e-6) << std::endl;
        std::cout << fdscorr_dx(10) << " " << dscorr_dx(10) << std::endl;
        std::cout << fdscorr_dx(42) << " " << dscorr_dx(42) << std::endl;

        Eigen::MatrixXd fd2sc_dx2;
        fd::finite_hessian(X_flat, scorr, fd2sc_dx2, accuracy);
        std::cout << "Norm Hessian: "<< (fd2sc_dx2-dscorr_dx).norm() << std::endl;
        std::cout << "finite diff: " << fd::compare_hessian(fd2sc_dx2, d2sc_dx2, 1e-6) << std::endl;
        std::cout << fd2sc_dx2(10,5) << " " << d2sc_dx2(10,5) << std::endl;
        std::cout << fd2sc_dx2(42,1) << " " << d2sc_dx2(42,1) << std::endl;


        VectorXd dpsi_dJ = kappa_dt_sqr * (J_curr - VectorXd::Ones(n));


        A = M + B.transpose() * (V_b_inv * H * V_b_inv) * B + d2sc_dx2.sparseView();
        b = -M * (X_flat - x_hat) - dscorr_dx
          + B.transpose() * (V_b_inv * dpsi_dJ
          + V_b_inv*H*V_b_inv*(Jx - J_curr));

        std::cout << "d2sc_dx2 size" << d2sc_dx2.size() << std::endl;
        std::cout << "dscorr_dx size" << dscorr_dx.size() << std::endl;
        std::cout << "A size" << A.size() << std::endl;
        std::cout << "b size" << b.size() << std::endl;
        std::cout << "b - inertial: " << (M * (X_flat - x_hat) ).norm() << std::endl;
        std::cout << "b - tensile instability: " << dscorr_dx.norm() << std::endl;
        std::cout << "b - elastic: " << (B.transpose()* (V_b_inv * dpsi_dJ)).norm() << std::endl;
        std::cout << "b - constraint: " << (B.transpose()* (V_b_inv*H*V_b_inv*(J_curr-Jx))).norm() << std::endl;


        std::cout << "start solve" << std::endl;
        solver.compute(A);
        if (solver.info() != Success) {
            std::cout << "decomposition failed" << std::endl;
        }

        VectorXd X_new_flat, delta_X, J_new, delta_J, lambda, lambda_1, lambda_2, lambda_3;
        X_new_flat.resize(n*2);
        delta_X.resize(n*2);
        J_new.resize(n);
        delta_J.resize(n);
        lambda.resize(n);
        lambda_1.resize(n);
        lambda_2.resize(n);
        lambda_3.resize(n);

        X_new_flat.setZero();
        J_new.setZero();
        delta_J.setZero();
        lambda.setZero();
        lambda_1.setZero();
        lambda_2.setZero();
        lambda_3.setZero();

        delta_X = solver.solve(b);

        std::cout << "solved" << std::endl;
        //delta_J = V_b_inv * (Jx - J_curr - B * delta_X);
        //lambda = -V_b_inv * (H * delta_J + dpsi_dJ);
        lambda_1 = -V_b_inv * kappa_dt_sqr * (J_curr - VectorXd::Ones(n));
        lambda_2 = V_b_inv * H * V_b_inv * (J_curr - Jx);
        lambda_3 = V_b_inv * H * V_b_inv * B * delta_X;
        lambda = lambda_1 + lambda_2 + lambda_3;
        delta_J = -H_inv * (dpsi_dJ + V_b * lambda);

        //do line search
        double alpha = 1.0;

        // Energy lambda function for use in line search
        auto energy_func = [&](double alpha) {
            X_new_flat = X_flat + alpha * delta_X;
            J_new = J_curr + alpha * delta_J;

            // Inertial energy
            double e_i = 0.5 * (X_new_flat - x_hat).transpose() * M
                       * (X_new_flat - x_hat);
            // Mixed potential energy
            double e_psi = 0.5 * kappa_dt_sqr
                         * (J_new - VectorXd::Ones(n)).squaredNorm();
            // Mixed constraint energy
            Jx_func(X_new_flat, Jx);
            double e_c = lambda.dot(J_new - Jx);
            std::cout << "e_i: " << e_i << std::endl;
            std::cout << "e_psi: " << e_psi << std::endl;
            std::cout << "e_c: " << e_c << std::endl;
            return e_i + e_psi + e_c;
        };
        
        e0 = energy_func(0);
        double e_new = energy_func(1.0);

        //while (e_new > e0 && alpha > 1e-10){ 
        //    //std::cout << "alpha: " << alpha << std::endl;
        //    alpha *= 0.5;
        //    e_new = energy_func(alpha);
        //}
        
        // std::cout << X_new_flat - X_flat << std::endl;
        std::cout << "alpha: " << alpha << std::endl;
        std::cout << "e_new: " << e_new << std::endl;
        std::cout << "e0: " << e0 << std::endl;
        std::cout << "delta_X = " << (X_new_flat - X_flat).norm() << std::endl;
        std::cout << "delta_J = " << (J_new - J_curr).norm() << std::endl;
        std::cout << "lambda norm = " << lambda.norm() << std::endl;
        std::cout << "J_curr = " << Jx(5) << " " << J_curr(16) << " " << J_curr(24)  << std::endl;
        std::cout << "J_new = " << Jx(5) << " " << J_new(16) << " " << J_new(24)  << std::endl;
        std::cout << "Jx = " << Jx(5) << " " << Jx(16) << " " << Jx(24)  << std::endl;
        std::cout << "lambda = " << lambda(5) << " " << lambda(16) << " " << lambda(24)  << std::endl;

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
