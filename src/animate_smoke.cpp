#include "animate_smoke.h"
#include "calculate_lambda.h"
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <finitediff.hpp>
#include <cassert>
#include <iostream>
#include <fstream>
#include <chrono>

//#include "CompactNSearch"


#define _USE_MATH_DEFINES
#include <math.h>

using namespace Eigen;

namespace {
}

template <>
void animate_smoke<2>(
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
        double kappa = 10;// 1000;//100000;
        double rho_0 = 1; //define later
        double m = 1;
        double h = 0.1; // h for particle distance
        double vol = 1;//m/rho_0;
        double n_corr = 4; //n for s_corr in tensile instability term
    
        std::ofstream output_file("output.txt", std::ios::app);

        const double kappa_dt_sqr = kappa * dt * dt;
        const double dt_sqr = dt * dt;
        MatrixXd f_ext(n, 2);
        f_ext.setZero();
        f_ext.col(1).setConstant(-9.8);
    
        VectorXd x_hat = VectorXd::Zero(2 * n);
    
        double e0;
        VectorXd X_flat = VectorXd::Zero(2 * n);
        VectorXd V_flat = VectorXd::Zero(2 * n);
        VectorXd dscorr_dx = VectorXd::Zero(2 * n);
        MatrixXd d2sc_dx2; //d2c_dx2
        
        d2sc_dx2.resize(2 * n, 2 * n);
        dscorr_dx.setZero();
        d2sc_dx2.setZero();

        for (int i = 0; i < n; i++) {
            X_flat(2 * i) = X(i, 0);
            X_flat(2 * i + 1) = X(i, 1);
        }

        double dq = 0.98; // 0.8 - 1.0 seem to be reasonable values
        double fac = 10/7/M_PI;///h/h;

        double W_dq = cubic_bspline(dq, fac);
        double k_spring = dt_sqr * 1000; //500000000;
        
        for (int i = 0; i < n; i++){
            for (int j = 0; j < n; j++){
                if (i == j) continue;
                const auto& xi = X_flat.segment<2>(2 * i);
                const auto& xj = X_flat.segment<2>(2 * j);
                double r = (xj - xi).norm()/h;
                double eps = 1e-6;
                double Wij = cubic_bspline(r, m*fac);
                //if (Wij < eps) continue;

                // Tensile instability energy:
                // E(x) = 0.5 * k_spring * \sum_i \sum_j (W_ij - W_dq)^2
                
                // Kernel derivative w.r.t to xi
                Vector2d dr_dxi = -norm_derivative<2>((xj-xi)/h, r) / h;
                double dphi_dr = cubic_bspline_derivative(r, m*fac);
                Vector2d dW_dxi = dphi_dr * dr_dxi; 

                // Spring energy derivative
                dscorr_dx.segment<2>(2*i) += (Wij - W_dq) * dW_dxi;
                dscorr_dx.segment<2>(2*j) += -(Wij - W_dq) * dW_dxi;

                // Hessian components for spring energy
                
                // Distance hessian w.r.t to xi
                // off-diagonal blocks are negative of this
                Matrix2d d2r_dxi2 = norm_hessian<2>((xj-xi)/h, r) / h / h;
                double d2phi_dr2 = cubic_bspline_hessian(r, m*fac);

                // weight, wij, hessian
                Matrix2d d2w_dxi2 = 
                    dphi_dr * d2r_dxi2 + 
                    dr_dxi * dr_dxi.transpose() * d2phi_dr2;

                // energy (wij - w_dq)^2 hessian
                Matrix2d hess_ii = (Wij - W_dq) * d2w_dxi2 +
                    dW_dxi * dW_dxi.transpose();

                // Diagonal blocks
                d2sc_dx2.block<2, 2>(2*i, 2*i) += hess_ii;     
                d2sc_dx2.block<2, 2>(2*j, 2*j) += hess_ii;     

                // Off-diagonals
                d2sc_dx2.block<2, 2>(2*i, 2*j) += -hess_ii;     
                d2sc_dx2.block<2, 2>(2*j, 2*i) += -hess_ii;

                ////d2c_dx2.block<2, 2>(2*i, 2*j) =  -Wij_hess * lambda(i)/rho_0;
                ////d2c_dx2.block<2, 2>(2*j, 2*i) =  -Wij_hess * lambda(i)/rho_0;
            }
        }
        dscorr_dx *= k_spring;
        d2sc_dx2 *= k_spring;
        //std::cout << "dscorr_dx: " << dscorr_dx << std::endl;
        //std::cout << "d2sc_dx2: " << d2sc_dx2 << std::endl;

        //Finite difference check
        fd::AccuracyOrder accuracy = fd::SECOND;
            
        const auto scorr = [&](const Eigen::VectorXd& x) -> double {
            double sc = 0; 
            for (int i = 0; i < n; i++){
                for (int j = 0; j < n; j++){
                    if (i == j) continue;
                    const auto& xi = x.segment<2>(2 * i);
                    const auto& xj = x.segment<2>(2 * j);
                    double r = (xj - xi).norm()/h;
                    double eps = 1e-6;
                    double Wij = cubic_bspline(r, m*fac);
                    //std::cout << xi << xj << std::endl;
                        //if (Wij < eps) continue;
                    sc += 0.5 * k_spring * (Wij - W_dq) * (Wij - W_dq);
                }
            }
            //std::cout<< "sc" << sc << std::endl;
            return sc;
        };

        Eigen::VectorXd fdscorr_dx;
        fd::finite_gradient(X_flat, scorr, fdscorr_dx, accuracy, 1.0e-7);
        //std::cout << "Gradient Norm: " << (fdscorr_dx).norm() << std::endl;
        std::cout << X_flat << std::endl;
        //std::cout << "dscorr:" << dscorr_dx << std::endl;
        //std::cout << "fdscorr:" << fdscorr_dx << std::endl;
        std::cout << "Gradient Error: " << (dscorr_dx - fdscorr_dx).norm() << std::endl;

        Eigen::MatrixXd fd2sc_dx2;
        fd::finite_hessian(X_flat, scorr, fd2sc_dx2, accuracy, 1.0e-7);
        std::cout << "Hessian Error: " << (fd2sc_dx2 - d2sc_dx2).norm() << std::endl;
        std::cout << "fd" << std::endl;
        std::cout << fd2sc_dx2 << std::endl;
        std::cout << "d2sc_dx2" << std::endl;
        std::cout << d2sc_dx2 << std::endl;
   
    return;
}

template <>
void animate_smoke<3>(
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
