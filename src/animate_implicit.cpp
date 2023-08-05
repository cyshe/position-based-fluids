#include "animate_implicit.h"
#include "calculate_lambda.h"
#include <igl/signed_distance.h>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <finitediff.hpp>
#include <cassert>
#include <iostream>
#include <fstream>
#include <chrono>
#include "TinyAD/Scalar.hh"
#include "TinyAD/ScalarFunction.hh"
#include "TinyAD/VectorFunction.hh"

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
    const double dt,
    const double kappa,
    const bool fd_check,
    const bool converge_check
    ){

    int n = numofparticles;
    double rho_0 = 1; //define later
    double m = 1;
    double h = 0.1; // h for particle distance
    double vol = 1;//m/rho_0;
    double n_corr = 4; //n for s_corr in tensile instability term
    int it = 0; //iteration number

    std::ofstream output_file("output.txt", std::ios::app);

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

    MatrixXd d2sc_dx2; //d2c_dx2
    d2sc_dx2.resize(2 * n, 2 * n);
    //d2c_dx2.resize(2 * n, 2 * n);

    // Vectors
    VectorXd b = VectorXd::Zero(2 * n);
    VectorXd Jx = VectorXd::Zero(n);
    MatrixXd X_curr = MatrixXd::Zero(n, 2);
    VectorXd J_curr = VectorXd::Zero(n);
    VectorXd X_flat = VectorXd::Zero(2 * n);
    VectorXd V_flat = VectorXd::Zero(2 * n);
    VectorXd f_ext_flat = VectorXd::Zero(2 * n);
    VectorXd dscorr_dx = VectorXd::Zero(2 * n);
    VectorXd lambda = VectorXd::Zero(2 * n);

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

    SimplicialLDLT<SparseMatrix<double>> solver;

    // Newton solver
    while (converge_check || it < iters) {
        // Flatten current position matrix
        //auto begin = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < n; i++) {
            X_flat(2 * i) = X_curr(i, 0);
            X_flat(2 * i + 1) = X_curr(i, 1);
        }

        // Assemble B matrix
        double fac = 10/7/M_PI;///h/h;

        std::vector<Triplet<double>> B_triplets;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                RowVector2d diff = X_curr.row(j) - X_curr.row(i);
                double r = diff.norm() / h;
                double deriv = cubic_bspline_derivative(r, m*fac) / rho_0;

                if (deriv != 0.0) {
                    // dci_dxj
                    // Negating because constraint is c(J,x) = J - J(x) 
                    RowVector2d dc_dx = -(deriv * diff / r / h);
                    B_triplets.push_back(Triplet<double>(i, 2 * j, dc_dx(0)));
                    B_triplets.push_back(Triplet<double>(i, 2 * j + 1, dc_dx(1)));
                }
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
                    Jx(i) += cubic_bspline(r, m*fac)/rho_0;
                }
            }
        };

        Jx_func(X_flat, Jx);

        //std::cout << "Jx = " << Jx(5) << " " << Jx(16) << " " << Jx(24)  << std::endl;
        
        //d2c_dx2.setZero();
        dscorr_dx.setZero();
        d2sc_dx2.setZero();
        //auto end = std::chrono::high_resolution_clock::now();
        //auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);

        //printf("Time measured 1: %.3f seconds.\n", elapsed.count() * 1e-9);

        //begin = std::chrono::high_resolution_clock::now();
        std::vector<Eigen::Vector2i> elements;



        for (int i = 0; i < n; i++){
            for (int j = 0; j < n; j++){
                const auto& xi = X_flat.segment<2>(2 * i);
                const auto& xj = X_flat.segment<2>(2 * j);
                if ((xj - xi).norm() < h && (xj - xi).norm() > 0){
                    elements.push_back(Eigen::Vector2i(i,j));
                }
            }
        }

        double dq = 0.98; // 0.8 - 1.0 seem to be reasonable values
        double k_spring = dt_sqr * 100; //500000000;
        double W_dq = cubic_bspline(dq, fac);

        auto func = TinyAD::scalar_function<2>(TinyAD::range(n));

        func.add_elements<2>(TinyAD::range(elements.size()), [&] (auto& element) -> TINYAD_SCALAR_TYPE(element){
            using T = TINYAD_SCALAR_TYPE(element);
            int idx = element.handle;
            Eigen::Vector2<T> xi = element.variables(elements[idx](0));
            Eigen::Vector2<T> xj = element.variables(elements[idx](1));
            T r = (xj - xi).norm()/h; //squaredNorm()/h;
            T Wij = cubic_bspline(r, T(m*fac));
            return 0.5 * k_spring * (Wij - W_dq) * (Wij - W_dq);
        });

        std::cout << "Evaluate gradient and hessian" << std::endl;
        auto [f, g, H_proj] = func.eval_with_hessian_proj(X_flat);
        dscorr_dx = g;
        d2sc_dx2 = H_proj;

/*
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
        d2sc_dx2 *= k_spring;*/


        if (fd_check) {
            fd::AccuracyOrder accuracy = fd::SECOND;
            
            const auto scorr = [&](const Eigen::VectorXd& x) -> double {
                double sc = 0; 
                for (int i = 0; i < n; i++){
                    for (int j = 0; j < n; j++){
                        if (i == j) continue;
                        const auto& xi = X_flat.segment<2>(2 * i);
                        const auto& xj = X_flat.segment<2>(2 * j);
                        double r = (xj - xi).norm()/h;
                        double eps = 1e-6;
                        double Wij = cubic_bspline(r, m*fac);
                        //if (Wij < eps) continue;
                        sc += 0.5 * k_spring * (Wij - W_dq) * (Wij - W_dq);
                        //std::cout<< "sc" << sc << std::endl;
                    }
                }
                return sc;
            };

            Eigen::VectorXd fdscorr_dx;
            fd::finite_gradient(X_flat, scorr, fdscorr_dx, accuracy, 1.0e-7);
        //std::cout << "Gradient Norm: " << (fdscorr_dx).norm() << std::endl;
            //std::cout << X_flat << std::endl;
        //std::cout << "dscorr:" << dscorr_dx << std::endl;
        //std::cout << "fdscorr:" << fdscorr_dx << std::endl;
            std::cout << "Gradient Error: " << (dscorr_dx - fdscorr_dx).array().abs().maxCoeff() << std::endl;

            Eigen::MatrixXd fd2sc_dx2;
            fd::finite_hessian(X_flat, scorr, fd2sc_dx2, accuracy, 1.0e-5);
            std::cout << "Hessian error: " << (fd2sc_dx2 - d2sc_dx2).norm() << std::endl;
            std::cout << "------------------" <<std::endl;
            std::cout << fd2sc_dx2(10,5) << " " << d2sc_dx2(10,5) << std::endl;
            std::cout << fd2sc_dx2.row(0) << std::endl; 
            std::cout << d2sc_dx2.row(0) << std::endl;
        }
        //end = std::chrono::high_resolution_clock::now();
        //elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);

        //printf("Time measured 2: %.3f seconds.\n", elapsed.count() * 1e-9);

        //begin = std::chrono::high_resolution_clock::now();

        //std::cout << "symmetry" << (d2c_dx2 - d2c_dx2.transpose()).norm() << std::endl;

        VectorXd dpsi_dJ = kappa_dt_sqr * (J_curr - VectorXd::Ones(n));
        A = M + B.transpose() * (V_b_inv * H * V_b_inv) * B + d2sc_dx2.sparseView();// + d2c_dx2.sparseView(); 
        b = -(M) * (X_flat - x_hat) - dscorr_dx
          + B.transpose() * (V_b_inv * dpsi_dJ
          + V_b_inv*H*V_b_inv*(Jx - J_curr));

        //std::cout << "d2sc_dx2 size" << d2sc_dx2.size() << std::endl;
        //std::cout << "dscorr_dx size" << dscorr_dx.size() << std::endl;
        //std::cout << "A size" << A.size() << std::endl;
        //std::cout << "b size" << b.size() << std::endl;
        //std::cout << "b - inertial: " << (M * (X_flat - x_hat) ).norm() << std::endl;
        //std::cout << "b - tensile instability: " << dscorr_dx.norm() << std::endl;
        //std::cout << "b - elastic: " << (B.transpose()* (V_b_inv * dpsi_dJ)).norm() << std::endl;
        //std::cout << "b - constraint: " << (B.transpose()* (V_b_inv*H*V_b_inv*(J_curr-Jx))).norm() << std::endl;


        //std::cout << "start solve" << std::endl;
        solver.compute(A);
        if (solver.info() != Success) {
            std::cout << "decomposition failed" << std::endl;
            exit(1);
        }

        //end = std::chrono::high_resolution_clock::now();
        //elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);

        //printf("Time measured 3: %.3f seconds.\n", elapsed.count() * 1e-9);

        //begin = std::chrono::high_resolution_clock::now();

        VectorXd X_new_flat, delta_X, J_new, delta_J, lambda_1, lambda_2, lambda_3;
        X_new_flat.resize(n*2);
        delta_X.resize(n*2);
        J_new.resize(n);
        delta_J.resize(n);
        lambda_1.resize(n);
        lambda_2.resize(n);
        lambda_3.resize(n);

        X_new_flat.setZero();
        J_new.setZero();
        delta_J.setZero();
        lambda_1.setZero();
        lambda_2.setZero();
        lambda_3.setZero();

        delta_X = solver.solve(b);

        //std::cout << "solved" << std::endl;
        lambda_1 = -V_b_inv * kappa_dt_sqr * (J_curr - VectorXd::Ones(n));
        lambda_2 = V_b_inv * H * V_b_inv * (J_curr - Jx);
        lambda_3 = V_b_inv * H * V_b_inv * B * delta_X;
        lambda = lambda_1 + lambda_2 + lambda_3;
        delta_J = -H_inv * (dpsi_dJ + V_b * lambda);

        //end = std::chrono::high_resolution_clock::now();
        //elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);

        //printf("Time measured 4: %.3f seconds.\n", elapsed.count() * 1e-9);

        //begin = std::chrono::high_resolution_clock::now();
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
            
            //E(x) = 0.5 * k_spring * \sum_i \sum_j (W_ij - W_dq)^2
            double e_s = 0;
            double W_dq = cubic_bspline(dq, fac);
            for (int i = 0; i < n; i++){
                for (int j = 0; j < n; j++){
                    if (i == j) continue;
                    const auto& xi = X_new_flat.segment<2>(2 * i);
                    const auto& xj = X_new_flat.segment<2>(2 * j);
                    double r = (xj - xi).norm()/h;
                    double eps = 1e-6;
                    double Wij = cubic_bspline(r, m*fac);
                    if (Wij < eps) continue;
                    e_s += 0.5 * k_spring * (Wij - W_dq) * (Wij - W_dq);
                }
            }
            //std::cout << "e_i: " << e_i << std::endl;
            //std::cout << "e_psi: " << e_psi << std::endl;
            //std::cout << "e_c: " << e_c << std::endl;
            //std::cout << "e_s: " << e_s << std::endl;
            return e_i + e_psi + e_c + e_s;
        };
        
        e0 = energy_func(0);
        std::cout << "e0: " << e0 << std::endl;
        double e_new = energy_func(1.0);

        while (e_new > e0 && alpha > 1e-10){ 
        //    //std::cout << "alpha: " << alpha << std::endl;
            alpha *= 0.5;
            e_new = energy_func(alpha);
        }
        
        if (alpha < 1e-10 && it == 0){
            std::cout << "line search failed" << std::endl;
            SelfAdjointEigenSolver<MatrixXd> es;
            es.compute(MatrixXd(A));
            std::cout << "The eigenvalues of A are: " << es.eigenvalues().transpose().head(10) << std::endl;
            //std::cout << delta_X << std::endl;
            exit(1);
        }

//        SelfAdjointEigenSolver<MatrixXd> es;
//        es.compute(MatrixXd(A));
//        std::cout << "The eigenvalues of A are: " << es.eigenvalues().transpose() << std::endl;
        
        //end = std::chrono::high_resolution_clock::now();
        //elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);

        //printf("Time measured 5: %.3f seconds.\n", elapsed.count() * 1e-9);

        
        // std::cout << X_new_flat - X_flat << std::endl;
        //std::cout << "alpha: " << alpha << std::endl;
        //std::cout << "e_new: " << e_new << std::endl;
        //std::cout << "e0: " << e0 << std::endl;
        //std::cout << "delta_X = " << (X_new_flat - X_flat).norm() << std::endl;
        //std::cout << "delta_J = " << (J_new - J_curr).norm() << std::endl;
        //std::cout << "lambda norm = " << lambda.norm() << std::endl;
        //std::cout << "J_curr = " << Jx(5) << " " << J_curr(16) << " " << J_curr(24)  << std::endl;
        //std::cout << "J_new = " << Jx(5) << " " << J_new(16) << " " << J_new(24)  << std::endl;
        //std::cout << "Jx = " << Jx(5) << " " << Jx(16) << " " << Jx(24)  << std::endl;
        //std::cout << "lambda = " << lambda(5) << " " << lambda(16) << " " << lambda(24)  << std::endl;

        output_file << (M * (X_flat - x_hat) ).norm()<< ", " 
        << dscorr_dx.norm() << ", " 
        << (B.transpose()* (V_b_inv * dpsi_dJ)).norm() << ", "
        << (B.transpose()* (V_b_inv*H*V_b_inv*(J_curr-Jx))).norm() << ", "
        << lambda.norm() << ", "
        << (J_new - J_curr).norm() << ", "
        << (X_new_flat - X_flat).norm() << std::endl;

        for (int i = 0; i < n; i++) {
            X_curr(i, 0) = X_new_flat(2*i);
            X_curr(i, 1) = X_new_flat(2*i+1);
            J_curr(i) = J_new(i);
        }
        std::cout << "iteration: " << it << "," << delta_X.norm() << std::endl;
        it += 1;
        if (delta_X.norm()/n < 2e-3) {
            std::cout << "converged" << std::endl;
            break;
        }
    }        

    V = (X_curr-X)/dt;
    X = X_curr;
    J = J_curr;     
        
    //std::cout << X << std::endl;
    // boundary detection
    
    for (int i = 0; i < n; i++){
        for (int j = 0; j < 2; ++j) {
            if (X(i,j) < low_bound(j)) {
                V(i, j) = 0; //abs(V(i, j));
                X(i,j) = low_bound(j);
            }
            if (X(i,j) > up_bound(j)) {
                V(i, j) = 0;// -1* abs(V(i, j));
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
    VectorXd & J,
    MatrixXi & N,
    const Eigen::Matrix<double, 3, 1> & low_bound,
    const Eigen::Matrix<double, 3, 1> & up_bound,
    const int numofparticles,
    const int iters, 
    const double dt,
    const double kappa,
    const bool fd_check,
    const bool converge_check
    ){}
