#pragma once
#include "animate_implicit.h"
#include "calculate_lambda.h"
#include "surface_tension.h"
#include "boundary_conditions.h"
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

#include "spacing_energy.h"
#include "pressure.h"
#include "find_neighbors_brute_force.h"
#include "calculate_densities.h"
#include "cubic_bspline.h"

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
    VectorXd & Jx,
    MatrixXi & N,
    Eigen::MatrixXd & grad_i,
    Eigen::MatrixXd & grad_psi,
    Eigen::MatrixXd & grad_s,
    Eigen::MatrixXd & grad_st,
    const Eigen::Matrix<double, 2, 1> & low_bound,
    const Eigen::Matrix<double, 2, 1> & up_bound,
    const int numofparticles,
    const int iters, 
    const double dt,
    const double kappa,
    const double k_st,
    const double k_s,
    const double st_threshold,
    const double rho_0,
    const double gravity,
    const bool fd_check,
    const bool bounds_bool,
    const bool converge_check,
    const bool do_line_search,
    const bool smooth_mol,
    const bool psi_bool,
    const bool spacing_bool,
    const bool st_bool,
    const bool primal
    ){

    std::ofstream output_file("output.txt", std::ios::app);

    const int n = numofparticles;
    const double m = 1;
    const double h = 0.3; // h for particle distance
    const double vol = 1;//m/rho_0;
    const double fac = 10/7/M_PI; // bspline normalizing coefficient

    // Energy scales
    const double dt_sqr = dt * dt;
    const double kappa_dt_sqr = dt_sqr * kappa; 
    const double k_st_dt_sqr = dt_sqr * k_st;   // surface tension

    // Spacing energy params
    const double dq = 0.98; // 0.8 - 1.0 seem to be reasonable values
    const double k_spacing = dt_sqr * k_s;
    const double W_dq = cubic_bspline(dq, fac); // fixed kernel value at dq

    MatrixXd f_ext(n, 2);
    f_ext.setZero();
    f_ext.col(1).setConstant(gravity);
        
    // Sparse matrices
    SparseMatrix<double> A, M, B, H, V_b, V_b_inv, H_inv;
    A.resize(2 * n, 2 * n);
    M.resize(2 * n, 2 * n);
    B.resize(n, 2 * n);
    H.resize(n, n);
    H_inv.resize(n, n);
    V_b.resize(n, n);
    V_b_inv.resize(n, n);

    // Vectors
    VectorXd b = VectorXd::Zero(2 * n);
    //VectorXd 
    Jx = VectorXd::Zero(n);
    VectorXd lambda = VectorXd::Zero(2 * n);

    // row-wise flatten function
    auto field_to_vec = [](const MatrixXd& m) {
        VectorXd v(m.size());
        for (int i = 0; i < m.rows(); i++) {
            for (int j = 0; j < m.cols(); j++) {
                v(i * m.cols() + j) = m(i, j);
            }
        }
        return v;
    };

    VectorXd x = field_to_vec(X);
    MatrixXd X_hat = X + dt * V + dt_sqr * f_ext;
    VectorXd x_hat = field_to_vec(X_hat);

    // Diagonal of particle masses
    M.setIdentity();
    M *= m;
   
    // V diagonal of particle volumes
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
    for (int it = 0; it < iters; it++) {
        //auto begin = std::chrono::high_resolution_clock::now();

        // Initialize new neighbor list
        std::vector<std::vector<int>> neighbors = find_neighbors_brute_force<2>(x, h);

        // Create list of neighbor pairs (as elements for TinyAD)
        std::vector<Eigen::Vector2i> elements;
        for (int i = 0; i < n; i++){
            //if (neighbors[i].size() <= 7) std::cout << "i = " << i << ", neighbors = " << neighbors[i].size() << std::endl; 
            for (int j = 0; j < neighbors[i].size(); j++){
                elements.push_back(Eigen::Vector2i(i,neighbors[i][j]));
            }
        }

        // Calculate densities (as function of x)
        Jx = calculate_densities<2>(x, neighbors, h, m, fac) / rho_0;
        
        // Assemble B matrix -- jacobian w.r.t of the J - J(x) constraint
        std::vector<Triplet<double>> B_triplets;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < neighbors[i].size(); j++) {
                const auto& xi = x.segment<2>(2 * i);
                const auto& xj = x.segment<2>(2 * neighbors[i][j]);

                // negating gradient because constraint is (J - J(x))
                Vector4d density_grad = -density_gradient<2>(xi, xj, h, m, fac) / rho_0;

                B_triplets.push_back(Triplet<double>(i, 2 * i, density_grad(0)));
                B_triplets.push_back(Triplet<double>(i, 2 * i + 1, density_grad(1)));
                B_triplets.push_back(Triplet<double>(i, 2 * neighbors[i][j], density_grad(2)));
                B_triplets.push_back(Triplet<double>(i, 2 * neighbors[i][j] + 1, density_grad(3)));
            }
        }
        B.setFromTriplets(B_triplets.begin(), B_triplets.end());
    

        //auto end = std::chrono::high_resolution_clock::now();
         //auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        //printf("Time measured 1: %.3f seconds.\n", elapsed.count() * 1e-9);
        //begin = std::chrono::high_resolution_clock::now();

        // Create spacing energy function and evaluate gradient and hessian
        auto spacing_energy = spacing_energy_func<2>(x, elements, 0.08, m, fac, W_dq, k_spacing);
        std::cout << "Evaluate gradient and hessian" << std::endl;
        auto [f, g_spacing, H_spacing] = spacing_energy.eval_with_hessian_proj(x);


        if (fd_check) {
            fd::AccuracyOrder accuracy = fd::SECOND;
            
            const auto scorr = [&](const Eigen::VectorXd& x) -> double {
                return spacing_energy.eval(x);
            };

            Eigen::VectorXd fg_spacing;
            fd::finite_gradient(x, scorr, fg_spacing, accuracy, 1.0e-7);
            std::cout << "Gradient Error: " << (g_spacing - fg_spacing).array().abs().maxCoeff() << std::endl;

            Eigen::MatrixXd fH_spacing;
            fd::finite_hessian(x, scorr, fH_spacing, accuracy, 1.0e-5);
            std::cout << "Hessian error: " << (fH_spacing - H_spacing).norm() << std::endl;
            std::cout << "------------------" <<std::endl;
            // std::cout << fH_spacing(10,5) << " " << H_spacing(10,5) << std::endl;
            // std::cout << fH_spacing.row(0) << std::endl; 
            // std::cout << H_spacing.row(0) << std::endl;
        }
        //end = std::chrono::high_resolution_clock::now();
        //elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        //printf("Time measured 2: %.3f seconds.\n", elapsed.count() * 1e-9);
        //begin = std::chrono::high_resolution_clock::now();
        
        //Eigen::MatrixXd points = Eigen::Map<MatrixXd>(x.data(), 2, n).transpose();


        // Assemble left and right hand sides of system
        
        A = M;
        if (psi_bool) {
            A += psi_hessian<2>(H, B, V_b_inv, primal);
        }
        if (spacing_bool) {
            A += H_spacing;
        }
        if (st_bool) {
            A += surface_tension_hessian<2>(x, neighbors, h, m, fac, k_st, rho_0 * st_threshold, smooth_mol).sparseView();
        }
        if (bounds_bool) {
            A += bounds_hessian<2>(x, low_bound, up_bound).sparseView();
        }
        
        VectorXd dpsi_dJ = VectorXd::Zero(n);
        VectorXd b_inertial = -M * (x - x_hat);
        VectorXd b_psi = VectorXd::Zero(2 * n);
        VectorXd b_spacing = VectorXd::Zero(2 * n);
        VectorXd b_st = VectorXd::Zero(2 * n);
        VectorXd b_bounds = VectorXd::Zero(2 * n);
        
        // TODO: check as rho_0 * st_thershold is valid boundary condition for both psi and st
        b = b_inertial; // + B.transpose() * (V_b_inv *H *V_b_inv * (J - Jx));
        if (psi_bool) {
            b_psi = -psi_gradient<2>(x, J, neighbors, V_b_inv, B, h, m, fac, kappa, rho_0 * st_threshold, rho_0, primal);
            b += b_psi;
        }
        if (spacing_bool) {
            b_spacing = -g_spacing;
            b += dt * dt * b_spacing;
        }
        if (st_bool) {
            b_st = -surface_tension_gradient<2>(x, neighbors, h, m, fac, k_st, rho_0 * st_threshold, smooth_mol);
            b += dt * dt * b_st;
        }
        if (bounds_bool) {
            b_bounds = -bounds_gradient<2>(x, low_bound, up_bound);
            b += dt * dt * b_bounds;
        }

        // Solve for descent direction
        solver.compute(A);
        if (solver.info() != Success) {
            std::cout << "decomposition failed" << std::endl;
            exit(1);
        }
        VectorXd delta_x = solver.solve(b);
        lambda = V_b_inv * H * V_b_inv * (J - Jx + B * delta_x) 
               - V_b_inv * kappa_dt_sqr * (J - VectorXd::Ones(n));
        
        VectorXd delta_J = -H_inv * (kappa_dt_sqr * (J - VectorXd::Ones(n)) + V_b * lambda);

        //end = std::chrono::high_resolution_clock::now();
        //elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        //printf("Time measured 4: %.3f seconds.\n", elapsed.count() * 1e-9);
        //begin = std::chrono::high_resolution_clock::now();

        // Temporary variables for line search
        VectorXd x_new = VectorXd::Zero(n*2);
        VectorXd J_new = VectorXd::Zero(n);

        // Energy function for line search
        auto energy_func = [&](double alpha) {
            x_new = x + alpha * delta_x;
            J_new = J + alpha * delta_J;

            // Inertial energy
            double e_i = 0.5 * (x_new - x_hat).transpose() * M * (x_new - x_hat);
            
            // Mixed potential energy
            double e_psi = psi_energy<2>(x_new, J_new, neighbors, h, m, fac, kappa, rho_0 * st_threshold, rho_0);
            //0.5 * kappa_dt_sqr * (J_new - VectorXd::Ones(n)).squaredNorm();

            // Mixed constraint energy
            // Jx = calculate_densities<2>(x_new, neighbors, h, m, fac) / rho_0;

            double e_c = lambda.dot(J_new - (calculate_densities<2>(x_new, neighbors, h, m, fac) / rho_0));
            
            // Spacing energy
            double e_s = spacing_energy.eval(x_new);

            // Surface tension energy
            double e_st = surface_tension_energy<2>(x_new, neighbors, h, m, fac, k_st_dt_sqr,
                rho_0 * st_threshold, smooth_mol);

            // Boundary energy
            double e_bound = bounds_energy<2>(x_new, low_bound, up_bound);
            
            return e_i + e_psi + e_c + e_s + e_st + e_bound;
        };

        // Perform line search (if enabled) and update variables
        double alpha = 1.0;
        if (do_line_search) {
            double e_new = energy_func(alpha);
            double e0 = energy_func(0);
            std::cout << "e0: " << e0 << std::endl;

            while (e_new > e0 && alpha > 1e-10){ 
            //    //std::cout << "alpha: " << alpha << std::endl;
                alpha *= 0.5;
                e_new = energy_func(alpha);
            }
            std::cout << "!!!alpha: " << alpha << std::endl;
            if (alpha < 1e-10 && it == 0){
                std::cout << "line search failed" << std::endl;
                SelfAdjointEigenSolver<MatrixXd> es;
                es.compute(MatrixXd(A));
                std::cout << "The eigenvalues of A are: " << es.eigenvalues().transpose().head(10) << std::endl;
                //std::cout << delta_X << std::endl;
                //exit(1);
            }
        }
        x += alpha * delta_x;
        J += alpha * delta_J;

        //end = std::chrono::high_resolution_clock::now();
        //elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        //printf("Time measured 5: %.3f seconds.\n", elapsed.count() * 1e-9);

        
        output_file << (M * (x - x_hat) ).norm()<< ", " 
        << g_spacing.norm() << ", " 
        << (B.transpose()* (V_b_inv * dpsi_dJ)).norm() << ", "
        << (B.transpose()* (V_b_inv*H*V_b_inv*(J-Jx))).norm() << ", "
        << lambda.norm() << ", "
        << (alpha * delta_J).norm() << ", "
        << (alpha * delta_x).norm() << std::endl;

        // Write gradients for visualization
        for (int i = 0; i < n; i++) {
            grad_i(i, 0) = b_inertial(2*i);
            grad_i(i, 1) = b_inertial(2*i+1);
            
            grad_psi(i, 0) = b_psi(2*i);
            grad_psi(i, 1) = b_psi(2*i+1);
            
            grad_s(i, 0) = b_spacing(2*i);
            grad_s(i, 1) = b_spacing(2*i+1);
            
            grad_st(i, 0) = b_st(2*i);
            grad_st(i, 1) = b_st(2*i+1);
        }

        double residual = delta_x.norm() / n / dt;
        std::cout << "iteration: " << it << ", residual: " << residual << std::endl;

        if (residual < 2e-3 && converge_check) {
            std::cout << "converged" << std::endl;
            break;
        }
    }        

    // Turn x back into a field
    MatrixXd X_new = Eigen::Map<MatrixXd>(x.data(), 2, n).transpose();

    V = (X_new-X)/dt;
    X = X_new;

    for (int i = 0; i < 16; i++) {std::cout<< "i = " << i << ", " << J(i) << " " << Jx(i) << std::endl;}

    //std::cout << X << std::endl;
    // boundary detection
    /*
    if (bounds) {
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
    }
    */
    return;
}

template <>
void animate_implicit<3>(
    MatrixXd & X, 
    MatrixXd & V, 
    VectorXd & J,
    VectorXd & Jx,
    MatrixXi & N,
    Eigen::MatrixXd & grad_i,
    Eigen::MatrixXd & grad_psi,
    Eigen::MatrixXd & grad_s,
    Eigen::MatrixXd & grad_st,
    const Eigen::Matrix<double, 3, 1> & low_bound,
    const Eigen::Matrix<double, 3, 1> & up_bound,
    const int numofparticles,
    const int iters, 
    const double dt,
    const double kappa,
    const double k_st,
    const double k_s,
    const double st_threshold,
    const double rho_0,
    const double gravity,
    const bool fd_check,
    const bool bounds,
    const bool converge_check,
    const bool do_line_search,
    const bool smooth_mol,
    const bool psi_bool,
    const bool spacing_bool,
    const bool st_bool,
    const bool primal
    ){}
