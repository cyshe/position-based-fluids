#pragma once
#include <igl/signed_distance.h>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <finitediff.hpp>
#include <cassert>
#include <iostream>
#include <fstream>
#include <deque>
#include <vector>
#include "TinyAD/Scalar.hh"
#include "TinyAD/ScalarFunction.hh"
#include "TinyAD/VectorFunction.hh"

#include "animate_lbfgs.h"
#include "calculate_lambda.h"
#include "surface_tension.h"
#include "boundary_conditions.h"
#include "spacing_energy.h"
#include "pressure.h"
#include "find_neighbors_compact.h"
#include "find_neighbors_brute_force.h"
#include "calculate_densities.h"
#include "cubic_bspline.h"
#include "line_search.h"

#define _USE_MATH_DEFINES
#include <math.h>


using namespace Eigen;

namespace {
}

template <>
void animate_lbfgs<2>(
    MatrixXd & X, 
    MatrixXd & V,
    VectorXd & J,
    VectorXd & Jx,
    MatrixXi & N,
    Eigen::MatrixXd & grad_i,
    Eigen::MatrixXd & grad_psi,
    Eigen::MatrixXd & grad_s,
    Eigen::MatrixXd & grad_st,
    Eigen::SparseMatrix<double> & A,
    const Eigen::Matrix<double, 2, 1> & low_bound,
    const Eigen::Matrix<double, 2, 1> & up_bound,
    const int numofparticles,
    const int iters, 
    const double dt,
    const double kappa,
    const double k_st,
    const double k_s,
    const double h, //kernel radius for calculating density
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
    const bool primal,
    const bool reset_A
    ){

    std::ofstream output_file("output.txt", std::ios::app);

    const int n = numofparticles;
    const double m = 1;
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
    SparseMatrix<double> M, B, H, V_b, V_b_inv, H_inv;
    M.resize(2 * n, 2 * n);
    B.resize(n, 2 * n);
    H.resize(n, n);
    H_inv.resize(n, n);
    V_b.resize(n, n);
    V_b_inv.resize(n, n);

    // Vectors
    VectorXd b = VectorXd::Zero(2 * n);

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
    std::deque<VectorXd> prev_Xs;
    std::deque<VectorXd> prev_grads;

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

    //L-bfgs
    
    // q = -gradient
    
    // for i = k-1, ..., k-l
    //      s = x_k+1 - x_k
    //      t = grad_k+1 - grad_k
    //      rho = tr(t.T * s)
    //      sigma = tr(s.t * q)/rho
    //      q = q + sigma * t
    // r = A^-1 * q
    // for i = k-m, ..., k-1
    //      eta = tr(t.T * r)/rho
    //      r = r - (sigma - eta) * s
    // d = -r
    // x = x + alpha * d
    // Lbfgs solver
    // Create spacing energy function and evaluate gradient and hessian
    // Assemble B matrix -- jacobian w.r.t of the J - J(x) constraint

    std::vector<std::vector<int>> neighbors = find_neighbors_compact<2>(x, h);

    // Create list of neighbor pairs (as elements for TinyAD)
    std::vector<Eigen::Vector2i> elements;
    for (int i = 0; i < n; i++){
    //if (neighbors[i].size() <= 7) std::cout << "i = " << i << ", neighbors = " << neighbors[i].size() << std::endl; 
        for (int j = 0; j < neighbors[i].size(); j++){
            elements.push_back(Eigen::Vector2i(i,neighbors[i][j]));
            }
    }

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

    auto spacing_energy = spacing_energy_func<2>(x, elements, 0.08, m, fac, W_dq, k_spacing);
    std::cout << "Evaluate gradient and hessian spacing" << std::endl;
    auto [f, g_spacing, H_spacing] = spacing_energy.eval_with_hessian_proj(x);
 
    if (reset_A){
        A = M;
        if (psi_bool) {
            A += psi_hessian<2>(H, B, V_b_inv, primal) *dt_sqr;
        }
        if (spacing_bool) {
            A += H_spacing * dt *dt;
        }
        if (st_bool) {
            A += surface_tension_hessian<2>(x, neighbors, h, m, fac, k_st, rho_0 * st_threshold, smooth_mol).sparseView() *dt_sqr;
        }
        if (bounds_bool) {
            A += bounds_hessian<2>(x, low_bound, up_bound).sparseView() *dt_sqr;
        }
    }
    SimplicialLDLT<SparseMatrix<double>> solver;
    solver.compute(A);
    if (solver.info() != Success) {
        std::cout << "decomposition failed" << std::endl;
        exit(1);
    }

    for (int it = 0; it < iters; it++){
        // Calculate neighbors (as list of indices
        neighbors = find_neighbors_compact<2>(x, h);

        // Create list of neighbor pairs (as elements for TinyAD)
        elements.clear();
        for (int i = 0; i < n; i++){
        //if (neighbors[i].size() <= 7) std::cout << "i = " << i << ", neighbors = " << neighbors[i].size() << std::endl; 
            for (int j = 0; j < neighbors[i].size(); j++){
                elements.push_back(Eigen::Vector2i(i,neighbors[i][j]));
            }
        }

        // Create spacing energy function and evaluate gradient and hessian
        auto spacing_energy = spacing_energy_func<2>(x, elements, 0.08, m, fac, W_dq, k_spacing);
        std::cout << "Evaluate gradient and hessian spacing" << std::endl;
        auto [f, g_spacing, H_spacing] = spacing_energy.eval_with_hessian_proj(x);


        VectorXd b_inertial = -M * (x - x_hat);
        VectorXd b_psi = VectorXd::Zero(2 * n);
        VectorXd b_spacing = VectorXd::Zero(2 * n);
        VectorXd b_st = VectorXd::Zero(2 * n);
        VectorXd b_bounds = VectorXd::Zero(2 * n);

        std::cout << "Calculate gradient" << std::endl;
        b = b_inertial;
        if (psi_bool) {
            b_psi = psi_gradient<2>(x, J, neighbors, V_b_inv, B, dt, m, fac, kappa, rho_0 * st_threshold, rho_0, primal);
            b += dt * dt * b_psi;
        }
        if (spacing_bool) {
            b_spacing = g_spacing;
            b += dt * dt * b_spacing;
        }
        if (st_bool) {
            b_st = surface_tension_gradient<2>(x, neighbors, h, m, fac, k_st, rho_0 * st_threshold, smooth_mol);
            b += dt * dt * b_st;
        }
        if (bounds_bool) {
            b_bounds = bounds_gradient<2>(x, low_bound, up_bound);
            b += dt * dt * b_bounds;
        }

        VectorXd q = -b;
        VectorXd rho = VectorXd::Zero(iters);
        VectorXd s = VectorXd::Zero(2 * n);
        VectorXd t = VectorXd::Zero(2 * n);
        VectorXd sigma = VectorXd::Zero(iters);

        std::deque<VectorXd>::iterator i = prev_Xs.begin();
        std::deque<VectorXd>::iterator j = prev_grads.begin();
        std::cout << "L-BFGS" << std::endl;
        //queue
        //present ------ past
        //1 2 3 4 5 6 7

        //rho and other vectors
        //present ------ past
        //1 2 3 4 5 6 7

        //loop 1
        //present ------ past
        //i = 1 2 3 4 5 6 7 

        //loop 2
        //past ------- present
        //i = 7 6 5 4 3 2 1

        std::cout << "gradient norm " << b.norm() << std::endl; 
        while (i != prev_Xs.end()){
            std::cout << (*i).norm() << std::endl;
            s.setZero();
            t.setZero();
            if (i == prev_Xs.begin()){
                s = x - (*i);
                t = b - (*j);
            }
            else{
                // new x - old x
                s = *(i-1) - *i;
                // new grad - old grad
                t = *(j-1) - *j;
            }

            std::cout << "s: " << s.norm() << std::endl;
            std::cout << "t: " << t.norm() << std::endl;
            
            rho(std::distance(prev_Xs.begin(), i)) = (t.transpose() * s).trace();
            if (abs(rho(std::distance(prev_Xs.begin(), i))) < 1e-10){
                sigma(std::distance(prev_Xs.begin(), i)) = 0;
                std::cout << "rho continue"<< std::endl;
                i ++;
                j ++;
                continue;
            }
            sigma(std::distance(prev_Xs.begin(), i)) = (s.transpose() * q).trace() / rho(std::distance(prev_Xs.begin(), i));
            std::cout << "sigma: " << sigma(std::distance(prev_Xs.begin(), i)) << std::endl;
            std::cout << "t" << t.norm() << std::endl;
            std::cout << "q: " << q.norm() << std::endl;
            std::cout << "st: "<< (sigma(std::distance(prev_Xs.begin(), i))*t).norm() << std::endl;
            q = q - (sigma(std::distance(prev_Xs.begin(), i)) * t);
            std::cout << "q: " << q.norm() << std::endl;
            i ++; 
            j ++;
        }        
        

        std::cout << "A^-1 * q" << std::endl;
        VectorXd delta_x = solver.solve(q);

        i = prev_Xs.end() -1;
        j = prev_grads.end() -1;
        while (i != prev_Xs.begin()-1){
            if (abs(rho(std::distance(prev_Xs.begin(), i))) < 1e-10) {
                std::cout << "rho continue :" << rho(std::distance(prev_Xs.begin(), i))<< std::endl;
                i --;
                j --;
                continue;
            }
            s.setZero();
            t.setZero();
            if (i == prev_Xs.begin()){
                s = x - *i;
                t = b - *j;
            }
            else{
                s = *(i-1) - *i;
                t = *(j-1) - *j;
            }

            double eta = (t.transpose() * delta_x).trace() / rho(std::distance(prev_Xs.begin(), i));
            delta_x = delta_x + (sigma(std::distance(prev_Xs.begin(), i)) - eta) * s;
            i --;
            j --;
        }
        prev_Xs.push_front(x);
        prev_grads.push_front(b);
        
        std::cout << "delta_x: " << delta_x.norm() << std::endl;


        // Temporary variables for line search
        VectorXd x_new = VectorXd::Zero(n*2);

        // Energy function for line search
        auto energy_func = [&](double alpha) {
            x_new = x + alpha * delta_x;

            neighbors = find_neighbors_compact<2>(x_new, h);
            // Inertial energy
            double e_i = 0.5 * (x_new - x_hat).transpose() * M * (x_new - x_hat);

            // Mixed potential energy
            double e_psi = psi_energy<2>(x_new, neighbors, dt, m, fac, kappa, rho_0 * st_threshold, rho_0) * dt_sqr;
            
            // Spacing energy
            double e_s = spacing_energy.eval(x_new) * dt_sqr;

            // Surface tension energy
            double e_st = surface_tension_energy<2>(x_new, neighbors, h, m, fac, k_st,
                rho_0 * st_threshold, smooth_mol) * dt_sqr;

            // Boundary energy
            double e_bound = bounds_energy<2>(x_new, low_bound, up_bound) *dt_sqr;

            return e_i + e_psi + e_s + e_st + e_bound;
        };

        // Perform line search (if enabled) and update variables
        double alpha = 1;
        if (do_line_search) {
            double e_new = energy_func(alpha);
            double e0 = energy_func(0);
            std::cout << "e0: " << e0 << std::endl;

            while (e_new > e0 && alpha > 1e-10){ 
            //    //std::cout << "alpha: " << alpha << std::endl;
                alpha *= 0.5;
                e_new = energy_func(alpha);
                std::cout << "e_new: " << e_new << std::endl;
            }
            std::cout << "!!!alpha: " << alpha << std::endl;
            //if (alpha < 1e-10 && it == 0){
            //    std::cout << "line search failed" << std::endl;
            //    //std::cout << delta_X << std::endl;
            //    //exit(1);
            //}
            //
        }
        std::cout << "e after line search: " << energy_func(alpha) << std::endl;
        x += alpha * delta_x;
        
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
        double residual = delta_x.lpNorm<Eigen::Infinity>() / dt;
        std::cout << "residual: " << residual << std::endl;
        std::cout << "iteration: " << it << ", residual: " << residual << std::endl;
        if (residual < 2e-3 && converge_check) {
            std::cout << "converged" << std::endl;
        }

        
    }
    // Turn x back into a field
    MatrixXd X_new = Eigen::Map<MatrixXd>(x.data(), 2, n).transpose();

    V = (X_new-X)/dt;
    X = X_new;

    return;
}

template <>
void animate_lbfgs<3>(
    MatrixXd & X, 
    MatrixXd & V, 
    VectorXd & J,
    VectorXd & Jx,
    MatrixXi & N,
    Eigen::MatrixXd & grad_i,
    Eigen::MatrixXd & grad_psi,
    Eigen::MatrixXd & grad_s,
    Eigen::MatrixXd & grad_st,
    Eigen::SparseMatrix<double> & A,
    const Eigen::Matrix<double, 3, 1> & low_bound,
    const Eigen::Matrix<double, 3, 1> & up_bound,
    const int numofparticles,
    const int iters, 
    const double dt,
    const double kappa,
    const double k_st,
    const double k_s,
    const double h,
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
    const bool primal,
    const bool reset_A
    ){}