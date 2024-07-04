#pragma once
#include <igl/signed_distance.h>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <finitediff.hpp>
#include <cassert>
#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include "TinyAD/Scalar.hh"
#include "TinyAD/ScalarFunction.hh"
#include "TinyAD/VectorFunction.hh"

#include "animate_implicit.h"
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
    const double dq
    ){

    std::ofstream output_file("output.txt", std::ios::app);

    const int n = numofparticles;
    const double m = 1;
    const double vol = 1;//m/rho_0;
    const double fac = 10/7/M_PI; // bspline normalizing coefficient

    // Energy scales
    const double dt_sqr = dt * dt;
    const double kappa_dt_sqr = dt_sqr * kappa; 

    // Spacing energy params
    //const double dq = 0.5; // 0.8 - 1.0 seem to be reasonable values
    const double k_spacing = k_s;
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
        // Initialize new neighbor list
        std::vector<std::vector<int>> neighbors = find_neighbors_compact<2>(x, h);


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
                Vector4d density_grad = -density_gradient<2>(xi, xj, h, m, fac) / rho_0 / h;

                B_triplets.push_back(Triplet<double>(i, 2 * i, density_grad(0)));
                B_triplets.push_back(Triplet<double>(i, 2 * i + 1, density_grad(1)));
                B_triplets.push_back(Triplet<double>(i, 2 * neighbors[i][j], density_grad(2)));
                B_triplets.push_back(Triplet<double>(i, 2 * neighbors[i][j] + 1, density_grad(3)));
            }
        }
        B.setFromTriplets(B_triplets.begin(), B_triplets.end());
    

        // Create spacing energy function and evaluate gradient and hessian
        auto spacing_energy = spacing_energy_func<2>(x, elements, h/2, m, fac, W_dq, k_spacing);
        std::cout << "Evaluate gradient and hessian" << std::endl;
        auto [f, g_spacing, H_spacing] = spacing_energy.eval_with_hessian_proj(x);
        std::cout << "initial spacing energy: " <<  spacing_energy.eval(x) * dt_sqr << " " << f *dt_sqr <<  " gnorm: " << g_spacing.norm() << std::endl;
        
        

        // Assemble left and right hand sides of system
        
        A = M;
        if (psi_bool) {
            A += psi_hessian<2>(H, B, V_b_inv, primal) * dt_sqr;
        }
        if (spacing_bool) {
            A += H_spacing * dt_sqr;
        }
        if (st_bool) {
            A += surface_tension_hessian<2>(x, neighbors, h, m, fac, k_st, rho_0, st_threshold, smooth_mol, B).sparseView() * dt_sqr;
        }
        if (bounds_bool) {
            A += bounds_hessian<2>(x, low_bound, up_bound).sparseView() * dt_sqr;
        }
        
        VectorXd b_inertial = -M * (x - x_hat);
        VectorXd b_psi = VectorXd::Zero(2 * n);
        VectorXd b_spacing = VectorXd::Zero(2 * n);
        VectorXd b_st = VectorXd::Zero(2 * n);
        VectorXd b_bounds = VectorXd::Zero(2 * n);
        
        b = b_inertial;
        if (psi_bool) {
            b_psi = -psi_gradient<2>(x, J, neighbors, V_b_inv, B, h, m, fac, kappa, rho_0 * st_threshold, rho_0, primal);
            b += dt * dt * b_psi;
            //std::cout << "b_psi norm: " << b_psi.norm() << std::endl;
        }
        if (spacing_bool) {
            b_spacing = -g_spacing;
            b += dt * dt * b_spacing;
        }
        if (st_bool) {
            b_st = -surface_tension_gradient<2>(x, neighbors, h, m, fac, k_st, rho_0, st_threshold, smooth_mol, B);
            b += dt * dt * b_st;
        }
        if (bounds_bool) {
            b_bounds = -bounds_gradient<2>(x, low_bound, up_bound);
            b += dt * dt * b_bounds;
        }

        if (fd_check) {
            fd::AccuracyOrder accuracy = fd::SECOND;
            
            //const auto bound_func = [&](const Eigen::VectorXd& x) -> double {
            //    return bounds_energy<2>(x, low_bound, up_bound);
            //};

            //Eigen::VectorXd fg_bounds;
            //fd::finite_gradient(x, bound_func, fg_bounds, accuracy, 1.0e-7);
            //std::cout << "Bounds Gradient Error: " << (-b_bounds - fg_bounds).array().abs().maxCoeff() << std::endl;
            //std::cout << -b_bounds.head(10) << std::endl;
            //std::cout << fg_bounds.head(10) << std::endl;

            //Eigen::MatrixXd fH_bounds;
            //fd::finite_hessian(x, bound_func, fH_bounds, accuracy, 1.0e-5);
            //std::cout << "Bounds Hessian error: " << (fH_bounds - bounds_hessian<2>(x, low_bound, up_bound)).norm() << std::endl;
            //std::cout << fH_bounds.row(0).head(10) << std::endl;
            //std::cout << bounds_hessian<2>(x, low_bound, up_bound).row(0).head(10) << std::endl;

            const auto st_func = [&](const Eigen::VectorXd& x) -> double {
                return surface_tension_energy<2>(x, neighbors, h, m, fac, k_st, rho_0 * st_threshold, smooth_mol);
            };

            Eigen::VectorXd fg_st;
            fd::finite_gradient(x, st_func, fg_st, accuracy, 1.0e-7);
            std::cout << "Surface Tension Gradient Error: " << (-b_st - fg_st).array().abs().maxCoeff() << std::endl;
            std::cout << -b_st.head(10) << std::endl;
            std::cout << fg_st.head(10) << std::endl;


            //Eigen::MatrixXd fH_st;
            //fd::finite_hessian(x, st_func, fH_st, accuracy, 1.0e-5);
            //std::cout << "Surface Tension Hessian error: " << (fH_st - surface_tension_hessian<2>(x, neighbors, h, m, fac, k_st, rho_0, st_threshold, smooth_mol, B)).norm() << std::endl;
            //std::cout << fH_st.row(0).head(10) << std::endl;
            //std::cout << surface_tension_hessian<2>(x, neighbors, h, m, fac, k_st, rho_0, st_threshold, smooth_mol, B).row(0).head(10) << std::endl;

            //const auto scorr = [&](const Eigen::VectorXd& x) -> double {
            //    return spacing_energy_a<2>(x, neighbors, h/2, m, fac, W_dq, k_spacing); 
            //};

            //Eigen::VectorXd fg_spacing;
            //fd::finite_gradient(x, scorr, fg_spacing, accuracy, 1.0e-7);
            //std::cout << "Gradient Error: " << (g_spacing - fg_spacing).array().abs().maxCoeff() << std::endl;
            //for (int i = 0; i < 10; i++){
            //    std::cout << "fd: " << fg_spacing(i) << " " << g_spacing(i) << std::endl;
            //}

            //Eigen::MatrixXd fH_spacing;
            //fd::finite_hessian(x, scorr, fH_spacing, accuracy, 1.0e-5);
            //std::cout << "Hessian error: " << (fH_spacing - H_spacing).norm() << std::endl;
            //std::cout << "------------------" <<std::endl;
            //std::cout << fH_spacing.row(0).head(10) << std::endl; 
            //std::cout << H_spacing.row(0).head(10) << std::endl;
            
            // fd check for pressure gradients
            //const auto psi_func = [&](const Eigen::VectorXd& x) -> double {
            //    return psi_energy<2>(x, neighbors, h, m, fac, kappa, rho_0 * st_threshold, rho_0);
            //};

            //VectorXd grad_psi = -b_psi; 

            // Eigen::VectorXd fg_psi;
            // fd::finite_gradient(x, psi_func, fg_psi, accuracy, 1.0e-8);
            // std::cout << "Gradient Error: " << (grad_psi - fg_psi).array().abs().maxCoeff() << std::endl;
            // std::cout << "max value: " << (fg_psi).array().abs().maxCoeff() << " " << grad_psi.array().abs().maxCoeff() << std::endl;

            //std::cout << maxcoef() << std::endl; 

            // Eigen::MatrixXd fg_density;
            // const auto density_func = [&](const Eigen::VectorXd& x) {
            //     return calculate_densities<2>(x, neighbors, h, m, fac)/rho_0;
            // };

            // Eigen::MatrixXd density_jacobian = -B;

            // fd::finite_jacobian(x, density_func, fg_density, accuracy, 1.0e-8);
            // std::cout << "Density Gradient Error: " << (density_jacobian - fg_density).array().abs().maxCoeff() << std::endl;

            // fd check for bspline (this is 0 for now)
            //const auto cubic_bspline_func = [&](const Eigen::VectorXd& x) -> double {
            //    return cubic_bspline(x.norm(), fac);
            //};
            //Eigen::VectorXd fd_bspline;
            //fd::finite_gradient(x, cubic_bspline_func, fd_bspline, accuracy, 1.0e-8);
            //double bspline_grad = cubic_bspline_derivative(x.norm(), fac);
            //std::cout << "Cubic Bspline Gradient Error: " << (bspline_grad - fd_bspline.norm())<< std::endl;
            //Eigen::VectorXd densities = calculate_densities<2>(x, neighbors, h, m, fac);
            //double threshold = rho_0 * st_threshold;
            //double mol_k = 200;
            //// fd check mollifier
            //for (int i = 0; i < 100; i +=7 ){
            //    std::cout << "mol value: " << 1/ (1 + exp(mol_k * (densities(i) - threshold))) << std::endl;
            //    std::cout << "fd: " <<  (1/ (1 + exp(mol_k * (densities(i)+0.0001 - threshold))) - 1/ (1 + exp(mol_k * (densities(i)-0.0001 - threshold))))/0.0002<< std::endl;
            //    std::cout << "an: " <<  -mol_k * exp(mol_k * (densities(i) - threshold)) / ((1 + exp(mol_k * (densities(i) - threshold))) * (1 + exp(mol_k * (densities(i) - threshold))))  << std::endl;
            //}
        }

        // Solve for descent direction
        solver.compute(A);
        if (solver.info() != Success) {
            std::cout << "decomposition failed" << std::endl;
            exit(1);
        }
        VectorXd delta_x = solver.solve(b);

        if (!primal){
            lambda = V_b_inv * H * V_b_inv * (J - Jx + B * delta_x) 
               - V_b_inv * kappa_dt_sqr * (J - VectorXd::Ones(n));
        }

        VectorXd delta_J = -H_inv * (kappa_dt_sqr * (J - VectorXd::Ones(n)) + V_b * lambda);

        // Temporary variables for line search
        VectorXd x_new = VectorXd::Zero(n*2);
        VectorXd J_new = VectorXd::Zero(n);

        // Energy function for line search
        auto energy_func = [&](double alpha) {
            x_new = x + alpha * delta_x;
            //std::cout << "energy func x norm: " << x_new.norm() << std::endl;
            double energy = 0;

            neighbors = find_neighbors_compact<2>(x_new, h);
            // Inertial energy
            double e_i = 0.5 * (x_new - x_hat).transpose() * M * (x_new - x_hat);
            //std::cout << "\t e_i " << e_i << std::endl;
            energy += e_i;
            
            // Mixed potential energy
            if (psi_bool) {
                double e_psi = psi_energy<2>(x_new, neighbors, h, m, fac, kappa, rho_0 * st_threshold, rho_0) * dt_sqr;
                energy += e_psi;
                //std::cout << "\t e_psi " << e_psi << std::endl;
            }

            // Mixed constraint energy
            if (!primal){
                J_new = J + alpha * delta_J;
                double e_c = lambda.dot(J_new - (calculate_densities<2>(x_new, neighbors, h, m, fac) / rho_0)) * dt_sqr;
                energy += e_c;
                //std::cout << "\t e_c " << e_c << std::endl;
            }
            
            // Spacing energy
            if (spacing_bool) {
                double e_s = spacing_energy_a<2>(x_new, neighbors, h/2, m, fac, W_dq, k_spacing) * dt_sqr; 
                energy += e_s;
                //std::cout << "\t spacing: " << e_s << " " << std::endl;
            }

            // Surface tension energy
            if (st_bool) {
                double e_st = surface_tension_energy<2>(x_new, neighbors, h, m, fac, k_st,
                    rho_0 * st_threshold, smooth_mol) * dt_sqr;
                energy += e_st;
                //std::cout << "\t st: " << e_st << std::endl;
            }

            // Boundary energy
            if (bounds_bool) {
                double e_bound = bounds_energy<2>(x_new, low_bound, up_bound) *dt_sqr;
                energy += e_bound;
                //std::cout << "\t bounds: " << e_bound << std::endl;
            }
            //std::cout << "energy: " << energy << std::endl;
            return energy;
        };

        // Perform line search (if enabled) and update variables
        double alpha = 1.0;

        // ccd here
        x_new = x + alpha * delta_x;
        
        if (bounds_bool) {
            // if x_new is outside of bounds, calculate alpha such that it doesn't go outside
            MatrixXd::Index maxIndex[1];
            if(x_new.minCoeff() < low_bound(0)){
                for (int i = 0; i < n; i++){
                    if (low_bound(0) > x(2*i)){
                        alpha= std::min(alpha, (low_bound(0) - x(2*i)) / delta_x(2*i));
                    }
                }
                alpha = alpha * 0.95;
            }
            if(x_new.maxCoeff() > up_bound(0)){
                for (int i = 0; i < n; i++){
                    if (up_bound(0) < x(2*i)){
                        alpha= std::min(alpha, (up_bound(0) - x(2*i)) / delta_x(2*i));
                    }
                }
                alpha = alpha * 0.95;
            }
            if(x_new.minCoeff() < low_bound(1)){
                for (int i = 0; i < n; i++){
                    if (low_bound(1) > x(2*i+1)){
                        alpha= std::min(alpha, (low_bound(1) - x(2*i+1))/ delta_x(2*i+1));
                        std::cout << low_bound(1) << x(2*i+1) << delta_x(2*i+1) << (low_bound(1) - x(2*i+1))/ delta_x(2*i+1) << alpha << std::endl;
                    }
                }
                std::cout << "alpha (3): " << alpha << std::endl;
                alpha = alpha * 0.95;
            }
            
            if(x_new.maxCoeff() > up_bound(1)){
                for (int i = 0; i < n; i++){
                    if (up_bound(1) < x(2*i+1)){
                        alpha= std::min(alpha, (up_bound(1) - x(2*i+1)) / delta_x(2*i+1));
                    }
                }
            
                std::cout << "alpha (4): " << alpha << std::endl;
                alpha = alpha * 0.95;
            }
            
            //std::cout << "alpha (ccd): " << alpha << std::endl;
        } 
 
        double residual = delta_x.lpNorm<Eigen::Infinity>() / dt;
        std::cout << "iteration: " << it << ", residual: " << residual << std::endl;

        if (residual < 5e-7 && converge_check) {
            std::cout << "converged" << std::endl;
            break;
        }


        if (do_line_search) {
            double e_new = energy_func(alpha);
            double e0 = energy_func(0);

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
                // std::cout << delta_X << std::endl;
                exit(1);
            }
            std::cout << "e0: " << e0 << " enew: " << e_new << std::endl;
        }
        x += alpha * delta_x;
        J += alpha * delta_J;



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

        if (it == iters - 1){
            std::cout << "not converged" << std::endl;
            std::cout << "residual: " << residual << std::endl;
            exit(1);
        } 
    }        

    // Turn x back into a field
    MatrixXd X_new = Eigen::Map<MatrixXd>(x.data(), 2, n).transpose();

    V = (X_new-X)/dt;
    X = X_new;

    // for (int i = 0; i < 16; i++) {std::cout<< "i = " << i << ", " << J(i) << " " << Jx(i) << std::endl;}

    //std::cout << X << std::endl;
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
    const double dq
    ){}
