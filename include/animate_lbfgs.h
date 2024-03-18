#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>
// use implicit method to iterate

template <int DIM>
void animate_lbfgs(
    Eigen::MatrixXd & X, 
    Eigen::MatrixXd & V, 
    Eigen::VectorXd & J, 
    Eigen::VectorXd & Jx, 
    Eigen::MatrixXi & N,
    Eigen::MatrixXd & grad_i,
    Eigen::MatrixXd & grad_psi,
    Eigen::MatrixXd & grad_s,
    Eigen::MatrixXd & grad_st,
    Eigen::SparseMatrix<double> & A,
    const Eigen::Matrix<double, DIM, 1> & low_bound,
    const Eigen::Matrix<double, DIM, 1> & up_bound,
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
    const bool fd_check = false,
    const bool bounds = true,
    const bool converge_check = false,
    const bool do_line_search = false,
    const bool smooth_mol = false,
    const bool psi_bool = true,
    const bool spacing_bool = true,
    const bool st_bool = true,
    const bool primal = true,
    const bool reset_A = true
);