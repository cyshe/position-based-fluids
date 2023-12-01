#pragma once

#include <Eigen/Core>
// use implicit method to iterate

template <int DIM>
void animate_implicit(
    Eigen::MatrixXd & X, 
    Eigen::MatrixXd & V, 
    Eigen::VectorXd & J, 
    Eigen::VectorXd & Jx, 
    Eigen::MatrixXi & N,
    Eigen::MatrixXd & grad_i,
    Eigen::MatrixXd & grad_psi,
    Eigen::MatrixXd & grad_s,
    Eigen::MatrixXd & grad_st,
    const Eigen::Matrix<double, DIM, 1> & low_bound,
    const Eigen::Matrix<double, DIM, 1> & up_bound,
    const int numofparticles,
    const int iters, 
    const double dt,
    const double kappa,
    const double k_st,
    const double k_s,
    const double st_threshold,
    const double rho_0,
    const double gravity,
    const bool fd_check = false,
    const bool bounds = true,
    const bool converge_check = false,
    const bool do_line_search = false,
    const bool smooth_mol = false
);

// Computes derivative of norm of a vector
// d/dx (||x||) = x / r  where r = ||x||
template <int DIM>
Eigen::Matrix<double,DIM,1> norm_derivative(
    const Eigen::Matrix<double,DIM,1>& x, double r,
    double eps = 1e-8) {
  return x / (r+eps);
}

// Computes hessian of norm of a vector
// d^2/dx^2 (||x||) = I / r - xx^T/r^3  
//  where r = ||x||
template <int DIM>
Eigen::Matrix<double,DIM,DIM> norm_hessian(
    const Eigen::Matrix<double,DIM,1>& x, double r, double eps=1e-8) {
  Eigen::Matrix<double,DIM,DIM> I = Eigen::Matrix<double,DIM,DIM>::Identity();
  r = r + eps;
  return I / r - x * x.transpose() / (r * r * r);
}
