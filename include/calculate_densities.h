#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>

template <int dim>
Eigen::Matrix<double, dim*dim, 1> density_gradient(
    const Eigen::Matrix<double, dim, 1>& xi,
    const Eigen::Matrix<double, dim, 1>& xj,
    const double h,
    const double m,
    const double fac);

template <int dim>
Eigen::Matrix<double, dim, 1> density_gradient_element(
    const Eigen::Matrix<double, dim, 1>& xi,
    const Eigen::Matrix<double, dim, 1>& xj,
    const double h,
    const double m,
    const double fac);

template <int dim>
Eigen::Matrix<double, dim, dim> density_hessian(
    Eigen::VectorXd & x,
    std::vector<std::vector<int>> neighbors,
    const int i,
    const int k,
    const int l,
    const double h,
    const double m,
    const double fac,
    const double rho_0,
    const Eigen::SparseMatrix<double> & B_sparse 
);

template <int dim>
Eigen::VectorXd calculate_densities(
    const Eigen::VectorXd & x,
    const std::vector<std::vector<int>>& neighbors,
    const double h,
    const double m,
    const double fac
);

template <int dim>
Eigen::Matrix<double, 1, 1> calculate_density_stencil(
    const Eigen::Matrix<double, dim*2, 1>& x,
    const double h,
    const double m,
    const double fac);