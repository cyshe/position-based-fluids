#pragma once

#include <Eigen/Core>

template <int dim>
Eigen::Matrix<double, dim*dim, 1> density_gradient(
    const Eigen::Matrix<double, dim, 1>& xi,
    const Eigen::Matrix<double, dim, 1>& xj,
    const double h,
    const double m,
    const double fac);

template <int dim>
Eigen::VectorXd calculate_densities(
    const Eigen::VectorXd & x,
    const std::vector<std::vector<int>>& neighbors,
    const double h,
    const double m,
    const double fac
);