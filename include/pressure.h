#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>
#include "cubic_bspline.h"

double mollifier_psi(
    double density,
    const double threshold
);

double molli_deriv_psi(
    double density,
    const double threshold
);


template <int dim>
double psi_energy(
    const Eigen::VectorXd & x,
    const std::vector<std::vector<int>> neighbors,
    const double h,
    const double m,
    const double fac,
    const double kappa,
    const double threshold,
    const double rho_0
);


template <int dim>
Eigen::VectorXd psi_gradient(
    const Eigen::VectorXd & x,
    const Eigen::VectorXd & J,
    const std::vector<std::vector<int>> neighbors,
    const Eigen::SparseMatrix<double> & V_b_inv,
    const Eigen::SparseMatrix<double> & B,
    const double h,
    const double m,
    const double fac,
    const double kappa,
    const double threshold,
    const double rho_0,
    const bool primal
);

template <int dim>
Eigen::SparseMatrix<double> psi_hessian(
    const Eigen::SparseMatrix<double> & H,
    const Eigen::SparseMatrix<double> & B,
    const Eigen::SparseMatrix<double> & V_b_inv,
    const bool primal
);