#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>
#include "cubic_bspline.h"
#include "calculate_densities.h"
#include "pressure.h"

double mollifier_psi(
    double density,
    const double threshold
){
    double mollifier;
    if (density >  1.5 * threshold){
        mollifier = 1;
    }
    else if (density > 0.75 * threshold) {
        mollifier =  - 16 * density * density/(9 * threshold * threshold) + 16 * density/ (3 * threshold) - 3;
    }
    else{
        mollifier = 0;
    }
    return mollifier;
}

double molli_deriv_psi(
    double density,
    const double threshold
){
    double mollifier;
    if (density >  1.5 * threshold){
        mollifier = 0;
    }
    else if (density > 0.75 * threshold) {
        mollifier =  - 32 * density/(9 * threshold * threshold) + 16 / (3 * threshold);
    }
    else{
        mollifier = 0;
    }
    return mollifier;
}


template <>
double psi_energy<2>(
    const Eigen::VectorXd & x,
    const Eigen::VectorXd & J,
    const std::vector<std::vector<int>> neighbors,
    const double h,
    const double m,
    const double fac,
    const double kappa,
    const double threshold,
    const double rho_0
){
    int dim = 2;
    int n = x.size()/dim;
    double e_psi = 0;
    
    Eigen::VectorXd densities = calculate_densities<2>(x, neighbors, h, m, fac)/rho_0;

    for (int i = 0; i < n; i++){
        double mollifier = mollifier_psi(densities(i) * rho_0, threshold);
        e_psi += 0.5 * kappa * h * h * (densities(i) - 1) * (densities(i) - 1) * mollifier; 
    }
    
    return e_psi;
}


template <>
Eigen::VectorXd psi_gradient<2>(
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
){
    int dim = 2;
    int n = J.size();
    Eigen::VectorXd grad = Eigen::VectorXd::Zero(x.size());
    
    Eigen::VectorXd densities = calculate_densities<2>(x, neighbors, h, m, fac)/rho_0;
    Eigen::VectorXd dpsi = Eigen::VectorXd::Zero(x.size());

    if (primal){
        dpsi = kappa * h * h * -B.transpose() * (densities - Eigen::VectorXd::Ones(n));
    }
    else{
        dpsi = kappa * h * h * -B.transpose() * V_b_inv * (densities - Eigen::VectorXd::Ones(n));
    }
   

    // multiply by mollifier
    double mol = 0;
    for (int i = 0; i < n; i++){
        double mollifier = mollifier_psi(densities(i)*rho_0, threshold); 
        Eigen::VectorXd mol_deriv = molli_deriv_psi(densities(i)*rho_0, threshold) * -B.row(i) * rho_0;

        for (int d = 0; d < dim; d++){
            grad(i * dim + d) += mollifier * dpsi(i * dim + d);
        }
        grad += mol_deriv * 0.5 * kappa * h * h * (densities(i) - 1) * (densities(i) - 1);
    }
    return grad;
}

template <>
Eigen::SparseMatrix<double> psi_hessian<2>(
    const Eigen::SparseMatrix<double> & H,
    const Eigen::SparseMatrix<double> & B,
    const Eigen::SparseMatrix<double> & V_b_inv,
    const bool primal
){
    if (primal){
        return -B.transpose() * -B;
    }
    return B.transpose() * V_b_inv * H * V_b_inv * B; 
}