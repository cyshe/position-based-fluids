#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>
#include "cubic_bspline.h"
#include <iostream> // remove

template <int dim>
double psi_energy(
    Eigen::VectorXd & x,
    Eigen::VectorXd & J,
    std::vector<std::vector<int>> neighbors,
    const double h,
    const double m,
    const double fac,
    const double kappa,
    const double threshold
){
    int n = J.size();
    double e_psi = 0;

    Eigen::VectorXd densities = calculate_densities<dim>(x, neighbors, h, m, fac);

    for (int i = 0; i < n; i++){
        double mollifier;
        if (densities(i) >  1.5 * threshold){
            mollifier = 1;
        }
        else if (densities(i) > 0.75 * threshold) {
            mollifier =  - 16 * densities(i) * densities(i)/(9 * threshold * threshold) + 16 * densities(i)/ (3 * threshold) - 3;
        }
        else{
            mollifier = 0;
        }
        e_psi += 0.5 * kappa * h * h * (J(i) - 1) * (J(i) - 1) * mollifier; 
    }
    

    return e_psi;
};


template <int dim>
Eigen::VectorXd psi_gradient(
    Eigen::VectorXd & x,
    Eigen::VectorXd & J,
    std::vector<std::vector<int>> neighbors,
    const double h,
    const double m,
    const double fac,
    const double kappa,
    const double threshold
){
    int n = J.size();
    Eigen::VectorXd grad = Eigen::VectorXd::Zero(n);

    Eigen::VectorXd densities = calculate_densities<dim>(x, neighbors, h, m, fac);
    VectorXd dpsi_dJ = kappa * (J - VectorXd::Ones(n));

    // multiply by mollifier
    double mol = 0;
    for (int i = 0; i < n; i++){
        double mollifier;
        if (densities(i) >  1.5 * threshold){
                mollifier = 1;
            }
            else if (densities(i) > 0.75 * threshold) {
                mollifier =  - 16 * densities(i) * densities(i)/(9 * threshold *  threshold) + 16 * densities(i)/ (3 * threshold) - 3;
            }
            else{
                mollifier = 0;
        }
        mol = 0;
        grad(i) *= mol;
    }
    return grad;
};

template <int dim>
Eigen::SparseMatrix<double> psi_hessian(
    Eigen::SparseMatrix<double> & B,
    Eigen::SparseMatrix<double> & V_b_inv,
    Eigen::SparseMatrix<double> & H
){
    return B.transpose() * (V_b_inv * H * V_b_inv) * B;
};

