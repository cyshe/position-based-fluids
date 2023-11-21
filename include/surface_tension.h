#pragma once

#include <Eigen/Core>
#include "cubic_bspline.h"
#include "calculate_densities.h"

#include <iostream> // remove

template <int dim>
double surface_tension_energy(
    Eigen::VectorXd & x,
    std::vector<std::vector<int>> neighbors,
    const double h,
    const double m,
    const double fac,
    const double kappa,
    const double threshold
){
    int n = x.size() / dim;
    Eigen::VectorXd densities = calculate_densities<dim>(x, neighbors, h, m, fac);

    double energy = 0.0;
    for (int i = 0; i < n; i++){
        for (int j = 0; j < neighbors[i].size(); j++){
            double mollifier;
            if (densities(i) >  1.5 * threshold){
                mollifier = 0;
            }
            else if (densities(i) > 0.75 * threshold) {
                double x_div_eps = -(densities(i) - 1.5 *threshold) / ((1.5 - 0.75) * threshold);
                mollifier =  (2 - x_div_eps) * x_div_eps;
            }
            else{
                mollifier = 1;
            }
            energy += 0.5 * kappa * (densities(i) - densities(neighbors[i][j])) * (densities(i) - densities(neighbors[i][j])) * mollifier;
        }
    }
    return energy;
};


template <int dim>
Eigen::VectorXd surface_tension_gradient(
    Eigen::VectorXd & x,
    std::vector<std::vector<int>> neighbors,
    const double h,
    const double m,
    const double fac,
    const double kappa,
    const double threshold
){
    int n = x.size() / dim;

    Eigen::VectorXd grad = Eigen::VectorXd::Zero(x.size());
    Eigen::VectorXd densities = calculate_densities<dim>(x, neighbors, h, m, fac);
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(x.size(), n);

    // First compute density gradient
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < neighbors[i].size(); j++) {
            const auto& xi = x.template segment<dim>(dim * i);
            const auto& xj = x.template segment<dim>(dim * neighbors[i][j]);

            Eigen::Vector<double, dim*dim> density_grad = density_gradient<dim>(xi, xj, h, m, fac);
            B(dim*i + 0, i) += density_grad(0);
            B(dim*i + 1, i) += density_grad(1);
            B(dim*neighbors[i][j] + 0, i) += density_grad(2);
            B(dim*neighbors[i][j] + 1, i) += density_grad(3);
        }
    }

    // Now compute surface tension gradient
    for (int i = 0; i < n; i++){
        for (int j = 0; j < neighbors[i].size(); j++){
            double mol, mol_grad;

            if (densities(i) >  1.5 * threshold){
                mol = 0;
                mol_grad = 0;
            }
            else if (densities(i) > 0.75 * threshold) {
                double x_div_eps = -(densities(i) - 1.5 *threshold) / ((1.5 - 0.75) * threshold);
                mol =  (2 - x_div_eps) * x_div_eps;
                mol_grad = -32/9 * densities(i) / (threshold * threshold) + 16/3/ threshold;
            }
            else{
                mol = 1;
                mol_grad = 0;
            }
            
            grad += kappa * ((B.col(i) - B.col(neighbors[i][j])) * (densities(i) - densities(neighbors[i][j])) * mol
               + B.col(i) * mol_grad * (densities(i)-densities(j)) * (densities(i)-densities(j)) *0.5);
        }
    }

    return grad;
};

template <int dim>
Eigen::MatrixXd surface_tension_hessian(
    Eigen::VectorXd & x,
    std::vector<std::vector<int>> neighbors,
    const double h,
    const double m,
    const double fac,
    const double kappa,
    const double threshold
){
    
    Eigen::VectorXd grad = surface_tension_gradient<dim>(x, neighbors, h, m, fac, kappa, threshold);
    if(kappa == 0){
        return grad * grad.transpose();
    }
    
    return grad * grad.transpose()/kappa;
};

