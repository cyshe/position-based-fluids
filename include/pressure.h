#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>
#include "cubic_bspline.h"

/*
template <int dim>
auto psi_energy_func(
    Eigen::VectorXd & x,
    std::vector<std::vector<int>> elements,
    const double h,
    const double m,
    const double fac,
    const double kappa,
    const double threshold
){
    int n = x.size() / dim;
    auto func = TinyAD::scalar_function<dim>(TinyAD::range(n));

    func.template add_elements<dim>(TinyAD::range(elements.size()),
        [&] (auto& element) -> TINYAD_SCALAR_TYPE(element) {
            using T = TINYAD_SCALAR_TYPE(element);
            using Vec = Eigen::Matrix<T, dim, 1>;
            int idx = element.handle;
            const auto& xi = element.variables(elements[idx](0));
            const auto& xj = element.variables(elements[idx](1));
            T r = (xj - xi).norm() / h;
            T Wij = cubic_bspline(r, T(m*fac));
            return 0.5 * kappa * h * h *  Wij - 1 * Wij;
    });
    return func;

}*/





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
    int n = x.size()/dim;
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
        mollifier = 1;
        e_psi += 0.5 * kappa * h * h * (densities(i) - 1) * (densities(i) - 1) * mollifier; 
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
    VectorXd dpsi_dJ = kappa * (densities - VectorXd::Ones(n));

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
        mollifier = 1;
        grad(i) = mollifier * dpsi_dJ(i);
    }
    return grad;
};

template <int dim>
Eigen::SparseMatrix<double> psi_hessian(
    Eigen::SparseMatrix<double> & H
){
    return H;
};

