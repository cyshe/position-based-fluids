#pragma once

#include <Eigen/Core>

template <int DIM>
double surface_tension_energy(
    Eigen::VectorXd & x,
    std::vector<std::vector<int>> neighbors,
    const double h,
    const double m,
    const double fac,
    const double kappa
){
    int n = x.size() / DIM;
    Eigen::VectorXd densities = Eigen::VectorXd::Zero(x.size() / DIM);
    densities = calculate_densities<DIM>(x, h, m, fac);
    
    double energy = 0.0;
    for (int i = 0; i < n; i++){
        for (int j = 0; j < neighbors[i].size(); j++){
            energy += 0.5 * kappa * (densities(i) - densities(neighbors[i][j])) * (densities(i) - densities(neighbors[i][j]));
        }
    }
    return energy;
};


template <int DIM>
Eigen::VectorXd surface_tension_gradient(
    Eigen::VectorXd & x,
    std::vector<std::vector<int>> neighbors,
    const double h,
    const double m,
    const double fac,
    const double kappa
){
    int n = x.size() / DIM;
    Eigen::VectorXd grad = Eigen::VectorXd::Zero(x.size());

    Eigen::VectorXd densities = Eigen::VectorXd::Zero(x.size() / DIM);
    
    densities = calculate_densities<DIM>(x, h, m, fac);

    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(x.size(), n);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            auto& xi = x.segment<2>(2 * i);
            auto& xj = x.segment<2>(2 * j);
            Eigen::Vector2d diff = xj - xi;
            double r = diff.norm() / h;
            double deriv = cubic_bspline_derivative(r, m*fac) ;
            if (deriv != 0.0) {
                // dci_dxj
                // Negating because constraint is c(J,x) = J - J(x) 
                Eigen::Vector2d dc_dx = -(deriv * diff / r / h);
                B(DIM*j, i) = dc_dx(0);
                B(DIM*j+1, i) = dc_dx(1);
            }
        }
    }

    for (int i = 0; i < n; i++){
        for (int j = 0; j < neighbors[i].size(); j++){
            grad += kappa * (B.col(i) - B.col(neighbors[i][j])) * (densities(i) - densities(neighbors[i][j]));
        }
    }
    
    return grad;
};

template <int DIM>
Eigen::MatrixXd surface_tension_hessian(
    Eigen::VectorXd & x,
    std::vector<std::vector<int>> neighbors,
    const double h,
    const double m,
    const double fac,
    const double kappa
){
    
    Eigen::VectorXd grad = surface_tension_gradient<DIM>(x, neighbors, h, m, fac, kappa);
    if(kappa == 0){
        return grad * grad.transpose();
    }
    
    return grad * grad.transpose()/kappa;
};

template <int DIM>
Eigen::VectorXd calculate_densities(
    const Eigen::VectorXd & x,
    const double h,
    const double m,
    const double fac
){
    int n = x.size() / DIM;
    Eigen::VectorXd densities = Eigen::VectorXd::Zero(x.size() / DIM);
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            auto& xi = x.segment<2>(2 * i);
            auto& xj = x.segment<2>(2 * j);
            double r = (xj - xi).norm()/h;
            densities(i) += cubic_bspline(r, m*fac);
        }
    }
    return densities;
};