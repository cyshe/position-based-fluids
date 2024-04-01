#pragma once

#include <Eigen/Core>
#include "TinyAD/Scalar.hh"
#include "TinyAD/ScalarFunction.hh"
#include "TinyAD/VectorFunction.hh"

template <int dim>
auto spacing_energy_func(
    const Eigen::VectorXd & x,
    const std::vector<Eigen::Vector2i>& elements,
    const double h,
    const double m,
    const double fac,
    const double W_dq,
    const double kappa
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
            return 0.5 * kappa * (Wij - W_dq) * (Wij - W_dq);
    });
    return func;
}

// spacing energy
template <int dim>
double spacing_energy_a(
    const Eigen::VectorXd & x,
    std::vector<std::vector<int>> neighbors,
    const double h,
    const double m,
    const double fac,
    const double W_dq,
    const double kappa
){
    double energy = 0.0;
    int idx = 0;
    double r = 1.0;
    double Wij = 0.0;
    
    for (int i = 0; i < neighbors.size(); i++){
        for (int j = 0; j < neighbors[i].size(); j++){
            idx = neighbors[i][j];
            r = (x.segment<dim>(dim*i) - x.segment<dim>(dim*idx)).norm() / h;
            Wij = cubic_bspline(r, m*fac);
            energy += 0.5 * kappa * (Wij - W_dq) * (Wij - W_dq);
        }
    }
    return energy;
}



// spacing gradient 
template <int dim>
Eigen::VectorXd spacing_gradient(
    const Eigen::VectorXd & x,
    std::vector<std::vector<int>> neighbors,
    const double h,
    const double m,
    const double fac,
    const double W_dq,
    const double kappa
){
    int n = x.size() / 2;
    Eigen::VectorXd grad = Eigen::VectorXd::Zero(n * 2);
    int idx = 0;
    double r = 0.0;
    double Wij = 0.0;
    double dWij = 0.0;
    Eigen::Vector2d xi, xj;
    for (int i = 0; i < neighbors.size(); i++){
        for (int j = 0; j < neighbors[i].size(); j++){
            idx = neighbors[i][j];
            xi = x.segment<dim>(2*i);
            xj = x.segment<dim>(2*idx);
            r = (xi - xj).norm() / h;
            Wij = cubic_bspline(r, m*fac);
            dWij = cubic_bspline_derivative(r, m*fac);
            grad.segment<dim>(2*i) += kappa * (Wij - W_dq) * dWij * (xi - xj) / (h * r);
        }
    }
    return grad;
}

// spacing hessian
template <int dim>
Eigen::MatrixXd spacing_hessian(
    const Eigen::VectorXd & x,
    std::vector<std::vector<int>> neighbors,
    const double h,
    const double m,
    const double fac,
    const double W_dq,
    const double kappa
){
    int n = x.size() / 2;
    Eigen::MatrixXd hessian = Eigen::MatrixXd::Zero(n * 2, n * 2);
    int idx = 0;
    double r = 0.0;
    double Wij = 0.0;
    double dWij = 0.0;
    double ddWij = 0.0;
    Eigen::Vector2d xi, xj;
    for (int i = 0; i < neighbors.size(); i++){
        for (int j = 0; j < neighbors[i].size(); j++){
            idx = neighbors[i][j];
            xi = x.segment<dim>(2*i);
            xj = x.segment<dim>(2*idx);
            r = (xi - xj).norm() / h;
            Wij = cubic_bspline(r, m*fac);
            dWij = cubic_bspline_derivative(r, m*fac);
            ddWij = cubic_bspline_second_derivative(r, m*fac);
            hessian.block<dim, dim>(2*i, 2*i) += kappa * dWij * dWij / (h * r) * Eigen::Matrix2d::Identity();
            hessian.block<dim, dim>(2*i, 2*idx) -= kappa * dWij * dWij / (h * r) * Eigen::Matrix2d::Identity();
            hessian.block<dim, dim>(2*i, 2*i) += kappa * (Wij - W_dq) * ddWij * (xi - xj) * (xi - xj).transpose() / (h * r * r);
            hessian.block<dim, dim>(2*i, 2*idx) -= kappa * (Wij - W_dq) * ddWij * (xi - xj) * (xi - xj).transpose() / (h * r * r);
        }
    }
    return hessian;
}
