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