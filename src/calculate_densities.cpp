#include "calculate_densities.h"
#include "cubic_bspline.h"
#include <Eigen/Core>
#include <iostream>

using namespace Eigen;

template <int dim>
Matrix<double, dim*dim, 1> density_gradient(
    const Matrix<double, dim, 1>& xi,
    const Matrix<double, dim, 1>& xj,
    const double h,
    const double m,
    const double fac)
{
    Matrix<double, dim*dim, 1> grad;
    Matrix<double, dim, 1> xij = xj - xi;
    double r = xij.norm() / h;
    double deriv = cubic_bspline_derivative(r, m*fac);

    Vector2d dphi_dx = deriv * xij / r / h; // direction
    grad.template segment<dim>(0) = -dphi_dx;
    grad.template segment<dim>(dim) = dphi_dx;
    return grad;
}

template <int dim>
Eigen::VectorXd calculate_densities(
    const Eigen::VectorXd & x,
    const std::vector<std::vector<int>>& neighbors,
    const double h,
    const double m,
    const double fac
)
{
    int n = x.size() / dim;
    assert(neighbors.size() == n);

    double initial_density = cubic_bspline(0.0, m*fac);

    // Eigen::VectorXd densities = Eigen::VectorXd::Constant(n, initial_density);
    Eigen::VectorXd densities = Eigen::VectorXd::Zero(n);
    for (int i = 0; i < n; i++){
        for (int j = 0; j < neighbors[i].size(); j++){
            const auto& xi = x.template segment<dim>(dim * i);
            const auto& xj = x.template segment<dim>(dim * neighbors[i][j]);
            double r = (xj - xi).norm()/h;
            densities(i) += cubic_bspline(r, m*fac);
        }
        densities(i) += m * fac;
    }
    
    return densities;
};


template <int dim>
Matrix<double, dim, dim> density_hessian(
    const Matrix<double, dim, 1>& xi,
    const Matrix<double, dim, 1>& xj,
    const double h,
    const double m,
    const double fac
){
    Matrix<double, dim, dim> hess;
    Matrix<double, dim, 1> xij = xj - xi;
    double r = xij.norm() / h;

    if (r == 0){
        return Matrix<double, dim, dim>::Zero();
    }

    Matrix<double, dim, dim> drho_dx2;
    double x1 = xi(0);
    double y1 = xi(1);
    double x2 = xj(0);
    double y2 = xj(1);


    drho_dx2(0, 0) = (pow(x1*2.0-x2*2.0,2.0)*1.0/pow(pow(x1-x2,2.0)+pow(y1-y2,2.0),3.0/2.0))/4.0-1.0/sqrt(pow(x1-x2,2.0)+pow(y1-y2,2.0)); ; 
    drho_dx2(1, 0) = ((x1*2.0-x2*2.0)*(y1*2.0-y2*2.0)*1.0/pow(pow(x1-x2,2.0)+pow(y1-y2,2.0),3.0/2.0))/4.0;;
    drho_dx2(0, 1) = ((x1*2.0-x2*2.0)*(y1*2.0-y2*2.0)*1.0/pow(pow(x1-x2,2.0)+pow(y1-y2,2.0),3.0/2.0))/4.0;;
    drho_dx2(1, 1) = (pow(y1*2.0-y2*2.0,2.0)*1.0/pow(pow(x1-x2,2.0)+pow(y1-y2,2.0),3.0/2.0))/4.0-1.0/sqrt(pow(x1-x2,2.0)+pow(y1-y2,2.0));
    hess = (xij/r/h/h) * -(xij/r/h/h).transpose() * cubic_bspline_hessian(r, m*fac) + cubic_bspline_derivative(r, m*fac) * drho_dx2;

    
   if ((0.0 < h*sqrt(pow(x1-x2,2.0)+pow(y1-y2,2.0))) && (sqrt(pow(x1-x2,2.0)+pow(y1-y2,2.0))/h < 1.0)) {
        drho_dx2(0, 0)  = -fac*(1.0/(h*h*h)*sqrt(pow(x1-x2,2.0)+pow(y1-y2,2.0))*(3.0/4.0)+1.0/(h*h)*(sqrt(pow(x1-x2,2.0)+pow(y1-y2,2.0))/(h*2.0)-1.0)*3.0+1.0/(h*h*h)*pow(x1*2.0-x2*2.0,2.0)*1.0/sqrt(pow(x1-x2,2.0)+pow(y1-y2,2.0))*(9.0/1.6E+1));
        drho_dx2(1, 0)= fac*1.0/(h*h*h)*(x1*2.0-x2*2.0)*(y1*2.0-y2*2.0)*1.0/sqrt(pow(x1-x2,2.0)+pow(y1-y2,2.0))*(-9.0/1.6E+1);
        drho_dx2(0, 1) = fac*1.0/(h*h*h)*(x1*2.0-x2*2.0)*(y1*2.0-y2*2.0)*1.0/sqrt(pow(x1-x2,2.0)+pow(y1-y2,2.0))*(-9.0/1.6E+1);
        drho_dx2(1, 1) = -fac*(1.0/(h*h*h)*sqrt(pow(x1-x2,2.0)+pow(y1-y2,2.0))*(3.0/4.0)+1.0/(h*h)*(sqrt(pow(x1-x2,2.0)+pow(y1-y2,2.0))/(h*2.0)-1.0)*3.0+1.0/(h*h*h)*pow(y1*2.0-y2*2.0,2.0)*1.0/sqrt(pow(x1-x2,2.0)+pow(y1-y2,2.0))*(9.0/1.6E+1));

       } else if ((pow(h-sqrt(pow(x1-x2,2.0)+pow(y1-y2,2.0))/2.0,2.0) < pow(x1-x2,2.0)/4.0+pow(y1-y2,2.0)/4.0) && (sqrt(pow(x1-x2,2.0)+pow(y1-y2,2.0))/h < 2.0)) {
        drho_dx2(0, 0) = (fac*pow(sqrt(pow(x1-x2,2.0)+pow(y1-y2,2.0))/h-2.0,2.0)*1.0/sqrt(pow(x1-x2,2.0)+pow(y1-y2,2.0))*(3.0/4.0))/h+(fac*1.0/(h*h)*pow(x1*2.0-x2*2.0,2.0)*(sqrt(pow(x1-x2,2.0)+pow(y1-y2,2.0))/h-2.0)*(3.0/8.0))/(pow(x1-x2,2.0)+pow(y1-y2,2.0))-(fac*pow(x1*2.0-x2*2.0,2.0)*pow(sqrt(pow(x1-x2,2.0)+pow(y1-y2,2.0))/h-2.0,2.0)*1.0/pow(pow(x1-x2,2.0)+pow(y1-y2,2.0),3.0/2.0)*(3.0/1.6E+1))/h;
       drho_dx2(1, 0) = (fac*(x1*2.0-x2*2.0)*(y1*2.0-y2*2.0)*pow(sqrt(pow(x1-x2,2.0)+pow(y1-y2,2.0))/h-2.0,2.0)*1.0/pow(pow(x1-x2,2.0)+pow(y1-y2,2.0),3.0/2.0)*(-3.0/1.6E+1))/h+(fac*1.0/(h*h)*(x1*2.0-x2*2.0)*(y1*2.0-y2*2.0)*(sqrt(pow(x1-x2,2.0)+pow(y1-y2,2.0))/h-2.0)*(3.0/8.0))/(pow(x1-x2,2.0)+pow(y1-y2,2.0));
       drho_dx2(0, 1) = (fac*(x1*2.0-x2*2.0)*(y1*2.0-y2*2.0)*pow(sqrt(pow(x1-x2,2.0)+pow(y1-y2,2.0))/h-2.0,2.0)*1.0/pow(pow(x1-x2,2.0)+pow(y1-y2,2.0),3.0/2.0)*(-3.0/1.6E+1))/h+(fac*1.0/(h*h)*(x1*2.0-x2*2.0)*(y1*2.0-y2*2.0)*(sqrt(pow(x1-x2,2.0)+pow(y1-y2,2.0))/h-2.0)*(3.0/8.0))/(pow(x1-x2,2.0)+pow(y1-y2,2.0));
       drho_dx2(1, 1) = (fac*pow(sqrt(pow(x1-x2,2.0)+pow(y1-y2,2.0))/h-2.0,2.0)*1.0/sqrt(pow(x1-x2,2.0)+pow(y1-y2,2.0))*(3.0/4.0))/h+(fac*1.0/(h*h)*pow(y1*2.0-y2*2.0,2.0)*(sqrt(pow(x1-x2,2.0)+pow(y1-y2,2.0))/h-2.0)*(3.0/8.0))/(pow(x1-x2,2.0)+pow(y1-y2,2.0))-(fac*pow(y1*2.0-y2*2.0,2.0)*pow(sqrt(pow(x1-x2,2.0)+pow(y1-y2,2.0))/h-2.0,2.0)*1.0/pow(pow(x1-x2,2.0)+pow(y1-y2,2.0),3.0/2.0)*(3.0/1.6E+1))/h;
       } else {
        drho_dx2(0, 0) = 0.0;
        drho_dx2(1, 0) = 0.0;
        drho_dx2(0, 1) = 0.0;
        drho_dx2(1, 1) = 0.0;
       
       
       
       
       
       } 

    return drho_dx2;
}


// explicit instantiations at bottom of file
template Matrix<double, 2*2, 1> density_gradient<2>(
    const Matrix<double, 2, 1>& xi,
    const Matrix<double, 2, 1>& xj,
    const double h,
    const double m,
    const double fac
);

template Matrix<double, 2, 1> density_gradient_element<2>(
    const Matrix<double, 2, 1>& xi,
    const Matrix<double, 2, 1>& xj,
    const double h,
    const double m,
    const double fac
);

template Matrix<double, 2, 2> density_hessian<2>(
    const Matrix<double, 2, 1>& xi,
    const Matrix<double, 2, 1>& xj,
    const double h,
    const double m,
    const double fac
);

template Eigen::VectorXd calculate_densities<2>(
    const Eigen::VectorXd& x,
    const std::vector<std::vector<int>>& neighbors,
    const double h,
    const double m,
    const double fac
);

template Eigen::VectorXd calculate_densities<3>(
    const Eigen::VectorXd& x,
    const std::vector<std::vector<int>>& neighbors,
    const double h,
    const double m,
    const double fac
);
