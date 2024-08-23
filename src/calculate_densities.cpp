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
Matrix<double, 1, 1> calculate_density_stencil<dim>(
    const Matrix<double, dim*2, 1>& x,
    const double h,
    const double m,
    const double fac
)
{
    int n = x.size() / dim;
    assert(neighbors.size() == n);
    
    const auto& xi = x.template segment<dim>(dim * 0);
    const auto& xj = x.template segment<dim>(dim * 1);

    Matrix<double, 1, 1> densities;
    double r = (xj - xi).norm()/h;
    densities(0,0) = cubic_bspline(r, m*fac);
    

    return densities;
};


template <int dim>
Matrix<double, dim, dim> density_hessian(
    Eigen::VectorXd & x,
    std::vector<std::vector<int>> neighbors,
    const int i,
    const int k,
    const int l,
    const double h,
    const double m,
    const double fac,
    const double rho_0,
    const Eigen::SparseMatrix<double> & B_sparse
){
    if ((i != k && i != l) || (std::find(neighbors[k].begin(), neighbors[k].end(), l) == neighbors[k].end())){
        return Matrix<double, dim, dim>::Zero();
    }
   
    int n = x.size() / dim;

    Matrix<double, dim, 1> xi;
    Matrix<double, dim, 1> xj;

    if (i == k){
        xi = x.template segment<dim>(dim * k);
        xj = x.template segment<dim>(dim * l);
    }
    else { // i == l
        xi = x.template segment<dim>(dim * l);
        xj = x.template segment<dim>(dim * k);
    }
    

    Matrix<double, dim, dim> hess;
    Matrix<double, dim, 1> xij = xj - xi;
    double r = xij.norm() / h;

    if (r == 0){
        return Matrix<double, dim, dim>::Zero();
    }
    // double x1 = xi(0);
    // double y1 = xi(1);
    // double x2 = xj(0);
    // double y2 = xj(1);
    double x1, x2, x3, x4;

    // Matrix<double, dim, 1> drho_dxi;
    // Matrix<double, dim, 1> drho_dxj;

    // Matrix<double, dim, 1> dr_dxi;
    // Matrix<double, dim, 1> dr_dxj;
    
    // dr_dxi(0) = //((x1*2.0-x2*2.0)*1.0/sqrt(pow(x1-x2,2.0)+pow(y1-y2,2.0)))/(h*2.0);
    // ((x1*2.0-x2*2.0)*1.0/sqrt(pow(x1-x2,2.0)+pow(y1-y2,2.0)))/(h*2.0);
    // dr_dxi(1) = //((y1*2.0-y2*2.0)*1.0/sqrt(pow(x1-x2,2.0)+pow(y1-y2,2.0)))/(h*2.0);
    // ((y1*2.0-y2*2.0)*1.0/sqrt(pow(x1-x2,2.0)+pow(y1-y2,2.0)))/(h*2.0);
    // dr_dxj(0) = //((x1*2.0-x2*2.0)*1.0/sqrt(pow(x1-x2,2.0)+pow(y1-y2,2.0))*(-1.0/2.0))/h;
    // ((x1*2.0-x2*2.0)*1.0/sqrt(pow(x1-x2,2.0)+pow(y1-y2,2.0))*(-1.0/2.0))/h;
    // dr_dxj(1) = //((y1*2.0-y2*2.0)*1.0/sqrt(pow(x1-x2,2.0)+pow(y1-y2,2.0))*(-1.0/2.0))/h; 
    // ((y1*2.0-y2*2.0)*1.0/sqrt(pow(x1-x2,2.0)+pow(y1-y2,2.0))*(-1.0/2.0))/h;

    // if ((0.0 < h*sqrt(pow(x1-x2,2.0)+pow(y1-y2,2.0))) && (sqrt(pow(x1-x2,2.0)+pow(y1-y2,2.0))/h < 1.0)) {
    //     drho_dxi(0) = fac*(1.0/(h*h*h)*(x1*2.0-x2*2.0)*sqrt(pow(x1-x2,2.0)+pow(y1-y2,2.0))*(3.0/8.0)+1.0/(h*h)*(x1*2.0-x2*2.0)*(sqrt(pow(x1-x2,2.0)+pow(y1-y2,2.0))/(h*2.0)-1.0)*(3.0/2.0));
    //     drho_dxi(1) = fac*(1.0/(h*h*h)*(y1*2.0-y2*2.0)*sqrt(pow(x1-x2,2.0)+pow(y1-y2,2.0))*(3.0/8.0)+1.0/(h*h)*(y1*2.0-y2*2.0)*(sqrt(pow(x1-x2,2.0)+pow(y1-y2,2.0))/(h*2.0)-1.0)*(3.0/2.0)); 
    //     drho_dxj(0) = -fac*(1.0/(h*h*h)*(x1*2.0-x2*2.0)*sqrt(pow(x1-x2,2.0)+pow(y1-y2,2.0))*(3.0/8.0)+1.0/(h*h)*(x1*2.0-x2*2.0)*(sqrt(pow(x1-x2,2.0)+pow(y1-y2,2.0))/(h*2.0)-1.0)*(3.0/2.0));
    //     drho_dxj(1) = -fac*(1.0/(h*h*h)*(y1*2.0-y2*2.0)*sqrt(pow(x1-x2,2.0)+pow(y1-y2,2.0))*(3.0/8.0)+1.0/(h*h)*(y1*2.0-y2*2.0)*(sqrt(pow(x1-x2,2.0)+pow(y1-y2,2.0))/(h*2.0)-1.0)*(3.0/2.0));
    // } else if ((pow(h-sqrt(pow(x1-x2,2.0)+pow(y1-y2,2.0))/2.0,2.0) < pow(x1-x2,2.0)/4.0+pow(y1-y2,2.0)/4.0) && (sqrt(pow(x1-x2,2.0)+pow(y1-y2,2.0))/h < 2.0)) {
    //     drho_dxi(0) = (fac*(x1*2.0-x2*2.0)*pow(sqrt(pow(x1-x2,2.0)+pow(y1-y2,2.0))/h-2.0,2.0)*1.0/sqrt(pow(x1-x2,2.0)+pow(y1-y2,2.0))*(-3.0/8.0))/h;
    //     drho_dxi(1) = (fac*(y1*2.0-y2*2.0)*pow(sqrt(pow(x1-x2,2.0)+pow(y1-y2,2.0))/h-2.0,2.0)*1.0/sqrt(pow(x1-x2,2.0)+pow(y1-y2,2.0))*(-3.0/8.0))/h; 
    //     drho_dxj(0) = (fac*(x1*2.0-x2*2.0)*pow(sqrt(pow(x1-x2,2.0)+pow(y1-y2,2.0))/h-2.0,2.0)*1.0/sqrt(pow(x1-x2,2.0)+pow(y1-y2,2.0))*(3.0/8.0))/h;
    //     drho_dxj(1) = (fac*(y1*2.0-y2*2.0)*pow(sqrt(pow(x1-x2,2.0)+pow(y1-y2,2.0))/h-2.0,2.0)*1.0/sqrt(pow(x1-x2,2.0)+pow(y1-y2,2.0))*(3.0/8.0))/h;
    // } else {
    //     drho_dxi(0) = 0.0;
    //     drho_dxi(1) = 0.0;
    //     drho_dxj(0) = 0.0;
    //     drho_dxj(1) = 0.0;
    // }

    //std::cout << "drho_dxi" << std::endl;
    //std::cout << drho_dxi/rho_0 << std::endl;
    //std::cout << "drho_dxj" << std::endl;
    //std::cout << drho_dxj/rho_0 << std::endl;
    


    Eigen::MatrixXd B = -B_sparse.toDense()* rho_0;

    //std::cout << "B shape" << B.rows() << B.cols() << std::endl;

    

    Matrix<double, dim, dim> dr2_dx2;
    Matrix<double, dim, dim> drho_dx2;
    


    // dr2_dx2(0, 0) = -1.0/sqrt(pow(x1-x2,2.0)+pow(y1-y2,2.0))/h+(pow(x1*2.0-x2*2.0,2.0)*1.0/pow(pow(x1-x2,2.0)+pow(y1-y2,2.0),3.0/2.0))/(h*4.0); 
    // dr2_dx2(1, 0) = ((x1*2.0-x2*2.0)*(y1*2.0-y2*2.0)*1.0/pow(pow(x1-x2,2.0)+pow(y1-y2,2.0),3.0/2.0))/(h*4.0);
    // dr2_dx2(0, 1) = ((x1*2.0-x2*2.0)*(y1*2.0-y2*2.0)*1.0/pow(pow(x1-x2,2.0)+pow(y1-y2,2.0),3.0/2.0))/(h*4.0);
    // dr2_dx2(1, 1) = -1.0/sqrt(pow(x1-x2,2.0)+pow(y1-y2,2.0))/h+(pow(y1*2.0-y2*2.0,2.0)*1.0/pow(pow(x1-x2,2.0)+pow(y1-y2,2.0),3.0/2.0))/(h*4.0);
    // hess = dr_dxj * dr_dxi.transpose() * cubic_bspline_hessian(r, m*fac) + cubic_bspline_derivative(r, m*fac) * dr2_dx2;
    //std::cout << "hess from stencil" << std::endl;
    //std::cout << hess/rho_0 << std::endl;


    //if (i == k){
    //    drho_dxi = (B.block<1, dim> (i, dim*k)).transpose();
    //    drho_dxj = (B.block<1, dim> (i, dim*l)).transpose();
    //}
    //else { // i == l
    //    drho_dxi = (B.block<1, dim> (i, dim*l)).transpose();
    //    drho_dxj = (B.block<1, dim> (i, dim*k)).transpose();
    //}


    //hess = drho_dxj * drho_dxi.transpose() * cubic_bspline_hessian(r, m*fac) + cubic_bspline_derivative(r, m*fac) * drho_dx2;
    //std::cout << "hess from whole drho" << std::endl;
    //std::cout << hess << std::endl;

    
    // if ((0.0 < h*sqrt(pow(x1-x2,2.0)+pow(y1-y2,2.0))) && (sqrt(pow(x1-x2,2.0)+pow(y1-y2,2.0))/h < 1.0)) {
    //     drho_dx2(0, 0)  = -m * fac*(1.0/(h*h*h)*sqrt(pow(x1-x2,2.0)+pow(y1-y2,2.0))*(3.0/4.0)+1.0/(h*h)*(sqrt(pow(x1-x2,2.0)+pow(y1-y2,2.0))/(h*2.0)-1.0)*3.0+1.0/(h*h*h)*pow(x1*2.0-x2*2.0,2.0)*1.0/sqrt(pow(x1-x2,2.0)+pow(y1-y2,2.0))*(9.0/1.6E+1));
    //     drho_dx2(1, 0)= m*fac*1.0/(h*h*h)*(x1*2.0-x2*2.0)*(y1*2.0-y2*2.0)*1.0/sqrt(pow(x1-x2,2.0)+pow(y1-y2,2.0))*(-9.0/1.6E+1);
    //     drho_dx2(0, 1) = m*fac*1.0/(h*h*h)*(x1*2.0-x2*2.0)*(y1*2.0-y2*2.0)*1.0/sqrt(pow(x1-x2,2.0)+pow(y1-y2,2.0))*(-9.0/1.6E+1);
    //     drho_dx2(1, 1) = -m*fac*(1.0/(h*h*h)*sqrt(pow(x1-x2,2.0)+pow(y1-y2,2.0))*(3.0/4.0)+1.0/(h*h)*(sqrt(pow(x1-x2,2.0)+pow(y1-y2,2.0))/(h*2.0)-1.0)*3.0+1.0/(h*h*h)*pow(y1*2.0-y2*2.0,2.0)*1.0/sqrt(pow(x1-x2,2.0)+pow(y1-y2,2.0))*(9.0/1.6E+1));
    // } else if ((pow(h-sqrt(pow(x1-x2,2.0)+pow(y1-y2,2.0))/2.0,2.0) < pow(x1-x2,2.0)/4.0+pow(y1-y2,2.0)/4.0) && (sqrt(pow(x1-x2,2.0)+pow(y1-y2,2.0))/h < 2.0)) {
    //    drho_dx2(0, 0) = (m*fac*pow(sqrt(pow(x1-x2,2.0)+pow(y1-y2,2.0))/h-2.0,2.0)*1.0/sqrt(pow(x1-x2,2.0)+pow(y1-y2,2.0))*(3.0/4.0))/h+(m*fac*1.0/(h*h)*pow(x1*2.0-x2*2.0,2.0)*(sqrt(pow(x1-x2,2.0)+pow(y1-y2,2.0))/h-2.0)*(3.0/8.0))/(pow(x1-x2,2.0)+pow(y1-y2,2.0))-(m*fac*pow(x1*2.0-x2*2.0,2.0)*pow(sqrt(pow(x1-x2,2.0)+pow(y1-y2,2.0))/h-2.0,2.0)*1.0/pow(pow(x1-x2,2.0)+pow(y1-y2,2.0),3.0/2.0)*(3.0/1.6E+1))/h;
    //    drho_dx2(1, 0) = (m*fac*(x1*2.0-x2*2.0)*(y1*2.0-y2*2.0)*pow(sqrt(pow(x1-x2,2.0)+pow(y1-y2,2.0))/h-2.0,2.0)*1.0/pow(pow(x1-x2,2.0)+pow(y1-y2,2.0),3.0/2.0)*(-3.0/1.6E+1))/h+(m*fac*1.0/(h*h)*(x1*2.0-x2*2.0)*(y1*2.0-y2*2.0)*(sqrt(pow(x1-x2,2.0)+pow(y1-y2,2.0))/h-2.0)*(3.0/8.0))/(pow(x1-x2,2.0)+pow(y1-y2,2.0));
    //    drho_dx2(0, 1) = (m*fac*(x1*2.0-x2*2.0)*(y1*2.0-y2*2.0)*pow(sqrt(pow(x1-x2,2.0)+pow(y1-y2,2.0))/h-2.0,2.0)*1.0/pow(pow(x1-x2,2.0)+pow(y1-y2,2.0),3.0/2.0)*(-3.0/1.6E+1))/h+(m*fac*1.0/(h*h)*(x1*2.0-x2*2.0)*(y1*2.0-y2*2.0)*(sqrt(pow(x1-x2,2.0)+pow(y1-y2,2.0))/h-2.0)*(3.0/8.0))/(pow(x1-x2,2.0)+pow(y1-y2,2.0));
    //    drho_dx2(1, 1) = (m*fac*pow(sqrt(pow(x1-x2,2.0)+pow(y1-y2,2.0))/h-2.0,2.0)*1.0/sqrt(pow(x1-x2,2.0)+pow(y1-y2,2.0))*(3.0/4.0))/h+(m*fac*1.0/(h*h)*pow(y1*2.0-y2*2.0,2.0)*(sqrt(pow(x1-x2,2.0)+pow(y1-y2,2.0))/h-2.0)*(3.0/8.0))/(pow(x1-x2,2.0)+pow(y1-y2,2.0))-(m*fac*pow(y1*2.0-y2*2.0,2.0)*pow(sqrt(pow(x1-x2,2.0)+pow(y1-y2,2.0))/h-2.0,2.0)*1.0/pow(pow(x1-x2,2.0)+pow(y1-y2,2.0),3.0/2.0)*(3.0/1.6E+1))/h;
    // } else {
    //     drho_dx2(0, 0) = 0.0;
    //     drho_dx2(1, 0) = 0.0;
    //     drho_dx2(0, 1) = 0.0;
    //     drho_dx2(1, 1) = 0.0;
    // } 

    //std::cout << "drho_dx2 old" << std::endl;
    //std::cout << drho_dx2/rho_0 << std::endl;    

    x1 = xi(0);
    x2 = xi(1);
    x3 = xj(0);
    x4 = xj(1);

    if ((0.0 < h*sqrt(pow(x1-x3,2.0)+pow(x2-x4,2.0))) && (sqrt(pow(x1-x3,2.0)+pow(x2-x4,2.0))/h < 1.0)){
        drho_dx2(0, 0) = -fac*(1.0/(h*h*h)*sqrt(pow(x1-x3,2.0)+pow(x2-x4,2.0))*(3.0/4.0)+1.0/(h*h)*(sqrt(pow(x1-x3,2.0)+pow(x2-x4,2.0))/(h*2.0)-1.0)*3.0+1.0/(h*h*h)*pow(x1*2.0-x3*2.0,2.0)*1.0/sqrt(pow(x1-x3,2.0)+pow(x2-x4,2.0))*(9.0/1.6E+1));
        drho_dx2(1, 0) = fac*1.0/(h*h*h)*(x1*2.0-x3*2.0)*(x2*2.0-x4*2.0)*1.0/sqrt(pow(x1-x3,2.0)+pow(x2-x4,2.0))*(-9.0/1.6E+1);
        drho_dx2(0, 1) = fac*1.0/(h*h*h)*(x1*2.0-x3*2.0)*(x2*2.0-x4*2.0)*1.0/sqrt(pow(x1-x3,2.0)+pow(x2-x4,2.0))*(-9.0/1.6E+1);    
        drho_dx2(1, 1) = -fac*(1.0/(h*h*h)*sqrt(pow(x1-x3,2.0)+pow(x2-x4,2.0))*(3.0/4.0)+1.0/(h*h)*(sqrt(pow(x1-x3,2.0)+pow(x2-x4,2.0))/(h*2.0)-1.0)*3.0+1.0/(h*h*h)*pow(x2*2.0-x4*2.0,2.0)*1.0/sqrt(pow(x1-x3,2.0)+pow(x2-x4,2.0))*(9.0/1.6E+1));
      
    } else if ((sqrt(pow(x1-x3,2.0)+pow(x2-x4,2.0))/h < 2.0) && (pow(h-sqrt(pow(x1-x3,2.0)+pow(x2-x4,2.0))/2.0,2.0) < pow(x1-x3,2.0)/4.0+pow(x2-x4,2.0)/4.0)) {
        drho_dx2(0, 0) = (fac*pow(sqrt(pow(x1-x3,2.0)+pow(x2-x4,2.0))/h-2.0,2.0)*1.0/sqrt(pow(x1-x3,2.0)+pow(x2-x4,2.0))*(3.0/4.0))/h+(fac*1.0/(h*h)*pow(x1*2.0-x3*2.0,2.0)*(sqrt(pow(x1-x3,2.0)+pow(x2-x4,2.0))/h-2.0)*(3.0/8.0))/(pow(x1-x3,2.0)+pow(x2-x4,2.0))-(fac*pow(x1*2.0-x3*2.0,2.0)*pow(sqrt(pow(x1-x3,2.0)+pow(x2-x4,2.0))/h-2.0,2.0)*1.0/pow(pow(x1-x3,2.0)+pow(x2-x4,2.0),3.0/2.0)*(3.0/1.6E+1))/h;
        drho_dx2(1, 0) = (fac*(x1*2.0-x3*2.0)*(x2*2.0-x4*2.0)*pow(sqrt(pow(x1-x3,2.0)+pow(x2-x4,2.0))/h-2.0,2.0)*1.0/pow(pow(x1-x3,2.0)+pow(x2-x4,2.0),3.0/2.0)*(-3.0/1.6E+1))/h+(fac*1.0/(h*h)*(x1*2.0-x3*2.0)*(x2*2.0-x4*2.0)*(sqrt(pow(x1-x3,2.0)+pow(x2-x4,2.0))/h-2.0)*(3.0/8.0))/(pow(x1-x3,2.0)+pow(x2-x4,2.0));
        drho_dx2(0, 1) = (fac*(x1*2.0-x3*2.0)*(x2*2.0-x4*2.0)*pow(sqrt(pow(x1-x3,2.0)+pow(x2-x4,2.0))/h-2.0,2.0)*1.0/pow(pow(x1-x3,2.0)+pow(x2-x4,2.0),3.0/2.0)*(-3.0/1.6E+1))/h+(fac*1.0/(h*h)*(x1*2.0-x3*2.0)*(x2*2.0-x4*2.0)*(sqrt(pow(x1-x3,2.0)+pow(x2-x4,2.0))/h-2.0)*(3.0/8.0))/(pow(x1-x3,2.0)+pow(x2-x4,2.0));
        drho_dx2(1, 1) = (fac*pow(sqrt(pow(x1-x3,2.0)+pow(x2-x4,2.0))/h-2.0,2.0)*1.0/sqrt(pow(x1-x3,2.0)+pow(x2-x4,2.0))*(3.0/4.0))/h+(fac*1.0/(h*h)*pow(x2*2.0-x4*2.0,2.0)*(sqrt(pow(x1-x3,2.0)+pow(x2-x4,2.0))/h-2.0)*(3.0/8.0))/(pow(x1-x3,2.0)+pow(x2-x4,2.0))-(fac*pow(x2*2.0-x4*2.0,2.0)*pow(sqrt(pow(x1-x3,2.0)+pow(x2-x4,2.0))/h-2.0,2.0)*1.0/pow(pow(x1-x3,2.0)+pow(x2-x4,2.0),3.0/2.0)*(3.0/1.6E+1))/h;
       
    } else {
        drho_dx2(0, 0) = 0.0;
        drho_dx2(1, 0) = 0.0;
        drho_dx2(0, 1) = 0.0;
        drho_dx2(1, 1) = 0.0;
    }
    
    //std::cout << "drho_dx2 new" << std::endl;
    //std::cout << drho_dx2/rho_0 << std::endl;

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
    Eigen::VectorXd & x,
    std::vector<std::vector<int>> neighbors,
    const int i,
    const int k,
    const int l,
    const double h,
    const double m,
    const double fac,
    const double rho_0,
    const Eigen::SparseMatrix<double> & B_sparse
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

template Matrix<double, 1, 1> calculate_density_stencil<2>(
    const Matrix<double, 4, 1>& x,
    const double h,
    const double m,
    const double fac
);