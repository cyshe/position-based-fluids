#pragma once

#include <Eigen/Core>
#include "cubic_bspline.h"
#include "calculate_densities.h"

#include <ipc/ipc.hpp>
#include <ipc/utils/eigen_ext.hpp>

template <int dim>
double surface_tension_energy(
    const Eigen::VectorXd & x,
    const std::vector<std::vector<int>> neighbors,
    const double h,
    const double m,
    const double fac,
    const double kappa,
    const double threshold,
    const bool smooth_mol
){
    int n = x.size() / dim;
    Eigen::VectorXd densities = calculate_densities<dim>(x, neighbors, h, m, fac);

    double energy = 0.0;
    double mol_k = 1000; // kappa in smooth mollifier exponential 
    //for (int i = 0; i < n; i++){
    //    for (int j = 0; j < neighbors[i].size(); j++){
    for (int i = 0; i < 2; i++){
        for (int j = 0; j < 2; j++){
            double mollifier;
           
            mollifier = 1/ (1 + exp(mol_k * (densities(i) - threshold)));
            if (!smooth_mol){
                mollifier = 1;
                if (densities(i) > threshold) {
                    mollifier = 0;
                }
            }
            mollifier = 1;
                if (densities(i) > threshold) {
                    mollifier = 0;
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
    const double rho_0,
    const double threshold,
    const bool smooth_mol,
    const Eigen::SparseMatrix<double> & B_sparse
){
    int n = x.size() / dim;

    Eigen::VectorXd grad = Eigen::VectorXd::Zero(x.size());
    Eigen::VectorXd densities = calculate_densities<dim>(x, neighbors, h, m, fac);

    Eigen::MatrixXd B = -B_sparse.toDense().transpose() * rho_0;

    double mol_k = 1000;
    //std::cout << B.cols() << B.rows() << std::endl;
    //std::cout << grad.size() << std::endl;
    // Now compute surface tension gradient
    double threshold_r = rho_0 * threshold;
    for (int i = 0; i < 2; i++){
        for (int j = 0; j < 2; j++){
            double mol, mol_grad;
            
            mol = 1/ (1 + exp(mol_k * (densities(i) - threshold_r)));
            mol_grad = -mol_k * exp(mol_k * (densities(i) - threshold_r)) / ((1 + exp(mol_k * (densities(i) - threshold_r))) * (1 + exp(mol_k * (densities(i) - threshold_r))));
            if (!smooth_mol){
                mol = 1; 
                mol_grad = 0;
                if (densities(i) > threshold_r) {
                    mol = 0;
                    mol_grad = 0;
                }
            }
            mol = 1; 
                mol_grad = 0;
                if (densities(i) > threshold_r) {
                    mol = 0;
                    mol_grad = 0;
                } 
            /*
            if (i == 0 && j == 0){
                std::cout << "Shapes: " << std::endl;
                std::cout << B.cols() << std::endl;
                std::cout << B.rows() << std::endl;
                std::cout << B.col(i).cols() << std::endl;
                std::cout << B.col(i).rows() << std::endl;

                std::cout << (((B.col(i) - B.col(neighbors[i][j])))).cols() << std::endl;
                std::cout << (((B.col(i) - B.col(neighbors[i][j])))).rows() << std::endl;
            }
            */
            grad += kappa * (((B.col(i) - B.col(neighbors[i][j])) * (densities(i) - densities(neighbors[i][j])) * mol)
                + B.col(i) * mol_grad * (densities(i)-densities(neighbors[i][j])) * (densities(i)-densities(neighbors[i][j])) *0.5);
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
    const double rho_0,
    const double threshold,
    const double smooth_mol,
    const Eigen::SparseMatrix<double> & B_sparse
){
    /*
    Eigen::VectorXd grad = surface_tension_gradient<dim>(x, neighbors, h, m, fac, kappa, rho_0, threshold, smooth_mol, B);
    if(kappa == 0){
        return grad * grad.transpose();
    }
    

    Eigen::MatrixXd hess = grad * grad.transpose()/kappa;
     return hess;
*/
    
    Eigen::MatrixXd B = -B_sparse.toDense()* rho_0;

    double mol_k = 1000;

    double threshold_r = rho_0 * threshold;
    Eigen::VectorXd densities = calculate_densities<dim>(x, neighbors, h, m, fac);

    Eigen::MatrixXd hessian = Eigen::MatrixXd::Zero(x.size(), x.size());
    int idx = 0;
    int n = x.size() / dim;

    //n = 5 + 2;

    for (int i = 0; i < 2; i++){
        for (int j = 0; j < 2; j++){
            
            Eigen::MatrixXd hessian_ij = Eigen::MatrixXd::Zero(n, n);
            double rho_i, rho_j, mol_i, mol_grad_i, mol_double_prime_i, u;
            rho_i = densities(i);
            rho_j = densities(j);
            u = rho_i - rho_j;


            mol_i = 1/ (1 + exp(mol_k * (rho_i - threshold_r)));
            mol_grad_i = -mol_k * exp(mol_k * (rho_i - threshold_r)) / ((1 + exp(mol_k * (rho_i - threshold_r))) * (1 + exp(mol_k * (rho_i - threshold_r))));
            mol_double_prime_i = -mol_k * (mol_k * exp(mol_k * (rho_i - threshold_r)) * (1 + exp(mol_k * (rho_i - threshold_r))) - 2 * exp(mol_k * (rho_i - threshold_r)) * exp(mol_k * (rho_i - threshold_r)))
                / ((1 + exp(mol_k * (rho_i - threshold_r))) * (1 + exp(mol_k * (rho_i - threshold_r))) * (1 + exp(mol_k * (rho_i - threshold_r))));
            
            if (!smooth_mol){
                mol_i = 1; 
                mol_grad_i = 0;
                mol_double_prime_i = 0;
                if (densities(i) > threshold_r) {
                    mol_i = 0;
                    mol_grad_i = 0;
                    mol_double_prime_i = 0;
                }
            }

            mol_i = 1; 
            mol_grad_i = 0;
            mol_double_prime_i = 0;
            if (densities(i) > threshold_r) {
                mol_i = 0;
                mol_grad_i = 0;
                mol_double_prime_i = 0;
            }
            
            for (int k = 0; k < 2; k++){
                Eigen::Matrix<double, dim, 1> w_k = Eigen::Matrix<double, dim, 1>::Zero(dim);
                Eigen::Matrix<double, dim, 1> z_k = Eigen::Matrix<double, dim, 1>::Zero();
                Eigen::Matrix<double, dim, 1> w_diff_k = Eigen::Matrix<double, dim, 1>::Zero();
                Eigen::Matrix<double, dim, 1> w_l = Eigen::Matrix<double, dim, 1>::Zero();
                Eigen::Matrix<double, dim, 1> z_l = Eigen::Matrix<double, dim, 1>::Zero();
                Eigen::Matrix<double, dim, 1> w_diff_l = Eigen::Matrix<double, dim, 1>::Zero();
                Eigen::Matrix<double, dim, dim> hess_rho_diff = Eigen::Matrix<double, dim, dim>::Zero();
                Eigen::Matrix<double, dim, dim> term1 = Eigen::Matrix<double, dim, dim>::Zero();
                Eigen::Matrix<double, dim, dim> term2 = Eigen::Matrix<double, dim, dim>::Zero();


                w_k = B.block<1, dim>(k, i * dim).transpose();
                z_k = B.block<1, dim>(k, j * dim).transpose();
                w_diff_k = w_k - z_k;

                for (int l=0; l < 2; l++){
                    w_l = B.block<1, dim>(l, i * dim).transpose();
                    z_l = B.block<1, dim>(l, j * dim).transpose();

                    w_diff_l = w_l - z_l;
                    const auto& xk = x.template segment<dim>(dim * k);
                    const auto& xl = x.template segment<dim>(dim * l);
                    double x_kl = (xk - xl).norm();


                    Eigen::Matrix<double, dim, dim> hess_rho_i_kl;  // Hessian of rho_i with respect to x_k and x_l
                    
                    //std::cout << i << " i";
                    hess_rho_i_kl = density_hessian<dim>(x, neighbors, i, k, l, h, m, fac, rho_0, B_sparse);

                    //if ((i == k) && (std::find(neighbors[i].begin(), neighbors[i].end(), l) != neighbors[i].end())){
                    //    hess_rho_i_kl = density_hessian<dim>(x.template segment<dim>(dim * k), x.template segment<dim>(dim * l), h, m, fac, rho_0, B_sparse);
                    //}
                    //else if ((i == l) && (std::find(neighbors[i].begin(), neighbors[i].end(), k) != neighbors[i].end())){
                    //    hess_rho_i_kl = density_hessian<dim>(x.template segment<dim>(dim * l), x.template segment<dim>(dim * k), h, m, fac, rho_0, B_sparse);
                    //}
                    
                    //std::cout << j << " j" << std::endl;
                    Eigen::Matrix<double, dim, dim> hess_rho_j_kl;  // Hessian of rho_j with respect to x_k and x_l

                    hess_rho_j_kl = density_hessian<dim>(x, neighbors, j, k, l, h, m, fac, rho_0, B_sparse);

                    //if ((j == k) && (std::find(neighbors[j].begin(), neighbors[j].end(), l) != neighbors[j].end())){
                    //    hess_rho_j_kl = density_hessian<dim>(x.template segment<dim>(dim * k), x.template segment<dim>(dim * l), h, m, fac, rho_0, B_sparse);
                    //}
                    //else if ((j == l) && (std::find(neighbors[j].begin(), neighbors[j].end(), k) != neighbors[j].end())){
                    //    hess_rho_j_kl = density_hessian<dim>(x.template segment<dim>(dim * l), x.template segment<dim>(dim * k), h, m, fac, rho_0, B_sparse);
                    //}
                    hess_rho_diff = hess_rho_i_kl - hess_rho_j_kl;  // Hessian difference

                    // First term
                    term1 = kappa * (w_diff_l * mol_i * w_diff_k.transpose() + u * mol_grad_i * w_l * w_diff_k.transpose() + u * mol_i * hess_rho_diff);

                    // Second term
                    term2 = 0.5 * kappa * (2 * u * w_diff_l * mol_grad_i * w_k.transpose() + u * u * (mol_double_prime_i * w_l * w_k.transpose() + mol_grad_i * hess_rho_i_kl));

                    // Add the terms to the Hessian
                    hessian.block<dim, dim>(dim*k, dim*l) = term1 + term2;
                }
            }
            
        } 
    }
    




    return hessian;
};

