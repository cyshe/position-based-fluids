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
    for (int i = 0; i < n; i++){
        for (int j = 0; j < neighbors[i].size(); j++){
            double mollifier;
           
            mollifier = 1/ (1 + exp(mol_k * (densities(i) - threshold)));
            if (!smooth_mol){
                mollifier = 1;
                if (densities(i) > threshold) {
                    mollifier = 0;
                }
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
    for (int i = 0; i < n; i++){
        for (int j = 0; j < neighbors[i].size(); j++){
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
    const Eigen::SparseMatrix<double> & B
){
    
    Eigen::VectorXd grad = surface_tension_gradient<dim>(x, neighbors, h, m, fac, kappa, rho_0, threshold, smooth_mol, B);
    if(kappa == 0){
        return grad * grad.transpose();
    }
    

    Eigen::MatrixXd hess = grad * grad.transpose()/kappa;
    /* 
    Eigen::MatrixXd hessian = Eigen::MatrixXd::Zero(x.size(), x.size());
    int idx = 0;
    for (int i = 0; i < neighbors.size(); i+=dim){
        for (int j = 0; j < neighbors[i].size(); j++){
            idx = neighbors[i][j];
            double mol_i, mol_grad_i, mol_grad2_i, mol_j, mol_grad_j, mol_grad2_j;
            mol_i = 1/ (1 + exp(mol_k * (densities(i) - threshold_r)));
            mol_grad_i = -mol_k * exp(mol_k * (densities(i) - threshold_r)) / ((1 + exp(mol_k * (densities(i) - threshold_r))) * (1 + exp(mol_k * (densities(i) - threshold_r))));
            if (!smooth_mol){
                mol_i = 1; 
                mol_grad_i = 0;
                if (densities(i) > threshold_r) {
                    mol_i = 0;
                    mol_grad_i = 0;
                }
            }

            mol_j = 1/ (1 + exp(mol_k * (densities(idx) - threshold_r)));
            mod_grad_j = -mol_k * exp(mol_k * (densities(idx) - threshold_r)) / ((1 + exp(mol_k * (densities(idx) - threshold_r))) * (1 + exp(mol_k * (densities(idx) - threshold_r))));
            if (!smooth_mol){
                mol_j = 1; 
                mol_grad_j = 0;
                if (densities(idx) > threshold_r) {
                    mol_j = 0;
                    mol_grad_j = 0;
                }
            }


            hessian.block<dim, dim>(dim*i, dim*i) += (//dB here * (densities(i) - densities(j)) * mol_i
                + (B.col(i) - B.col(idx)) * (B.col(i) - B.col(idx)) * mol_i
                + (B.col(i) - B.col(idx)) * (densities(i) - densities(idx)) * mol_grad_i
                +  dB.col(i) * mol_grad_i * (densities(i)-densities(idx)) * (densities(i)-densities(idx)) * 0.5
                +  B.col(i) * B.col(i) * mol_grad2_i * (densities(i)-densities(idx)) * (densities(i)-densities(idx)) * 0.5
                +  B.col(i) * mol_grad_i * (B.col(i) - B.col(idx)) * (densities(i)-densities(neighbors[i][j]))
                );

            hessian.block<dim, dim>(dim*i, dim*idx) += ;
            hessian.block<dim, dim>(dim*idx, dim*i) += ;
            hessian.block<dim, dim>(dim*idx, dim*idx) += ;
        }
    }
    */




    return hess;
};

