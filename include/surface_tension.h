#pragma once

#include <Eigen/Core>
#include "cubic_bspline.h"
#include "calculate_densities.h"

#include <ipc/ipc.hpp>
#include <ipc/utils/eigen_ext.hpp>
#include "oneapi/tbb.h"

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
    double mol_k = 10; // kappa in smooth mollifier exponential 
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

    double mol_k = 10;
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
            if (isnan(densities(i)) || isnan(densities(neighbors[i][j]))){
                std::cout << "Density is nan" << std::endl;
            }
            if (isnan(mol) || isnan(mol_grad)){
                std::cout << "Mol is nan" << std::endl;
                std::cout << "denominator " << (1 + exp(mol_k * (densities(i) - threshold_r))) << std::endl;
                std::cout << "denominator " << ((mol_k * (densities(i) - threshold_r))) << std::endl;
            }
            if (isnan(B.col(i).norm()) || isnan(B.col(neighbors[i][j]).norm())){
                std::cout << "B is nan" << std::endl;
            }
            grad += kappa * (((B.col(i) - B.col(neighbors[i][j])) * (densities(i) - densities(neighbors[i][j])) * mol)
                + B.col(i) * mol_grad * (densities(i)-densities(neighbors[i][j])) * (densities(i)-densities(neighbors[i][j])) *0.5);
        }
    }
    std::cout << "densities norm " << densities.norm() << std::endl;
    std::cout << "Gradient norm: " << grad.norm() << std::endl;

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
    double mol_k = 10;

    double threshold_r = rho_0 * threshold;
    Eigen::VectorXd densities = calculate_densities<dim>(x, neighbors, h, m, fac);

    Eigen::MatrixXd hessian = Eigen::MatrixXd::Zero(x.size(), x.size());

    int idx = 0;
    int n = x.size() / dim;

    //TODO: create list of particles to loop over, skip mol = 0s
    std::vector<int> particles_nonzero_mol;
    double rho_i, rho_j, mol_i, mol_grad_i, mol_double_prime_i, u;
    Eigen::Matrix<double, dim, 1> w_k = Eigen::Matrix<double, dim, 1>::Zero();
    Eigen::Matrix<double, dim, 1> z_k = Eigen::Matrix<double, dim, 1>::Zero();
    Eigen::Matrix<double, dim, 1> w_diff_k = Eigen::Matrix<double, dim, 1>::Zero();
    Eigen::Matrix<double, dim, 1> w_l = Eigen::Matrix<double, dim, 1>::Zero();
    Eigen::Matrix<double, dim, 1> z_l = Eigen::Matrix<double, dim, 1>::Zero();
    Eigen::Matrix<double, dim, 1> w_diff_l = Eigen::Matrix<double, dim, 1>::Zero();
    Eigen::Matrix<double, dim, dim> hess_rho_diff = Eigen::Matrix<double, dim, dim>::Zero();
    Eigen::Matrix<double, dim, dim> term1 = Eigen::Matrix<double, dim, dim>::Zero();
    Eigen::Matrix<double, dim, dim> term2 = Eigen::Matrix<double, dim, dim>::Zero();
    Eigen::Matrix<double, dim, dim> hess_subblock = Eigen::Matrix<double, dim, dim>::Zero();
    Eigen::Matrix<double, dim, dim> hess_rho_i_kl;  // Hessian of rho_i with respect to x_k a
    Eigen::Matrix<double, dim, dim> hess_rho_j_kl;  // Hessian of rho_j with respect to x_k and x_l
    
    for (int i = 0; i < n; i++) {
        rho_i = densities(i);
        mol_i = 1 / (1 + exp(mol_k * (rho_i - threshold_r)));
        // std::cout << "mol_i " << mol_i << std::endl;
        // std::cout << "rho_i " << rho_i << std::endl;
        // std::cout << "threshold_r" << threshold_r << std::endl;


        mol_grad_i = -mol_k * exp(mol_k * (rho_i - threshold_r)) / 
                     ((1 + exp(mol_k * (rho_i - threshold_r))) * (1 + exp(mol_k * (rho_i - threshold_r))));
        mol_double_prime_i = -mol_k * (mol_k * exp(mol_k * (rho_i - threshold_r)) * 
                          (1 + exp(mol_k * (rho_i - threshold_r))) - 2 * exp(mol_k * (rho_i - threshold_r)) * 
                          exp(mol_k * (rho_i - threshold_r))) / 
                          ((1 + exp(mol_k * (rho_i - threshold_r))) * 
                          (1 + exp(mol_k * (rho_i - threshold_r))) * 
                          (1 + exp(mol_k * (rho_i - threshold_r))));
        mol_i = 1;
        mol_grad_i = 0;
        mol_double_prime_i = 0;
        if (rho_i > threshold_r) {
            mol_i = 0;
            mol_grad_i = 0;
            mol_double_prime_i = 0;
        }
        if (std::abs(mol_i) >= 0.000001) {
            particles_nonzero_mol.push_back(i);
        }
    }

    // create new list of neighbors with only nonzero mol particles
    // sort neighbors list 
    

    //n = 5 + 2;
    //tbb::parallel_for(0, n, [&](int i) {
    int i;
    int j_idx;
    std::cout << "Particles nonzero mol size: " << particles_nonzero_mol.size() << std::endl;
    for (std::vector<int>::iterator it = particles_nonzero_mol.begin(); it < particles_nonzero_mol.end(); it++) {
        i = *it;
        rho_i = densities(i);
        mol_i = 1 / (1 + exp(mol_k * (rho_i - threshold_r)));
        mol_grad_i = -mol_k * exp(mol_k * (rho_i - threshold_r)) / 
                     ((1 + exp(mol_k * (rho_i - threshold_r))) * (1 + exp(mol_k * (rho_i - threshold_r))));
        mol_double_prime_i = -mol_k * (mol_k * exp(mol_k * (rho_i - threshold_r)) * 
                          (1 + exp(mol_k * (rho_i - threshold_r))) - 2 * exp(mol_k * (rho_i - threshold_r)) * 
                          exp(mol_k * (rho_i - threshold_r))) / 
                          ((1 + exp(mol_k * (rho_i - threshold_r))) * 
                          (1 + exp(mol_k * (rho_i - threshold_r))) * 
                          (1 + exp(mol_k * (rho_i - threshold_r))));
    for (int j = 0; j < neighbors[i].size(); j++) {
        j_idx = neighbors[i][j];
        //TODO: loop over neighbors of i instead
        rho_j = densities(j_idx);
        u = rho_i - rho_j;

        if (!smooth_mol) {
            mol_i = 1;
            mol_grad_i = 0;
            mol_double_prime_i = 0;
            if (densities(i) > threshold_r) {
                mol_i = 0;
                mol_grad_i = 0;
                mol_double_prime_i = 0;
            }
        }

        //TODO: std::set of neighbors i and j, k and l would be members of the set
        // take union of sorted loop
        // std::set<int> neighbors_i(neighbors[i].begin(), neighbors[i].end());
        // std::set<int> neighbors_j(neighbors[j_idx].begin(), neighbors[j_idx].end());
        std::vector<int> neighbors_union(neighbors[i].size() + neighbors[j_idx].size());
        std::vector<int>::iterator union_it = std::set_union(neighbors[i].begin(), neighbors[i].end(), neighbors[j_idx].begin(), neighbors[j_idx].end(), neighbors_union.begin());
        neighbors_union.resize(union_it - neighbors_union.begin());

        for (std::vector<int>::iterator k_it=neighbors_union.begin(); k_it!=neighbors_union.end(); ++k_it) {
            int k = *k_it;
            w_k = Eigen::Matrix<double, dim, 1>::Zero();
            z_k = Eigen::Matrix<double, dim, 1>::Zero();
            w_diff_k = Eigen::Matrix<double, dim, 1>::Zero();
            w_l = Eigen::Matrix<double, dim, 1>::Zero();
            z_l = Eigen::Matrix<double, dim, 1>::Zero();
            w_diff_l = Eigen::Matrix<double, dim, 1>::Zero();
            hess_rho_diff = Eigen::Matrix<double, dim, dim>::Zero();
            term1 = Eigen::Matrix<double, dim, dim>::Zero();
            term2 = Eigen::Matrix<double, dim, dim>::Zero();

            w_k = B.block<1, dim>(i, k * dim).transpose();
            z_k = B.block<1, dim>(j_idx, k * dim).transpose();
            w_diff_k = w_k - z_k;

            for (std::vector<int>::iterator l_it=neighbors_union.begin(); l_it!=neighbors_union.end(); ++l_it) {
                int l = *l_it;
                w_l = B.block<1, dim>(i, l * dim).transpose();
                z_l = B.block<1, dim>(j_idx, l * dim).transpose();

                w_diff_l = w_l - z_l;
                const auto& xk = x.template segment<dim>(dim * k);
                const auto& xl = x.template segment<dim>(dim * l);
                double x_kl = (xk - xl).norm();

                // Eigen::Matrix<double, dim, dim> hess_rho_i_kl;  // Hessian of rho_i with respect to x_k a
                hess_rho_i_kl = density_hessian<dim>(x, neighbors, i, k, l, h, m, fac, rho_0, B_sparse);

                // Eigen::Matrix<double, dim, dim> hess_rho_j_kl;  // Hessian of rho_j with respect to x_k and x_l
                hess_rho_j_kl = density_hessian<dim>(x, neighbors, j_idx, k, l, h, m, fac, rho_0, B_sparse);

                hess_rho_diff = hess_rho_i_kl - hess_rho_j_kl;  // Hessian difference

                // First term
                term1 = kappa * (w_diff_l * mol_i * w_diff_k.transpose() + u * mol_grad_i * w_l * w_diff_k.transpose() + u * mol_i * hess_rho_diff);

                // Second term
                term2 = 0.5 * kappa * (2 * u * w_diff_l * mol_grad_i * w_k.transpose() + u * u * (mol_double_prime_i * w_l * w_k.transpose() + mol_grad_i * hess_rho_i_kl));

                // Add the terms to the Hessian
                if (k == l) {
                    hess_subblock = term1 + term2;
                    hessian.block<dim, dim>(dim * k, dim * l) += ipc::project_to_psd(hess_subblock); //TODO: psd this if needed
                }
            }
        }
    }
    }



    std::cout << "Hessian norm: " << hessian.norm() << std::endl; 


    return hessian;
};

