#pragma once

#include <Eigen/Core>
#include "cubic_bspline.h"
#include "calculate_densities.h"
#include <ipc/ipc.hpp>
#include <ipc/utils/eigen_ext.hpp>
#include <iostream> // remove
#include <ipc/distance/point_edge.hpp>
#include <ipc/distance/edge_edge.hpp>
#include <ipc/distance/edge_edge_mollifier.hpp>
#include <ipc/barrier/barrier.hpp>

template <int dim>
double bounds_energy(
    Eigen::VectorXd & x,
    const Eigen::Matrix<double, dim, 1> & lower_bound,
    const Eigen::Matrix<double, dim, 1> & upper_bound
){
    int n = x.size() / dim;
    Eigen::Vector2d e0, e1;

    double energy = 0.0;
    double d = 0.0;

    for (int i = 0; i < n; i++){
        const auto & point = x.template segment<dim>(dim * i);
        
        for (int j = 0; j < 4; j++) {
            if (j == 0) {
                e0 = lower_bound;
                e1 = Eigen::Vector2d(upper_bound(0), lower_bound(1));
            }
            else if (j == 1) {
                e0 = Eigen::Vector2d(upper_bound(0), lower_bound(1));
                e1 = upper_bound;
            }
            else if (j == 2) {
                e0 = upper_bound;
                e1 = Eigen::Vector2d(lower_bound(0), upper_bound(1));
            }
            else if (j == 3) {
                e0 = Eigen::Vector2d(lower_bound(0), upper_bound(1));
                e1 = lower_bound;
            }

            d = ipc::point_edge_distance(point, e0, e1);
            energy += ipc::barrier(d, 0.2);
        }
    }
    return energy;
};


template <int dim>
Eigen::VectorXd bounds_gradient(
    Eigen::VectorXd & x,
    const Eigen::Matrix<double, dim, 1> & lower_bound,
    const Eigen::Matrix<double, dim, 1> & upper_bound
){
    int n = x.size() / dim;
    Eigen::Vector2d e0, e1;
    Eigen::VectorXd grad = Eigen::VectorXd::Zero(x.size());
    double d = 0.0;

    for (int i = 0; i < n; i++){
        const auto & point = x.template segment<dim>(dim * i);
        
        for (int j = 0; j < 4; j++) {
            if (j == 0) {
                e0 = lower_bound;
                e1 = Eigen::Vector2d(upper_bound(0), lower_bound(1));
            }
            else if (j == 1) {
                e0 = Eigen::Vector2d(upper_bound(0), lower_bound(1));
                e1 = upper_bound;
            }
            else if (j == 2) {
                e0 = upper_bound;
                e1 = Eigen::Vector2d(lower_bound(0), upper_bound(1));
            }
            else if (j == 3) {
                e0 = Eigen::Vector2d(lower_bound(0), upper_bound(1));
                e1 = lower_bound;
            }
        
            d = ipc::point_edge_distance(point, e0, e1);
            grad.template segment<dim>(dim * i) += 
                ipc::point_edge_distance_gradient(point, e0, e1).template segment<dim>(0) * 
                ipc::barrier_gradient(d, 0.2);
        }
    }

    return grad;
};

template <int dim>
Eigen::MatrixXd bounds_hessian(
    Eigen::VectorXd & x,
    const Eigen::Matrix<double, dim, 1> & lower_bound,
    const Eigen::Matrix<double, dim, 1> & upper_bound
){
    int n = x.size() / dim;
    Eigen::Vector2d e0, e1;
    Eigen::MatrixXd hessian = Eigen::MatrixXd::Zero(x.size(), x.size());
    double d = 0.0;

    for (int i = 0; i < n; i++){
        const auto & point = x.template segment<dim>(dim * i);

        for (int j = 0; j < 4; j++) {
            if (j == 0) {
                e0 = lower_bound;
                e1 = Eigen::Vector2d(upper_bound(0), lower_bound(1));
            }
            else if (j == 1) {
                e0 = Eigen::Vector2d(upper_bound(0), lower_bound(1));
                e1 = upper_bound;
            }
            else if (j == 2) {
                e0 = upper_bound;
                e1 = Eigen::Vector2d(lower_bound(0), upper_bound(1));
            }
            else if (j == 3) {
                e0 = Eigen::Vector2d(lower_bound(0), upper_bound(1));
                e1 = lower_bound;
            }
        
            d = ipc::point_edge_distance(point, e0, e1);
            hessian.block<dim, dim>(dim*i, dim*i) += ipc::barrier_gradient(d, 0.05)
                * ipc::point_edge_distance_hessian(point, e0, e1).template block<dim, dim>(0, 0);


            Eigen::Vector<double, dim> g = ipc::point_edge_distance_gradient(point, e0, e1).template segment<dim>(0);
                 
            hessian.block<dim, dim>(dim*i, dim*i) += ipc::barrier_hessian(d, 0.05) * (g * g.transpose());
        }   
    }
    
    return hessian;
};