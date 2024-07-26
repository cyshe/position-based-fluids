#pragma once

#include <Eigen/Core>
#include "cubic_bspline.h"
#include "calculate_densities.h"
#include <ipc/ipc.hpp>
#include <ipc/utils/eigen_ext.hpp>

#include <ipc/utils/logger.hpp>
#include <iostream>
#include <ipc/distance/point_edge.hpp>
#include <ipc/distance/edge_edge.hpp>
#include <ipc/distance/edge_edge_mollifier.hpp>
#include <ipc/barrier/barrier.hpp>

#include <ipc/candidates/vertex_vertex.hpp>
#include <ipc/candidates/edge_vertex.hpp>
#include <ipc/candidates/face_vertex.hpp>
#include <ipc/candidates/edge_face.hpp>
#include <ipc/utils/save_obj.hpp>


template <int dim>
double bounds_energy(
    const Eigen::VectorXd & x,
    const Eigen::Matrix<double, dim, 1> & lower_bound,
    const Eigen::Matrix<double, dim, 1> & upper_bound,
    const double barrier_width,
    const double kappa
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
            energy += kappa * ipc::barrier(d, barrier_width);
        }
    }
    return energy;
};


template <int dim>
Eigen::VectorXd bounds_gradient(
    Eigen::VectorXd & x,
    const Eigen::Matrix<double, dim, 1> & lower_bound,
    const Eigen::Matrix<double, dim, 1> & upper_bound,
    const double barrier_width,
    const double kappa
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
                ipc::barrier_gradient(d, barrier_width);
        }
    }

    grad = kappa * grad;

    return grad;
};

template <int dim>
Eigen::MatrixXd bounds_hessian(
    Eigen::VectorXd & x,
    const Eigen::Matrix<double, dim, 1> & lower_bound,
    const Eigen::Matrix<double, dim, 1> & upper_bound,
    const double barrier_width,
    const double kappa
){
    int n = x.size() / dim;
    Eigen::Vector2d e0, e1;
    Eigen::MatrixXd hessian = Eigen::MatrixXd::Zero(x.size(), x.size());
    double d = 0.0;
    Eigen::Matrix<double, dim, dim> hess = Eigen::Matrix<double, dim, dim>::Zero();

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

            hess = ipc::barrier_gradient(d, barrier_width) * ipc::point_edge_distance_hessian(point, e0, e1).template block<dim, dim>(0, 0);

            
            //hessian.block<dim, dim>(dim*i, dim*i) += ipc::project_to_psd(hess);
            hessian.block<dim, dim>(dim*i, dim*i) += hess;


            Eigen::Vector<double, dim> g = ipc::point_edge_distance_gradient(point, e0, e1).template segment<dim>(0);
                 
            hess = ipc::barrier_hessian(d, barrier_width) * (g * g.transpose());
            //hessian.block<dim, dim>(dim*i, dim*i) += ipc::project_to_psd(hess);
            hessian.block<dim, dim>(dim*i, dim*i) += hess;
        }   
    }
    hessian = kappa * hessian;

    return hessian;
};