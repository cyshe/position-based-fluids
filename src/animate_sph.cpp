#include "animate_sph.h"
#include "calculate_lambda.h"
#include <igl/signed_distance.h>
#include <Eigen/Core>
#include <iostream>

#define _USE_MATH_DEFINES
#include <math.h>

using namespace Eigen;

template <>
void animate_sph<2>(
    MatrixXd & X, 
    MatrixXd & V, 
    MatrixXi & N,
    const Eigen::Matrix<double, 2, 1> & low_bound,
    const Eigen::Matrix<double, 2, 1> & up_bound,
    const int numofparticles,
    const int iters, 
    const double dt
    ){
    
    MatrixXd X_half, V_new, a_half;
    MatrixXd rho, p;
    double kappa = 100000;
    double rho_0 = 1; //define later
    double m = 1;
    double h = 0.1;
    X_half.resize(numofparticles, 2);
    V_new.resize(numofparticles, 2);
    a_half.resize(numofparticles, 2);
    rho.resize(numofparticles, 1);
    p.resize(numofparticles, 1);

    X_half = X + dt/2 * V;
    a_half.setZero();
    // use x_half to calculate density rho
    
    for (int i = 0; i < numofparticles; i++){
        for (int j = 0; j < numofparticles; j++){
            rho(i) += m * W<2>(X_half.row(i) - X_half.row(j), h);
        }
    }

    // use x_half and rho to calculate pressure p
    for (int i = 0; i < numofparticles; i++){
        p(i) = (rho_0/rho(i) -1) * kappa;
    }


    // calulate x_half with rho and p 
    for (int i = 0; i < numofparticles; i++){
        for(int j = 0; j < numofparticles; j++){

            double fac = -45.0 / M_PI / std::pow(h,6);
            double r = (X_half.row(i) - X_half.row(j)).norm();

            if (r <= h){
                a_half.row(i) -= m * (p(j) + p(i))/rho(j) /rho(j) * (fac * std::pow(h-r,2)
                * (X_half.row(i) - X_half.row(j)) / r) / rho_0; 
            }
        }
    }

    V_new = V + dt * a_half;


    X += dt/2 * (V + V_new);
    V = V_new;

    return;
}

template <>
void animate_sph<3>(
    MatrixXd & X, 
    MatrixXd & V, 
    MatrixXi & N,
    const Eigen::Matrix<double, 3, 1> & low_bound,
    const Eigen::Matrix<double, 3, 1> & up_bound,
    const int numofparticles,
    const int iters, 
    const double dt
    ){}