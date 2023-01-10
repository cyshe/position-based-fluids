#include "animate_implicit.h"
#include "calculate_lambda.h"
#include <igl/signed_distance.h>
#include <Eigen/Core>
#include <iostream>

#define _USE_MATH_DEFINES
#include <math.h>

using namespace Eigen;

template <>
void animate_implicit<2>(
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
    double kappa = 1000;//100000;
    double rho_0 = 1; //define later
    double m = 1;
    double h = 0.1;
    X_half.resize(numofparticles, 2);
    V_new.resize(numofparticles, 2);
    a_half.resize(numofparticles, 2);
    rho.resize(numofparticles, 1);
    p.resize(numofparticles, 1);

    MatrixXd f_ext(numofparticles, 2);
    f_ext.setZero();
    f_ext.col(1).setConstant(-9.8);

    X_half = X + dt/2 * V+ dt * dt/8 * f_ext;
    a_half.setZero();
    // use x_half to calculate density rho
    double fac = 10.0 / M_PI / std::pow(h,2)/7.0;    


    for (int i = 0; i < numofparticles; i++){
        for (int j = 0; j < numofparticles; j++){
            // rho(i) += m * W<2>(X_half.row(i) - X_half.row(j), h);
            if (i != j){
                double r = (X_half.row(i) - X_half.row(j)).norm();
                double q = r / h;
                if (q <= 1 && q > 0){
                    rho(i) += m * fac * (1 - 1.5* q * q * (1-q/2));
                }
                else if (q > 1 && q <= 2){
                    rho(i) += m * (fac / 4) * (2 - q) * (2 - q) * (2 - q);
                }
                else{
                    rho(i) += 0;
                }
            }
        }
    }
    std::cout << rho << std::endl;

    // use x_half and rho to calculate pressure p
    for (int i = 0; i < numofparticles; i++){
        p(i) = (rho_0/rho(i) - 1) * kappa;
    }

    std::cout <<"p = " <<p(5) << std::endl;
    // TODO 
    // Double check that acceleration formula is right
    // Switched to 2D Cubic b-spline
    //      (https://pysph.readthedocs.io/en/latest/reference/kernels.html)


    // calulate x_half with rho and p 
    for (int i = 0; i < numofparticles; i++){
        for(int j = 0; j < numofparticles; j++){

            
            double r = (X_half.row(i) - X_half.row(j)).norm();
            double q = r / h;

            if (q <= 1 && q > 0){
                //a_half.row(i) -= m * (p(j) + p(i))/rho(j) /rho(j) 
                //    * (fac * std::pow(h-r,2)
                //    * (X_half.row(i) - X_half.row(j)) / r) / rho_0; 
                //
                // TODO double check for negative...
                a_half.row(i) -= m * (p(j)/(rho(j)*rho(j)) + p(i)/(rho(i)*rho(i)))
                    * (fac * (9 * q * q/ 4 - 3 * q)
                    * (X_half.row(i) - X_half.row(j))/r); 
            }
            else if (q > 1 && q <= 2){
                a_half.row(i) -= m * (p(j)/(rho(j)*rho(j)) + p(i)/(rho(i)*rho(i)))
                    * ((-3 * fac / 4 * (2 - q) * (2 - q))
                    * (X_half.row(i) - X_half.row(j))/r);
            }
        }
    }

    //std::cout << a_half << std::endl;
    V_new = V + dt * a_half;
    

    X += dt/2 * (V + V_new) + dt * dt/2 * f_ext;
    V = V_new;
    
    for (int i = 0; i < numofparticles; i++){
        for (int j = 0; j < 2; ++j) {
            if (X(i,j) < low_bound(j)) {
                V(i, j) = abs(V(i, j));
                X(i,j) = low_bound(j);
            }
            if (X(i,j) > up_bound(j)) {
                V(i, j) = -1* abs(V(i, j));
                X(i,j) = up_bound(j);
            }
        }
    }
    
    return;
}

template <>
void animate_implicit<3>(
    MatrixXd & X, 
    MatrixXd & V, 
    MatrixXi & N,
    const Eigen::Matrix<double, 3, 1> & low_bound,
    const Eigen::Matrix<double, 3, 1> & up_bound,
    const int numofparticles,
    const int iters, 
    const double dt
    ){}
