#include "vorticity_confinement.h"
#include <Eigen/Core>
#include <Eigen/Geometry>


void vorticity_confinement(
    const Eigen::MatrixXd & x,
    const Eigen::MatrixXd & v, 
    Eigen::MatrixXd & f,
    const int numofparticles, 
    const double h
    ){

    double epsilon = 0.0001;

    for (int i=0; i < numofparticles; i++){
        Eigen::Vector3d omega_i;
        omega_i.setZero();

        for (int j = 0; j < numofparticles; j++){
            Eigen::Vector3d v_ij, eta, r;
            double a;
            
            r = x.row(i) -x.row(j);

            v_ij = v.row(j) - v.row(i);
            a = -45 * pow(h - r.norm(), 2) / (3.14 * pow(h, 6))* r.norm();

            Eigen::Vector3d x_j = x.row(j);
            omega_i = a * v_ij.cross(x_j);

            eta << (v_ij(2) * v_ij(2) * r(0) - v_ij(0) * v_ij(2) * r(2) - v_ij(0) * v_ij(1) * r(1) + v_ij(1) * v_ij(1) * r(0)),  
                  (v_ij(0) * v_ij(0) * r(1) - v_ij(1) * v_ij(0) * r(0) - v_ij(1) * v_ij(2) * r(2) + v_ij(2) * v_ij(2) * r(1)),
                  (v_ij(1) * v_ij(1) * r(2) - v_ij(2) * v_ij(1) * r(1) - v_ij(2) * v_ij(0) * r(0) + v_ij(0) * v_ij(0) * r(2));

            eta = eta * abs(a)/omega_i.norm();
            eta.normalized();

            Eigen::Vector3d f_i;
            f_i = eta.cross(omega_i);

            f.row(i) = f_i * epsilon;
        }
        
    }
    return;
}