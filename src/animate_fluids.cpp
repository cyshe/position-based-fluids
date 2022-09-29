#include "forces.h"
#include "predict_position.h"
#include "animate_fluids.h"
#include "find_neighbor.h"
#include "energy_refinement.h"

#include <igl/signed_distance.h>
#include <Eigen/Core>
#include <iostream>

using namespace Eigen;

void animate_fluids(
    MatrixXd & X, 
    MatrixXd & V, 
    MatrixXi & N,
    const Vector3d & low_bound,
    const Vector3d & up_bound,
    const int numofparticles,
    const int iters, 
    const double dt
    ){
        
    double cell_size = 0.1;

    MatrixXd X_star = X;

    MatrixXd f_ext(X.rows(), 3);
    f_ext.setZero();
    f_ext.col(1).setConstant(-9.8);
    std::cout << "X* " << X_star.row(65) << std::endl;
    
    predict_position(X_star, V, f_ext, dt);
    std::cout << "predict_pos" << X_star.row(65) << std::endl;
    
    find_neighbor(X_star, low_bound, up_bound, cell_size, numofparticles, N);
    //std::cout << N << std::endl << std::endl << std::endl;

    for (int i = 0; i < iters; i++){
        forces(X_star, N, numofparticles);
    }

    V = (X_star - X)/dt;
    energy_refinement(X_star, N, V, numofparticles, 0.2, dt);

    // Enforce box boundary conditions
    for (int i = 0; i < numofparticles; i++){
        for (int j = 0; j < 3; ++j) {
            if (X_star(i,j) < low_bound(j)) {
                V(i, j) = 0.1 * abs(V(i, j));
                X_star(i,j) = low_bound(j);
            }
            if (X_star(i,j) > up_bound(j)) {
                V(i, j) = -0.1 * abs(V(i, j));
                X_star(i,j) = up_bound(j);
            }
        }
    }

   

    X = X_star;
    std::cout <<  X.row(65) << std::endl;
    return;
}