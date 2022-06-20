#include "forces.h"
#include "predict_position.h"
#include "animate_fluids.h"
#include "find_neighbor.h"

#include <Eigen/Core>
#include <iostream>

void animate_fluids(
    Eigen::MatrixXd & X, 
    Eigen::MatrixXd & V, 
    Eigen::MatrixXd & N,
    const Eigen::Vector3d & low_bound,
    const Eigen::Vector3d & up_bound,
    const int numofparticles,
    const int iters, 
    const double dt
    ){

    double cell_size = 0.1;

    Eigen::MatrixXd f_ext;
    f_ext.resize(X.rows(), 3);
    f_ext.setZero();
    for (int i = 0; i < X.rows(); i++){
        f_ext(i,1) = -9.8;
    }

    
    predict_position(X, V, f_ext, dt);
    find_neighbor(X, low_bound, up_bound, cell_size, numofparticles, N);

    for (int i = 0; i < iters; i++){
        forces(X, N, numofparticles);
        
        //energy_refinement();
        
    }
    std::cout << X << std::endl;
    return;
}