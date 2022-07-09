#include "forces.h"
#include "predict_position.h"
#include "animate_fluids.h"
#include "find_neighbor.h"
#include "energy_refinement.h"

#include <igl/signed_distance.h>
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

    Eigen::MatrixXd X_star;
    X_star.resize(numofparticles, 3);
    X_star = X;

    Eigen::MatrixXd f_ext;
    f_ext.resize(X.rows(), 3);
    f_ext.setZero();
    for (int i = 0; i < X.rows(); i++){
        f_ext(i,1) = -9.8;
    }

    //collision detection
    
    igl::SignedDistanceType type = igl::SIGNED_DISTANCE_TYPE_PSEUDONORMAL;
    
    Eigen::VectorXd S;
    Eigen::VectorXd I;
    Eigen::MatrixXd C;
    Eigen::MatrixXd Normals;

    S.resize(numofparticles);
    I.resize(numofparticles);
    C.resize(numofparticles, 3);
    Normals.resize(numofparticles, 3);

    Eigen::MatrixXd V_bound;
    V_bound.resize(12, 3);
    V_bound << 2.0, 1.0, 2.0,
             2.0, 1.0, -2.0,
            -2.0, 1.0, -2.0,
            -2.0, 1.0, 2.0,
             0.0, 0.0, 0.0,
             1.8,-2.5, -1.8,
            -1.8,-2.5, -1.8,
            -1.8,-2.5, 1.8,
             2.0,-3.0,  2.0,
             2.0,-3.0, -2.0,
            -2.0,-3.0, -2.0,
            -2.0,-3.0,  2.0;
  
    Eigen::MatrixXi F_bound = (Eigen::MatrixXi(20, 3) << 1, 2, 5,
            2, 6, 5,
            2, 3, 6,
            3, 7, 6,
            3, 4, 7,
            4, 8, 7,
            4, 1, 8,
            1, 5, 8,
            2, 1, 9,
            2, 9, 10,
            3, 2, 10,
            3, 10, 11,
            4, 3, 12,
            3, 11, 12,
            4, 12, 9,
            4, 9, 1,
            6, 7, 8,
            5, 6, 8,
            9, 12, 11,
            9, 11, 10).finished().array()-1;

    
    predict_position(X_star, V, f_ext, dt);
    
    
    
    
    find_neighbor(X_star, low_bound, up_bound, cell_size, numofparticles, N);

    //std::cout << N << std::endl << std::endl << std::endl;

    for (int i = 0; i < iters; i++){
        forces(X_star, N, numofparticles);
    }

    V = (X_star - X)/dt;
    energy_refinement(X_star, V, numofparticles, 0.15, dt);



    igl::signed_distance(X_star, V_bound, F_bound,type,S,I,C,Normals);
    
    //add constraint to solver V
    for (int i = 0; i < numofparticles; i++){
        if (S(i) <= 0){
            X_star(i, 0) = C(i, 0);
            X_star(i, 1) = C(i, 1);
            X_star(i, 2) = C(i, 2);

            //Eigen::Vector3d v_i = V.row(i);
            V.row(i) = V.row(i) - (V.row(i).dot(N.row(i).transpose()) * N.row(i));
        }
    }

    

    X = X_star;

    //std::cout << "forces" << X.row(0) << std::endl;
    return;
}