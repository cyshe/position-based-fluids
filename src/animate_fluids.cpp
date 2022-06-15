#include "animate_fluids.h"
#include <iostream>

void animate_fluids(Eigen::MatrixXd & X, double iters){
    //std::cout << iters << std::endl;
    
    for (int i = 0; i < iters; i++){
        X(0,0) += 0.01;
        //std::cout << i << std::endl;
        //predict_position(X, v, f_ext, dt);
        // find_neighbor(x,N);
        //forces();
        //energy_refinement();
        //update_velocity();
    }
    return;
}