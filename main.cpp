#include<iostream>
#include<Eigen/Core>

int main(int argc, char *argv){

    //input list of positions
    Eigen::MatrixXd v;
    Eigen::MatrixXd x;
    Eigen::MatrixXd f_ext;
    Eigen::MatrixXd N;
    double dt = 0.001;
    
    
    predict_position(x, v, f_ext, dt);
    find_neighbor(x,N);
    forces();
    energy_refinement();
    update_velocity();

    return 0;
}