#include "animate_fluids.h"
#include <iostream>
#include <igl/opengl/glfw/Viewer.h>
#include <Eigen/Core>

int main(int argc, char *argv){

  igl::opengl::glfw::Viewer viewer;

  
  //initial conditions
  int numofparticles = 100;
  Eigen::Vector3d lower_bound;
  Eigen::Vector3d upper_bound;

  lower_bound << -2.0, -2.0, -3.0;
  upper_bound << 2.0, 2.0, 2.0; 

  Eigen::MatrixXd q = Eigen::MatrixXd::Random(numofparticles, 3);
  
  
  Eigen::MatrixXd q_dot;
  q_dot.resize(q.rows(), 3);
  q_dot.setZero();

  Eigen::MatrixXd N;
  N.resize(numofparticles, numofparticles);

  Eigen::MatrixXd C = (Eigen::MatrixXd(1,3) << 0, 0, 1.0).finished();
  
  int iters = 10;
  double dt = 0.01;

  /*
  const auto update = [&]()
  {
    viewer.data().set_points(X, C);
  };
*/
  const auto step = [&]()
  {
    // animation
    animate_fluids(q, q_dot, N, lower_bound, upper_bound, numofparticles, iters, dt);
    viewer.data().set_points(q, C);
  };

  viewer.data().set_points(q,C);
  viewer.core().is_animating = false;
  viewer.callback_pre_draw = [&](igl::opengl::glfw::Viewer & )->bool
  {
    step();
    return false;
  };
  viewer.launch();

    //input list of positions
//    Eigen::MatrixXd v;
  //  Eigen::MatrixXd x;
    //Eigen::MatrixXd f_ext;
//     Eigen::MatrixXd N;
  //  double dt = 0.001;
    
    
    //predict_position(x, v, f_ext, dt);
   // find_neighbor(x,N);
    //forces();
    //energy_refinement();
    //update_velocity();

    //return 0;
}