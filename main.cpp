#include "animate_fluids.h"
#include <iostream>
#include <igl/opengl/glfw/Viewer.h>
#include <Eigen/Core>

int main(int argc, char *argv){

  igl::opengl::glfw::Viewer viewer;

  //initial conditions
  Eigen::MatrixXd V= (Eigen::MatrixXd(8,3)<<
    0.0,0.0,0.0,
    0.0,0.0,1.0,
    0.0,1.0,0.0,
    0.0,1.0,1.0,
    1.0,0.0,0.0,
    1.0,0.0,1.0,
    1.0,1.0,0.0,
    1.0,1.0,1.0).finished();

  Eigen::MatrixXd C = (Eigen::MatrixXd(1,3) << 0, 0, 1.0).finished();
  int iters = 10;
  
  /*
  const auto update = [&]()
  {
    viewer.data().set_points(V, C);
  };
*/
  const auto step = [&]()
  {
    // animation
    animate_fluids(V, iters);
    viewer.data().set_points(V, C);
  };

  viewer.data().set_points(V, C);
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
//    Eigen::MatrixXd N;
  //  double dt = 0.001;
    
    
    //predict_position(x, v, f_ext, dt);
   // find_neighbor(x,N);
    //forces();
    //energy_refinement();
    //update_velocity();

    //return 0;
}