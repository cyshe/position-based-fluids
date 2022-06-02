#include<iostream>
#include <igl/opengl/glfw/Viewer.h>

//#include<Eigen/Core>

int main(int argc, char *argv){

    // Inline mesh of a cube
  const Eigen::MatrixXd V= (Eigen::MatrixXd(8,3)<<
    0.0,0.0,0.0,
    0.0,0.0,1.0,
    0.0,1.0,0.0,
    0.0,1.0,1.0,
    1.0,0.0,0.0,
    1.0,0.0,1.0,
    1.0,1.0,0.0,
    1.0,1.0,1.0).finished();

  const Eigen::MatrixXd C = (Eigen::MatrixXd(1,3) << 0, 0, 1.0).finished();
  // Plot the mesh
  igl::opengl::glfw::Viewer viewer;
  viewer.data().set_points(V, C);
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