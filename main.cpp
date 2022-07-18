#include "animate_fluids.h"
#include <iostream>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/writeOBJ.h>
#include <Eigen/Core>
#include <sstream>

int main(int argc, char *argv){

  igl::opengl::glfw::Viewer viewer;

  
  //initial conditions
  int numofparticles = 200;
  Eigen::Vector3d lower_bound;
  Eigen::Vector3d upper_bound;

  lower_bound << -6.0, -6.0, -6.0;
  upper_bound << 6.0, 6.0, 6.0; 

  Eigen::MatrixXd q;
  q.resize(numofparticles, 3);
  q.setRandom();
  q =  q * 0.5;


  Eigen::MatrixXd q_dot;
  q_dot.resize(numofparticles, 3);
  q_dot.setZero();

  //std::cout << q << std::endl;

  Eigen::MatrixXd N;
  N.resize(numofparticles, numofparticles);

  Eigen::MatrixXd C = (Eigen::MatrixXd(1,3) << 0, 0, 1.0).finished();
  
  int iters = 3;
  double dt = 0.1;

  /*
  const auto update = [&]()
  {
    viewer.data().set_points(q, C);
  };
  */
  Eigen::MatrixXd V_bound;
    V_bound.resize(12, 3);
    V_bound << 2.0, 1.0, 2.0,
             2.0, 1.0, -2.0,
            -2.0, 1.0, -2.0,
            -2.0, 1.0, 2.0,
             1.8,-2.5, 1.8,
             1.8,-2.5, -1.8,
            -1.8,-2.5, -1.8,
            -1.8,-2.5, 1.8,
             3.0,-4.0,  3.0,
             3.0,-4.0, -3.0,
            -3.0,-4.0, -3.0,
            -3.0,-4.0,  3.0;
  
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
  int frame = 0;
  Eigen::MatrixXd F;

  const auto step = [&]()
  {
    // animation
    animate_fluids(q, q_dot, N, lower_bound, upper_bound, numofparticles, iters, dt);
    viewer.data().set_points(q, C);
    
    std::stringstream buffer;
    buffer << "./Sequence/seq_" << std::setfill('0') << std::setw(3) << frame << ".obj";
    std::string file = buffer.str();
    std::cout << file << std::endl;
    igl::writeOBJ(file,q, F);
    frame ++;
  };

  
 // viewer.core().is_animating = false;

  viewer.callback_pre_draw = [&](igl::opengl::glfw::Viewer & )->bool
  {
    step();
    return false;
  };
  viewer.data().set_points(q,C);
 // viewer.data().set_mesh(V_bound, F_bound);
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