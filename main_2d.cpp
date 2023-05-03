#include "animate_sph.h"
#include "animate_fluids.h"
#include "animate_implicit.h"
#include "polyscope/polyscope.h"
#include "polyscope/point_cloud.h"
#include "polyscope/curve_network.h"

#include <iostream>
#include <igl/writeOBJ.h>
#include <Eigen/Core>
#include <sstream>
#include <cmath>
#include <igl/grid.h>

#define _USE_MATH_DEFINES
#include <math.h>

using namespace Eigen;

polyscope::PointCloud* psCloud;

MatrixXd q0, q, q_dot;  // particle positions, velocities
MatrixXi N;             // Per-particle neighbors
VectorXd J;

int numofparticles; //number of particles

// Boundary extents
Vector2d lower_bound;
Vector2d upper_bound;

int iters = 5;
double dt = 0.03;

void callback() {

  static bool is_simulating = false; static bool write_sequence = false;
  static int frame = 0;

  ImGui::PushItemWidth(100);

  // Export particles to OBJ
  ImGui::Checkbox("Write OBJ", &write_sequence);

  if (write_sequence) {
    std::stringstream buffer;
    Eigen::MatrixXd F;
    buffer << "./Sequence/seq_" << std::setfill('0') << std::setw(3) << frame << ".obj";
    std::string file = buffer.str();
    std::cout << file << std::endl;
    igl::writeOBJ(file,q, F);
  }

  ImGui::InputInt("Solver Iterations", &iters);

  // Perform simulation step
  ImGui::Checkbox("Simulate", &is_simulating);
  ImGui::SameLine();
  if (ImGui::Button("One Step") || is_simulating) {
    //animate_sph<2>(q, q_dot, N, lower_bound, upper_bound, numofparticles, iters, dt);
    //animate_fluids<2>(q, q_dot, N, lower_bound, upper_bound, numofparticles, iters, dt);
    animate_implicit<2>(q, q_dot, J, N, lower_bound, upper_bound, numofparticles, iters, dt);
    psCloud->updatePointPositions2D(q);
    psCloud->addVectorQuantity("velocity", q_dot, polyscope::VectorType::STANDARD)->setEnabled(true);
    ++frame;
    // std::cout <<"X = " << q << std:: endl;
    std::cout << frame << std::endl;
    //polyscope::screenshot();
  }

  // Resets the simulation
  if (ImGui::Button("Reset")) {
    q = q0;
    q_dot.setZero();
    J.setConstant(1);
    psCloud->updatePointPositions2D(q);
    frame = 0;
  }
}

int main(int argc, char *argv[]){

  // Options
  polyscope::options::autocenterStructures = false;
  polyscope::view::windowWidth = 1024;
  polyscope::view::windowHeight = 1024;

  
  // Initialize polyscope
  polyscope::init();
  polyscope::view::style = polyscope::view::NavigateStyle::Planar;
  // Add the callback
  polyscope::state::userCallback = callback;

  // Init bounding box
  lower_bound << -1.0, -1.0;
  upper_bound << 4.366, 2.0; 

  // Initialize positions
  double l = 15;
  numofparticles = l* 2 *l;

  // (-1, -1), (1.2, 0.1) 800 particles
  Eigen::Vector2d res(2*l,l);
  igl::grid(res,q);
  q.array() = 0.55 * 2 * (q.array() - 0.5);
  q.col(0) = q.col(0) * 2;

  q.col(0) =  q.col(0) + Eigen::MatrixXd(numofparticles,1).setConstant(0.1);
  q.col(1) =  q.col(1) - Eigen::MatrixXd(numofparticles,1).setConstant(0.45);
  //std::cout << q.row(0) << q.row(399) << std::endl;
  // 
  // q.resize(numofparticles, 3);
  // q.setZero();
  // for (int i = 0; i < numofparticles; i++){
  //   q(i, 0) = 0.1 * (i%10) - 0.5;
  //   q(i, 1) = 0.1 * (int(floor(i/10.0)) % 10);
  //   q(i, 2) = 0.1 * floor(i/100) -0.5;
  // }
  q0 = q; // initial positions
  q_dot.resize(numofparticles, 2);
  q_dot.setZero();
  
  J.resize(numofparticles, 1);
  J.setConstant(1);

  // Initialize J values
  double h = 0.1;
  double rho_0 = 1;
  double fac = 10/7/M_PI;

  J.setZero();
  for (int i = 0; i < numofparticles; i++){
    for (int j = 0; j < numofparticles; j++){
        double r = (q.row(j) - q.row(i)).norm()/h;
        J(i) += cubic_bspline(r, fac) / rho_0;
    }
  }

  
  std::cout << "Stop initializing J this way" << std::endl;
  std::cout << "initializing J with h = " << h << " and rho_0 = " << rho_0 << std::endl;
  std::cout << "J first 20 " << J.head(20) << std::endl;
  std::cout << "Jx = " << J(5) << " " << J(16) << " " << J(24)  << std::endl;
  //std::cout << " J * rho: " << J.transpose() * rho_0 << std::endl;

  // Create point cloud polyscope object
  psCloud = polyscope::registerPointCloud2D("particles", q);
  //psCloud->addVectorQuantity("velocity", q, polyscope::VectorType::STANDARD);
  // set some options
  psCloud->setPointRadius(0.005);

  // If rendering becomes slow, enable this
  // psCloud->setPointRenderMode(polyscope::PointRenderMode::Quad);

  // Visualize the bounding box
  MatrixXd V_bbox(4,2);
  V_bbox << lower_bound(0), lower_bound(1),
            upper_bound(0), lower_bound(1),
            upper_bound(0), upper_bound(1),
            lower_bound(0), upper_bound(1);

  MatrixXi E_bbox(4,2);
  E_bbox << 0, 1,
            1, 2,
            2, 3,
            3, 0;

  polyscope::registerCurveNetwork2D("boundary", V_bbox, E_bbox);
  polyscope::getCurveNetwork("boundary")->setColor(glm::vec3(1.0,0.0,0.0));
  polyscope::getCurveNetwork("boundary")->setRadius(0.001);


  // Take a screenshot
  //polyscope::screenshot();

  // Show the gui
  polyscope::show();
  return 0;
}
