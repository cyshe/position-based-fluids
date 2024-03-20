#include "animate_sph.h"
#include "animate_fluids.h"
#include "animate_implicit.h"
#include "animate_lbfgs.h"
#include "polyscope/polyscope.h"
#include "polyscope/point_cloud.h"
#include "polyscope/curve_network.h"

#include <iostream>
#include <igl/writeOBJ.h>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <sstream>
#include <cmath>
#include <deque>
#include <igl/grid.h>
#include "ipc/ipc.hpp"
#include "cubic_bspline.h"
#include "calculate_densities.h"
#include "find_neighbors_brute_force.h"


#define _USE_MATH_DEFINES
#include <math.h>

using namespace Eigen;

polyscope::PointCloud* psCloud;

MatrixXd q0, q, q_dot; // particle positions, velocities
MatrixXi N;             // Per-particle neighbors
VectorXd J, Jx;
MatrixXd grad_i, grad_psi, grad_c, grad_s, grad_st;
SparseMatrix<double> A;

int numofparticles; //number of particles

// Boundary extents
Vector2d lower_bound;
Vector2d upper_bound;

int iters = 10;
double dt = 1;
double k_psi = 1;
double k_s = 10;
double k_st = 10;
double st_threshold = 2.0;
double rho_0 = 3.6;
double h = 0.2;
double fac = 10/7/M_PI;
bool resetA = true;

void callback() {

  static bool is_simulating = false; static bool write_sequence = false;
  static bool fd_check = false;
  static bool converge_check = true;
  static bool do_line_search = true;
  static bool bounds = true;
  static bool smooth_mol = false;
  static bool psi_bool = true;
  static bool spacing_bool = true;
  static bool st_bool = true;
  static bool primal = true;


  static int frame = 0;
  static double gravity = 0.0;

  ImGui::PushItemWidth(100);

  // Export particles to OBJ
  ImGui::Checkbox("Write OBJ", &write_sequence);

  ImGui::Checkbox("Finite Difference Check", &fd_check);

  if (write_sequence) {
    std::stringstream buffer;
    Eigen::MatrixXd F;
    buffer << "./Sequence/seq_" << std::setfill('0') << std::setw(3) << frame << ".obj";
    std::string file = buffer.str();
    std::cout << file << std::endl;
    igl::writeOBJ(file,q, F);
  }

  ImGui::Checkbox("Iterate until Convergence", &converge_check);
  ImGui::Checkbox("Do line search", &do_line_search);
  ImGui::Checkbox("Boundaries", &bounds);
  ImGui::Checkbox("Smooth Surface Tension", &smooth_mol);
  ImGui::Checkbox("psi", &psi_bool);
  ImGui::Checkbox("spacing", &spacing_bool);
  ImGui::Checkbox("surface tension", &st_bool);
  ImGui::Checkbox("primal", &primal);

  ImGui::InputInt("solver max iterations", &iters);
  ImGui::InputDouble("k_psi", &k_psi);
  ImGui::InputDouble("k_spacing", &k_s);
  ImGui::InputDouble("k_surface_tension", &k_st);
  ImGui::InputDouble("surface_tension_threshold", &st_threshold);
  ImGui::InputDouble("rho_0", &rho_0);
  ImGui::InputDouble("gravity", &gravity);

  // Perform simulation step
  ImGui::Checkbox("Simulate", &is_simulating);
  ImGui::SameLine();
  if (ImGui::Button("One Step") || is_simulating) {
    //animate_sph<2>(q, q_dot, N, lower_bound, upper_bound, numofparticles, iters, dt);
    //animate_fluids<2>(q, q_dot, N, lower_bound, upper_bound, numofparticles, iters, dt);
    grad_i.setZero(); grad_psi.setZero();
    grad_s.setZero();
    grad_st.setZero();

    animate_implicit<2>(q, q_dot, J, Jx, N, 
      grad_i, grad_psi, grad_s, grad_st,
      lower_bound, upper_bound, numofparticles, iters, dt, 
      k_psi, k_st, k_s, h, st_threshold, rho_0, gravity,
      fd_check, bounds, converge_check, do_line_search, smooth_mol, psi_bool, spacing_bool, st_bool, primal);
    resetA = (frame % 1 == 0);

    //animate_lbfgs<2>(q, q_dot, J, Jx, N,
    //  grad_i, grad_psi, grad_s, grad_st, A,
    //  lower_bound, upper_bound, numofparticles, iters, dt, 
    //  k_psi, k_st, k_s, h, st_threshold, rho_0, gravity,
    //  fd_check, bounds, converge_check, do_line_search, smooth_mol, psi_bool, spacing_bool, st_bool, primal, resetA);

    psCloud->updatePointPositions2D(q);
    psCloud->addVectorQuantity("total gradient", grad_i + grad_psi + grad_s + grad_st, polyscope::VectorType::STANDARD);
    psCloud->addVectorQuantity("inertia grad", grad_i, polyscope::VectorType::STANDARD);
    psCloud->addVectorQuantity("psi grad", grad_psi, polyscope::VectorType::STANDARD);
    psCloud->addVectorQuantity("spacing grad", grad_s, polyscope::VectorType::STANDARD)->setEnabled(true);
    psCloud->addVectorQuantity("surface tension grad", grad_st, polyscope::VectorType::STANDARD)->setEnabled(true);
    psCloud->addScalarQuantity("J", J)->setEnabled(true);
    
    VectorXd m = VectorXd::Ones(numofparticles);
    m = Jx * rho_0;
    double threshold = st_threshold * rho_0;
    // TODO: use mollifier function
    for (int i = 0; i < numofparticles; i++){
        double mollifier; 
        if (m(i) >  1.5 * threshold){
            mollifier = 1;
        }
        else if (m(i) > 0.75 * threshold) {
            mollifier =  - 16 * m(i) * m(i)/(9 * threshold * threshold) + 16 * m(i)/ (3 * threshold) - 3;
        }
        else{
            mollifier = 0;
        }
        mollifier =  1/ (1 + exp(200 * (m(i) - threshold)));
        m(i) = mollifier;
    }
    
    psCloud->addScalarQuantity("Mollifier", m)->setEnabled(true);


    ++frame;
    // std::cout <<"X = " << q << std:: endl;
    std::cout << frame << std::endl;
    polyscope::screenshot();
  }

  // Resets the simulation
  if (ImGui::Button("Reset")) {
    q = q0;
    q_dot.setZero();
    psCloud->updatePointPositions2D(q);
    frame = 0;
    MatrixXd X = q.transpose();
    VectorXd x = Eigen::Map<VectorXd>(X.data(), X.size());
    std::vector<std::vector<int>> neighbors = find_neighbors_brute_force<2>(x, h);
    J = calculate_densities<2>(x, neighbors, h, 1.0, fac);
    rho_0 = J.mean();
    J = J / rho_0;
    psCloud->addScalarQuantity("J", J)->setEnabled(true);

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
  lower_bound << -1.05, -1.05;
  upper_bound << 4.366, 2.0; 

  // Initialize positions

  double l = 15;

  //rectangle
  numofparticles = l* 2 *l;
/*
  // (-1, -1), (1.2, 0.1) 800 particles
  Eigen::Vector2d res(2*l,l);
  igl::grid(res,q);
  q.array() = 0.55 * 2 * (q.array() - 0.5);
  q.col(0) = q.col(0) * 2;
  q.col(0) =  q.col(0) + Eigen::MatrixXd(numofparticles,1).setConstant(0.1);
  q.col(1) =  q.col(1) - Eigen::MatrixXd(numofparticles,1).setConstant(0.45);
*/

  // square
  numofparticles = l * l;
  Eigen::Vector2d res(l,l);
  igl::grid(res,q);
  q.array() = 0.55 * 2 * q.array();

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


  // initialize gradients
  grad_i.resize(numofparticles, 3);
  grad_psi.resize(numofparticles, 3);
  grad_c.resize(numofparticles, 3);
  grad_s.resize(numofparticles, 3);
  grad_st.resize(numofparticles, 3);

  grad_i.setZero();
  grad_psi.setZero();
  grad_c.setZero();
  grad_s.setZero();
  grad_st.setZero();

  A.resize(numofparticles * 2, numofparticles * 2);

  // Initialize J values
  MatrixXd X = q.transpose();
  VectorXd x = Eigen::Map<VectorXd>(X.data(), X.size());
  std::vector<std::vector<int>> neighbors = find_neighbors_brute_force<2>(x, h);
  J = calculate_densities<2>(x, neighbors, h, 1.0, fac);
  rho_0 = J.mean();
  J = J / rho_0; 
  std::cout << "initializing J with h = " << h << " and rho_0 = " << rho_0 << std::endl;
  std::cout << "J first 20 " << J.head(20) << std::endl;
  std::cout << "Jx = " << J(5) << " " << J(16) << " " << J(24)  << std::endl;
  //std::cout << " J * rho: " << J.transpose() * rho_0 << std::endl;
  Jx.resize(numofparticles);
  Jx.setZero();
  // Create point cloud polyscope object
  

  psCloud = polyscope::registerPointCloud2D("particles", q);
  //psCloud->addVectorQuantity("velocity", q, polyscope::VectorType::STANDARD);
  // set some options
  psCloud->setPointRadius(0.005);

  psCloud->addScalarQuantity("J", J)->setEnabled(true);

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
  polyscope::screenshot();

  // Show the gui
  polyscope::show();
  return 0;
}
