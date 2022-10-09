#include "find_neighbor.h"
#include <map>
#include <set>
#include <tuple>
#include <iostream>
#include "igl/octree.h"
#include "igl/knn.h"

template <>
void find_neighbor<2>(const Eigen::MatrixXd & X, 
    const Eigen::Matrix<double, 2, 1> lower_bound,
    const Eigen::Matrix<double, 2, 1> upper_bound, 
    const double cell_size,
    const int numofparticles,
    Eigen::MatrixXi & N){
    
    int k = 50;
    Eigen::MatrixXd X_3d;
    
    X_3d.resize(numofparticles, 3);
    X_3d.setZero();
    for (int i = 0; i < numofparticles; i++){
        for (int j = 0; j < 2; j++){
            X_3d(i, j) = X(i,j);
        }
    }


    //build octtree
    std::vector<std::vector<int>> O_PI;
    Eigen::MatrixXi O_CH;
    Eigen::MatrixXd O_CN;
    Eigen::VectorXd O_W;
    igl::octree(X_3d,O_PI,O_CH,O_CN,O_W);

    N.resize(numofparticles, k);

    igl::knn(X_3d,k,O_PI,O_CH,O_CN,O_W,N);

    return;
}

template<int DIM>
void find_neighbor(const Eigen::MatrixXd & X, 
    const Eigen::Matrix<double, DIM, 1> lower_bound,
    const Eigen::Matrix<double, DIM, 1> upper_bound, //can comment out later
    const double cell_size,
    const int numofparticles,
    Eigen::MatrixXi & N){
    
    int k = 100;

    //build octtree
    std::vector<std::vector<int > > O_PI;
    Eigen::MatrixXi O_CH;
    Eigen::MatrixXd O_CN;
    Eigen::VectorXd O_W;
    igl::octree(X,O_PI,O_CH,O_CN,O_W);

    N.resize(X.rows(), k);
    igl::knn(X,k,O_PI,O_CH,O_CN,O_W,N);

    return;
}

template void find_neighbor<3>(const Eigen::MatrixXd & X, 
    const Eigen::Matrix<double, 3, 1> lower_bound,
    const Eigen::Matrix<double, 3, 1> upper_bound, 
    const double cell_size,
    const int numofparticles,
    Eigen::MatrixXi & N);
    



/*
    for (int i = 0; i < X.rows(); i++){
        for (int j = 0; j < 20; j++){
            N(i, I(i, j)) = 1;
        }
    }
*/










/*
    std::map<std::tuple<int, int, int>, std::set<int>> cells;
    N.setZero();
    std::tuple<int, int, int> grid_coord;
    int grid_x, grid_y, grid_z; //x, y, z of grid coord


    // form grid
    // add each particle to its corresponding grid cell, if the cell doesn't exist add cell

    for (int i = 0; i < x.rows(); i++){
        //
        grid_x = round((x(i,0) - lower_bound(0))/cell_size);
        grid_y = round((x(i,1) - lower_bound(1))/cell_size);
        grid_z = round((x(i,2) - lower_bound(2))/cell_size);

        grid_coord = std::make_tuple(grid_x, grid_y, grid_z);
        if (cells.find(grid_coord) == cells.end()){
            cells[grid_coord] = std::set<int>();
        }

        cells[grid_coord].insert(i);

        //std::cout << std::get<0>(grid_coord) << std::get<1>(grid_coord) << std::get<2>(grid_coord) << std::endl;
        
        for (auto it=cells[grid_coord].begin(); it != cells[grid_coord].end(); ++it) {
          //  std::cout <<  *it << std::endl;
        }

    }


    //find neighbors
    std::tuple<int, int, int> neighbor_coord;

    for (grid_x = 0; grid_x < ceil((upper_bound(0) - lower_bound(0))/cell_size); grid_x ++){
        for (grid_y = 0; grid_y < ceil((upper_bound(1) - lower_bound(1))/cell_size); grid_y ++){
            for (grid_z = 0; grid_z < ceil((upper_bound(2) - lower_bound(2))/cell_size); grid_z++){
               
                grid_coord = std::make_tuple(grid_x, grid_y, grid_z);
            
                //check there are particles in this cell
                if (cells.find(grid_coord) != cells.end()){ 
                    
                    std::set<int> neighboring_particles;

                    for (int x_iter = -1; x_iter < 2; x_iter ++){
                        for (int y_iter = -1; y_iter < 2; y_iter ++){
                            for (int z_iter = -1; z_iter < 2; z_iter ++){
                                neighbor_coord  = std::make_tuple(grid_x + x_iter, grid_y + y_iter, grid_z + z_iter);
                                if (cells.find(neighbor_coord) != cells.end()){
                                    // add elements to neghbor_particles
                                    neighboring_particles.insert(cells[neighbor_coord].begin(), cells[neighbor_coord].end());
                                }
                            } 
                        }
                    }
                    
                    //put neighbors into a single row vector
                    Eigen::RowVectorXd row;
                    row.resize(numofparticles);
                    row.setZero();
                    std::set<int>::iterator it1 = neighboring_particles.begin();

                    while (it1 != neighboring_particles.end()){
                        row(*it1) = 1;
                        it1 ++; 
                    }

                    std::set<int>::iterator it2 = cells[grid_coord].begin();
                    //loop through all particles in this box to change N
                    while (it2 != cells[grid_coord].end()){
                        N.row(*it2) = row;
                        it2 ++;
                    }
                }
            }
        }
    }*/



    