#include "find_neighbor.h"
#include <map>
#include <set>
#include <tuple>

void find_neighbor(const Eigen::MatrixXd x, 
    const Eigen::Vector3d lower_bound,
    const Eigen::Vector3d upper_bound, 
    const double cell_size,
    const int numofparticles,
    Eigen::MatrixXd N){
    
    std::map<std::tuple<int, int, int>, std::set<int>> cells;
    N.setZero();
    std::tuple<int, int, int> grid_coord;
    int grid_x, grid_y, grid_z; //x, y, z of grid coord


    //form grid
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
    }
    return;
}