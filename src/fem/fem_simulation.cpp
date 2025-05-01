#include "fem_simulation.hpp"

#include <omp.h>

#include <cmath>
#include <fstream>
#include <iostream>

FemSimulation::FemSimulation(double domain_size, const std::array<int, 3>& grid_size,
                             double time_step, double permittivity, double permeability,
                             int time_frequency)
    : grid_size_(grid_size),
      domain_size_(domain_size),
      time_step_(time_step),
      permittivity_(permittivity),
      permeability_(permeability),
      time_frequency_(time_frequency),
      current_time_(0.0),
      time_0_(0),
      time_1_(1),
      time_2_(2) {
    initializeMesh();
    calculateElementStiffnessMatrix();
}

void FemSimulation::initializeMesh() {
    electric_field_x_.resize(
        3, std::vector<double>(grid_size_[0] * (grid_size_[1] + 1) * (grid_size_[2] + 1), 0.0));
    electric_field_y_.resize(
        3, std::vector<double>((grid_size_[0] + 1) * grid_size_[1] * (grid_size_[2] + 1), 0.0));
    electric_field_z_.resize(
        3, std::vector<double>((grid_size_[0] + 1) * (grid_size_[1] + 1) * grid_size_[2], 0.0));
    connectivity_x_.resize(grid_size_[0] * grid_size_[1] * grid_size_[2], std::vector<int>(4, 0));
    connectivity_y_.resize(grid_size_[0] * grid_size_[1] * grid_size_[2], std::vector<int>(4, 0));
    connectivity_z_.resize(grid_size_[0] * grid_size_[1] * grid_size_[2], std::vector<int>(4, 0));
    for (int i = 0; i < grid_size_[0]; ++i) {
        for (int j = 0; j < grid_size_[1]; ++j) {
            for (int k = 0; k < grid_size_[2]; ++k) {
                int idx = i + j * grid_size_[0] + k * grid_size_[0] * grid_size_[1];
                connectivity_x_[idx] = {idx + k * grid_size_[0], idx + (k + 1) * grid_size_[0],
                                        idx + (k + grid_size_[1] + 1) * grid_size_[0],
                                        idx + (k + grid_size_[1] + 2) * grid_size_[0]};
                connectivity_y_[idx] = {idx + j + k * grid_size_[1],
                                        idx + j + (k + grid_size_[0] + 1) * grid_size_[1],
                                        idx + j + k * grid_size_[1] + 1,
                                        idx + j + (k + grid_size_[0] + 1) * grid_size_[1] + 1};
                connectivity_z_[idx] = {
                    idx + j + k * (grid_size_[0] + grid_size_[1] + 1),
                    idx + j + k * (grid_size_[0] + grid_size_[1] + 1) + 1,
                    idx + j + k * (grid_size_[0] + grid_size_[1] + 1) + grid_size_[0] + 1,
                    idx + j + k * (grid_size_[0] + grid_size_[1] + 1) + grid_size_[0] + 2};
            }
        }
    }
}

void FemSimulation::calculateElementStiffnessMatrix() {
    element_stiffness_matrix_ = std::array<std::array<double, 12>, 12>{
        {{2, 0, 0, -2, -1, -1, 1, 1, -1, 1, -1, 1},
         {0, 2, -2, 0, 1, 1, -1, -1, -1, 1, -1, 1},
         {0, -2, 2, 0, -1, -1, 1, 1, 1, -1, 1, -1},
         {-2, 0, 0, 2, 1, 1, -1, -1, 1, -1, 1, -1},
         {-1, 1, -1, 1, 2, 0, 0, -2, -1, -1, 1, 1},
         {-1, 1, -1, 1, 0, 2, -2, 0, 1, 1, -1, -1},
         {1, -1, 1, -1, 0, -2, 2, 0, -1, -1, 1, 1},
         {1, -1, 1, -1, -2, 0, 0, 2, 1, 1, -1, -1},
         {-1, -1, 1, 1, -1, 1, -1, 1, 2, 0, 0, -2},
         {1, 1, -1, -1, -1, 1, -1, 1, 0, 2, -2, 0},
         {-1, -1, 1, 1, 1, -1, 1, -1, 0, -2, 2, 0},
         {1, 1, -1, -1, 1, -1, 1, -1, -2, 0, 0, 2}}};  // 2をかけた値
    for (int i = 0; i < 12; i++) {
        for (int j = 0; j < 12; j++) {
            element_stiffness_matrix_[i][j] = element_stiffness_matrix_[i][j] / 2.0 * time_step_ *
                                              time_step_ * domain_size_ / 2.0 / permeability_;
        }
    }
}

void FemSimulation::setSource_x(const std::array<int, 3>& position,
                                std::function<double(double)> source_function) {
    source_position_x_.push_back(position[0] + position[1] * grid_size_[0] +
                                 position[2] * grid_size_[0] * (grid_size_[1] + 1));
    source_function_x_.push_back(source_function);
}

void FemSimulation::setSource_y(const std::array<int, 3>& position,
                                std::function<double(double)> source_function) {
    source_position_y_.push_back(position[0] + position[1] * (grid_size_[0] + 1) +
                                 position[2] * (grid_size_[0] + 1) * grid_size_[1]);
    source_function_y_.push_back(source_function);
}

void FemSimulation::setSource_z(const std::array<int, 3>& position,
                                std::function<double(double)> source_function) {
    source_position_z_.push_back(position[0] + position[1] * (grid_size_[0] + 1) +
                                 position[2] * (grid_size_[0] + 1) * (grid_size_[1] + 1));
    source_function_z_.push_back(source_function);
}

void FemSimulation::setObservationPoint(
    const std::array<std::vector<std::array<int, 3>>, 3>& position) {
    observation_points_.push_back(position);
    saved_electric_field_.push_back(std::vector<std::array<double, 3>>());
}

void FemSimulation::updateTimeStep() {
    electric_field_x_[time_2_] =
        std::vector<double>(grid_size_[0] * (grid_size_[1] + 1) * (grid_size_[2] + 1), 0.0);
    electric_field_y_[time_2_] =
        std::vector<double>((grid_size_[0] + 1) * grid_size_[1] * (grid_size_[2] + 1), 0.0);
    electric_field_z_[time_2_] =
        std::vector<double>((grid_size_[0] + 1) * (grid_size_[1] + 1) * grid_size_[2], 0.0);

#pragma omp parallel
    {
        int idx;
        std::array<double, 12> temps;
        std::array<double, 12> electric_field_0;
        double temp;
        int l, m;

#pragma omp for collapse(3)
        for (int i = 0; i < grid_size_[0]; ++i) {
            for (int j = 0; j < grid_size_[1]; ++j) {
                for (int k = 0; k < grid_size_[2]; ++k) {
                    idx = i + j * grid_size_[0] + k * grid_size_[0] * grid_size_[1];
                    temps = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
                    electric_field_0 = {
                        electric_field_x_[time_1_][connectivity_x_[idx][0]],
                        electric_field_x_[time_1_][connectivity_x_[idx][1]],
                        electric_field_x_[time_1_][connectivity_x_[idx][2]],
                        electric_field_x_[time_1_][connectivity_x_[idx][3]],
                        electric_field_y_[time_1_][connectivity_y_[idx][0]],
                        electric_field_y_[time_1_][connectivity_y_[idx][1]],
                        electric_field_y_[time_1_][connectivity_y_[idx][2]],
                        electric_field_y_[time_1_][connectivity_y_[idx][3]],
                        electric_field_z_[time_1_][connectivity_z_[idx][0]],
                        electric_field_z_[time_1_][connectivity_z_[idx][1]],
                        electric_field_z_[time_1_][connectivity_z_[idx][2]],
                        electric_field_z_[time_1_][connectivity_z_[idx][3]],
                    };
                    for (l = 0; l < 12; ++l) {
                        temp = 0;
                        for (m = 0; m < 12; ++m) {
                            temp += element_stiffness_matrix_[l][m] * electric_field_0[m];
                        }
                        temps[l] = temp;
                    }

#pragma omp atomic
                    electric_field_x_[time_2_][connectivity_x_[idx][0]] -=
                        temps[0] / permittivity_ / domain_size_ / domain_size_ / domain_size_;
#pragma omp atomic
                    electric_field_x_[time_2_][connectivity_x_[idx][1]] -=
                        temps[1] / permittivity_ / domain_size_ / domain_size_ / domain_size_;
#pragma omp atomic
                    electric_field_x_[time_2_][connectivity_x_[idx][2]] -=
                        temps[2] / permittivity_ / domain_size_ / domain_size_ / domain_size_;
#pragma omp atomic
                    electric_field_x_[time_2_][connectivity_x_[idx][3]] -=
                        temps[3] / permittivity_ / domain_size_ / domain_size_ / domain_size_;
#pragma omp atomic
                    electric_field_y_[time_2_][connectivity_y_[idx][0]] -=
                        temps[4] / permittivity_ / domain_size_ / domain_size_ / domain_size_;
#pragma omp atomic
                    electric_field_y_[time_2_][connectivity_y_[idx][1]] -=
                        temps[5] / permittivity_ / domain_size_ / domain_size_ / domain_size_;
#pragma omp atomic
                    electric_field_y_[time_2_][connectivity_y_[idx][2]] -=
                        temps[6] / permittivity_ / domain_size_ / domain_size_ / domain_size_;
#pragma omp atomic
                    electric_field_y_[time_2_][connectivity_y_[idx][3]] -=
                        temps[7] / permittivity_ / domain_size_ / domain_size_ / domain_size_;
#pragma omp atomic
                    electric_field_z_[time_2_][connectivity_z_[idx][0]] -=
                        temps[8] / permittivity_ / domain_size_ / domain_size_ / domain_size_;
#pragma omp atomic
                    electric_field_z_[time_2_][connectivity_z_[idx][1]] -=
                        temps[9] / permittivity_ / domain_size_ / domain_size_ / domain_size_;
#pragma omp atomic
                    electric_field_z_[time_2_][connectivity_z_[idx][2]] -=
                        temps[10] / permittivity_ / domain_size_ / domain_size_ / domain_size_;
#pragma omp atomic
                    electric_field_z_[time_2_][connectivity_z_[idx][3]] -=
                        temps[11] / permittivity_ / domain_size_ / domain_size_ / domain_size_;
                }
            }
        }
    }

#pragma omp parallel for
    for (size_t i = 0; i < electric_field_x_[time_2_].size(); ++i) {
        electric_field_x_[time_2_][i] +=
            2.0 * electric_field_x_[time_1_][i] - electric_field_x_[time_0_][i];
    }
#pragma omp parallel for
    for (size_t i = 0; i < electric_field_y_[time_2_].size(); ++i) {
        electric_field_y_[time_2_][i] +=
            2.0 * electric_field_y_[time_1_][i] - electric_field_y_[time_0_][i];
    }
#pragma omp parallel for
    for (size_t i = 0; i < electric_field_z_[time_2_].size(); ++i) {
        electric_field_z_[time_2_][i] +=
            2.0 * electric_field_z_[time_1_][i] - electric_field_z_[time_0_][i];
    }

    for (size_t i = 0; i < source_position_x_.size(); ++i) {
        electric_field_x_[time_2_][source_position_x_[i]] -=
            (source_function_x_[i](current_time_ + time_step_ / 2) -
             source_function_x_[i](current_time_ - time_step_ / 2)) *
            time_step_ / permittivity_ / domain_size_ / domain_size_ / domain_size_;
    }
    for (size_t i = 0; i < source_position_y_.size(); ++i) {
        electric_field_y_[time_2_][source_position_y_[i]] -=
            (source_function_y_[i](current_time_ + time_step_ / 2) -
             source_function_y_[i](current_time_ - time_step_ / 2)) *
            time_step_ / permittivity_ / domain_size_ / domain_size_ / domain_size_;
    }
    for (size_t i = 0; i < source_position_z_.size(); ++i) {
        electric_field_z_[time_2_][source_position_z_[i]] -=
            (source_function_z_[i](current_time_ + time_step_ / 2) -
             source_function_z_[i](current_time_ - time_step_ / 2)) *
            time_step_ / permittivity_ / domain_size_ / domain_size_ / domain_size_;
    }
}

void FemSimulation::applyBoundaryConditions() {
// 完全導体境界条件（電場の接線成分を0に）
#pragma omp parallel
    {
        int idx_0, idx_1;

#pragma omp for collapse(2)
        for (int i = 0; i < grid_size_[0]; ++i) {
            for (int j = 0; j < grid_size_[1] + 1; ++j) {
                idx_0 = i + j * grid_size_[0];
                idx_1 = i + j * grid_size_[0] + grid_size_[0] * (grid_size_[1] + 1) * grid_size_[2];
                electric_field_x_[time_2_][idx_0] = 0.0;
                electric_field_x_[time_2_][idx_1] = 0.0;
            }
        }

#pragma omp for collapse(2)
        for (int i = 0; i < grid_size_[0] + 1; ++i) {
            for (int k = 0; k < grid_size_[2]; ++k) {
                idx_0 = i + k * grid_size_[0] * (grid_size_[1] + 1);
                idx_1 = i + k * grid_size_[0] * (grid_size_[1] + 1) + grid_size_[0] * grid_size_[1];
                electric_field_x_[time_2_][idx_0] = 0.0;
                electric_field_x_[time_2_][idx_1] = 0.0;
            }
        }

#pragma omp for collapse(2)
        for (int j = 0; j < grid_size_[1] + 1; ++j) {
            for (int k = 0; k < grid_size_[2]; ++k) {
                idx_0 = j * grid_size_[0] + k * grid_size_[0] * (grid_size_[1] + 1);
                idx_1 =
                    j * grid_size_[0] + k * grid_size_[0] * (grid_size_[1] + 1) + grid_size_[0] - 1;
                electric_field_y_[time_2_][idx_0] = 0.0;
                electric_field_y_[time_2_][idx_1] = 0.0;
            }
        }

#pragma omp for collapse(2)
        for (int j = 0; j < grid_size_[1] + 1; ++j) {
            for (int i = 0; i < grid_size_[0]; ++i) {
                idx_0 = i + j * grid_size_[0];
                idx_1 = i + j * grid_size_[0] + grid_size_[0] * (grid_size_[1] + 1) * grid_size_[2];
                electric_field_y_[time_2_][idx_0] = 0.0;
                electric_field_y_[time_2_][idx_1] = 0.0;
            }
        }

#pragma omp for collapse(2)
        for (int k = 0; k < grid_size_[2] + 1; ++k) {
            for (int i = 0; i < grid_size_[0]; ++i) {
                idx_0 = i + k * grid_size_[0] * (grid_size_[1] + 1);
                idx_1 = i + k * grid_size_[0] * (grid_size_[1] + 1) + grid_size_[0] * grid_size_[1];
                electric_field_z_[time_2_][idx_0] = 0.0;
                electric_field_z_[time_2_][idx_1] = 0.0;
            }
        }

#pragma omp for collapse(2)
        for (int k = 0; k < grid_size_[2] + 1; ++k) {
            for (int j = 0; j < grid_size_[1]; ++j) {
                idx_0 = j * grid_size_[0] + k * grid_size_[0] * (grid_size_[1] + 1);
                idx_1 =
                    j * grid_size_[0] + k * grid_size_[0] * (grid_size_[1] + 1) + grid_size_[0] - 1;
                electric_field_z_[time_2_][idx_0] = 0.0;
                electric_field_z_[time_2_][idx_1] = 0.0;
            }
        }
    }
}

void FemSimulation::run(int num_steps) {
    saved_electric_field_.resize(observation_points_.size());
    for (auto& point : saved_electric_field_) {
        point.resize(static_cast<int>(std::round(num_steps / time_frequency_)) + 1);
    }
    for (int step = 1; step < num_steps + 1; ++step) {
        std::cerr << "step: " << step << std::endl;
        updateTimeStep();
        applyBoundaryConditions();

        // 観測点での値を保存
        if (step % time_frequency_ == 0) {
            for (size_t i = 0; i < observation_points_.size(); ++i) {
                for (size_t j = 0; j < observation_points_[i][0].size(); ++j) {
                    int idx_x =
                        observation_points_[i][0][j][0] +
                        observation_points_[i][0][j][1] * grid_size_[0] +
                        observation_points_[i][0][j][2] * grid_size_[0] * (grid_size_[1] + 1);
                    saved_electric_field_[i][static_cast<int>(std::round(step / time_frequency_))]
                                         [0] += electric_field_x_[time_2_][idx_x];
                }
                saved_electric_field_[i][static_cast<int>(std::round(step / time_frequency_))][0] /=
                    observation_points_[i][0].size();
                for (size_t j = 0; j < observation_points_[i][1].size(); ++j) {
                    int idx_y =
                        observation_points_[i][1][j][0] +
                        observation_points_[i][1][j][1] * (grid_size_[0] + 1) +
                        observation_points_[i][1][j][2] * (grid_size_[0] + 1) * grid_size_[1];
                    saved_electric_field_[i][static_cast<int>(std::round(step / time_frequency_))]
                                         [1] += electric_field_y_[time_2_][idx_y];
                }
                saved_electric_field_[i][static_cast<int>(std::round(step / time_frequency_))][1] /=
                    observation_points_[i][1].size();
                for (size_t j = 0; j < observation_points_[i][2].size(); ++j) {
                    int idx_z =
                        observation_points_[i][2][j][0] +
                        observation_points_[i][2][j][1] * (grid_size_[0] + 1) +
                        observation_points_[i][2][j][2] * (grid_size_[0] + 1) * (grid_size_[1] + 1);
                    saved_electric_field_[i][static_cast<int>(std::round(step / time_frequency_))]
                                         [2] += electric_field_z_[time_2_][idx_z];
                }
                saved_electric_field_[i][static_cast<int>(std::round(step / time_frequency_))][2] /=
                    observation_points_[i][2].size();
            }
            current_time_ += time_step_;
            time_0_ = (time_0_ + 1) % 3;
            time_1_ = (time_1_ + 1) % 3;
            time_2_ = (time_2_ + 1) % 3;
        }
    }
}

void FemSimulation::saveResults(double observe_r, std::function<double(double)> source_function,
                                int num_steps, double c) {
    for (size_t i = 0; i < saved_electric_field_.size(); ++i) {
        std::string filename_i = "results_" + std::to_string(i) + ".csv";
        std::ofstream file(filename_i);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << filename_i << std::endl;
            return;
        }

        // 結果をCSV形式で保存
        file << "time,Ex,Ey,Ez" << std::endl;
        for (size_t j = 0; j < saved_electric_field_[i].size(); ++j) {
            file << j * time_frequency_ * time_step_ << "," << saved_electric_field_[i][j][0] << ","
                 << saved_electric_field_[i][j][1] << "," << saved_electric_field_[i][j][2]
                 << std::endl;
        }
        file.close();
    }
    std::string filename_r = "result_solution.csv";
    std::ofstream file_r(filename_r);
    if (!file_r.is_open()) {
        std::cerr << "Error: Could not open file " << filename_r << std::endl;
        return;
    }
    file_r << "time,Ex,Ey,Ez" << std::endl;
    for (int i = 0; i < num_steps + 1; ++i) {
        file_r << i * time_frequency_ * time_step_ << ","
               << source_function(i * time_frequency_ * time_step_ - observe_r / c) / 4 / M_PI /
                      observe_r
               << "," << 0 << "," << 0 << std::endl;
    }
    file_r.close();
}

bool check_params(std::array<double, 3> domain_sizes, std::array<int, 3> grid_num, double duration,
                  double time_step, double domain_size, double c) {
    if (domain_sizes[0] <= 0 || domain_sizes[1] <= 0 || domain_sizes[2] <= 0) {
        return false;
    }
    if (grid_num[0] <= 0 || grid_num[1] <= 0 || grid_num[2] <= 0) {
        return false;
    }
    if (duration <= 0) {
        return false;
    }
    if (time_step <= 0) {
        return false;
    }
    if (std::round(domain_sizes[0] / grid_num[0]) != std::round(domain_sizes[1] / grid_num[1]) ||
        std::round(domain_sizes[0] / grid_num[0]) != std::round(domain_sizes[2] / grid_num[2])) {
        std::cerr << "Invalid grid size" << std::endl;
        return false;
    }
    if (c * time_step >= domain_size) {
        std::cerr << "Invalid time step" << std::endl;
        std::cerr << "c * time_step: " << c * time_step << std::endl;
        std::cerr << "domain_size: " << domain_size << std::endl;
        return false;
    }
    return true;
};