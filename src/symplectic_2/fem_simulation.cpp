#include <mpi.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <Eigen/Dense>
#include <sstream>
#include <string>
#include <vector>
#include <stdexcept>
#include <openacc.h>

#include "fem_simulation.hpp"
#include "fem_kernels.hpp"

// vectorのポインタ
double *ef_x_ptr_;
double *ef_y_ptr_;
double *ef_z_ptr_;
int *conn_x_ptr_;
int *conn_y_ptr_;
int *conn_z_ptr_;
int *src_pos_x_ptr_;
int *src_pos_y_ptr_;
int *src_pos_z_ptr_;
double *send_buf_x_plane_0_y_ptr_;
double *recv_buf_x_plane_0_y_ptr_;
double *send_buf_x_plane_1_y_ptr_;
double *recv_buf_x_plane_1_y_ptr_;
double *send_buf_x_plane_0_z_ptr_;
double *recv_buf_x_plane_0_z_ptr_;
double *send_buf_x_plane_1_z_ptr_;
double *recv_buf_x_plane_1_z_ptr_;
double *send_buf_y_plane_0_x_ptr_;
double *recv_buf_y_plane_0_x_ptr_;
double *send_buf_y_plane_1_x_ptr_;
double *recv_buf_y_plane_1_x_ptr_;
double *send_buf_y_plane_0_z_ptr_;
double *recv_buf_y_plane_0_z_ptr_;
double *send_buf_y_plane_1_z_ptr_;
double *recv_buf_y_plane_1_z_ptr_;
double *send_buf_z_plane_0_x_ptr_;
double *recv_buf_z_plane_0_x_ptr_;
double *send_buf_z_plane_1_x_ptr_;
double *recv_buf_z_plane_1_x_ptr_;
double *send_buf_z_plane_0_y_ptr_;
double *recv_buf_z_plane_0_y_ptr_;
double *send_buf_z_plane_1_y_ptr_;
double *recv_buf_z_plane_1_y_ptr_;
double *send_buf_x_line_0_ptr_;
double *recv_buf_x_line_0_ptr_;
double *send_buf_x_line_1_ptr_;
double *recv_buf_x_line_1_ptr_;
double *send_buf_x_line_2_ptr_;
double *recv_buf_x_line_2_ptr_;
double *send_buf_x_line_3_ptr_;
double *recv_buf_x_line_3_ptr_;
double *send_buf_y_line_0_ptr_;
double *recv_buf_y_line_0_ptr_;
double *send_buf_y_line_1_ptr_;
double *recv_buf_y_line_1_ptr_;
double *send_buf_y_line_2_ptr_;
double *recv_buf_y_line_2_ptr_;
double *send_buf_y_line_3_ptr_;
double *recv_buf_y_line_3_ptr_;
double *send_buf_z_line_0_ptr_;
double *recv_buf_z_line_0_ptr_;
double *send_buf_z_line_1_ptr_;
double *recv_buf_z_line_1_ptr_;
double *send_buf_z_line_2_ptr_;
double *recv_buf_z_line_2_ptr_;
double *send_buf_z_line_3_ptr_;
double *recv_buf_z_line_3_ptr_;
int *outer_elems_ptr_;
double *element_stiffness_matrix_ptr_;

#pragma acc routine seq
double source_function(double t, double permeability, double domain_size)
{
    // ガウス関数の2階微分
    const double freq = 1.0e9;     // 中心周波数 [Hz]
    const double chi = 1.0 / freq; // 中心時間 [s]
    double delay = t - chi;
    double zeta = 2.0 * M_PI * M_PI * freq * freq;
    return 2.0 * zeta * (2.0 * zeta * delay * delay - 1.0) *
           std::exp(0.0 - zeta * delay * delay) * 0.001; // ヘルツダイポールのdl = 0.001
};

FemSimulation::FemSimulation(int order, const std::array<int, 3> &grid_size, double domain_size,
                             double time_step, double permittivity, double permeability,
                             int time_frequency, int use_ofem, const std::array<int, 3> &dims, const std::array<int, 3> &coords, int rank,
                             MPI_Comm comm_x_plane_0, MPI_Comm comm_x_plane_1, MPI_Comm comm_y_plane_0, MPI_Comm comm_y_plane_1, MPI_Comm comm_z_plane_0, MPI_Comm comm_z_plane_1,
                             MPI_Comm comm_x_line_0, MPI_Comm comm_x_line_1, MPI_Comm comm_x_line_2, MPI_Comm comm_x_line_3,
                             MPI_Comm comm_y_line_0, MPI_Comm comm_y_line_1, MPI_Comm comm_y_line_2, MPI_Comm comm_y_line_3,
                             MPI_Comm comm_z_line_0, MPI_Comm comm_z_line_1, MPI_Comm comm_z_line_2, MPI_Comm comm_z_line_3,
                             int rank_x_plane_0, int rank_x_plane_1, int rank_y_plane_0, int rank_y_plane_1, int rank_z_plane_0, int rank_z_plane_1,
                             int rank_x_line_0, int rank_x_line_1, int rank_x_line_2, int rank_x_line_3,
                             int rank_y_line_0, int rank_y_line_1, int rank_y_line_2, int rank_y_line_3,
                             int rank_z_line_0, int rank_z_line_1, int rank_z_line_2, int rank_z_line_3)
    : order_(order),
      grid_size_x_(grid_size[0] / dims[0]),
      grid_size_y_(grid_size[1] / dims[1]),
      grid_size_z_(grid_size[2] / dims[2]),
      domain_size_(domain_size),
      time_step_(time_step),
      permittivity_(permittivity),
      permeability_(permeability),
      time_frequency_(time_frequency),
      current_time_(0.0),
      use_ofem_(use_ofem),
      dim_x_(dims[0]),
      dim_y_(dims[1]),
      dim_z_(dims[2]),
      coord_x_(coords[0]),
      coord_y_(coords[1]),
      coord_z_(coords[2]),
      rank_(rank),
      comm_x_plane_0_(comm_x_plane_0),
      comm_x_plane_1_(comm_x_plane_1),
      comm_y_plane_0_(comm_y_plane_0),
      comm_y_plane_1_(comm_y_plane_1),
      comm_z_plane_0_(comm_z_plane_0),
      comm_z_plane_1_(comm_z_plane_1),
      comm_x_line_0_(comm_x_line_0),
      comm_x_line_1_(comm_x_line_1),
      comm_x_line_2_(comm_x_line_2),
      comm_x_line_3_(comm_x_line_3),
      comm_y_line_0_(comm_y_line_0),
      comm_y_line_1_(comm_y_line_1),
      comm_y_line_2_(comm_y_line_2),
      comm_y_line_3_(comm_y_line_3),
      comm_z_line_0_(comm_z_line_0),
      comm_z_line_1_(comm_z_line_1),
      comm_z_line_2_(comm_z_line_2),
      comm_z_line_3_(comm_z_line_3),
      rank_x_plane_0_(rank_x_plane_0),
      rank_x_plane_1_(rank_x_plane_1),
      rank_y_plane_0_(rank_y_plane_0),
      rank_y_plane_1_(rank_y_plane_1),
      rank_z_plane_0_(rank_z_plane_0),
      rank_z_plane_1_(rank_z_plane_1),
      rank_x_line_0_(rank_x_line_0),
      rank_x_line_1_(rank_x_line_1),
      rank_x_line_2_(rank_x_line_2),
      rank_x_line_3_(rank_x_line_3),
      rank_y_line_0_(rank_y_line_0),
      rank_y_line_1_(rank_y_line_1),
      rank_y_line_2_(rank_y_line_2),
      rank_y_line_3_(rank_y_line_3),
      rank_z_line_0_(rank_z_line_0),
      rank_z_line_1_(rank_z_line_1),
      rank_z_line_2_(rank_z_line_2),
      rank_z_line_3_(rank_z_line_3)
{
    initializeMesh();
    calculateElementStiffnessMatrix();
}

void FemSimulation::initializeMesh()
{
    int idx;
    int idx_element;
    int count = 0;
    int l, m, n;

    electric_field_x_.resize(3 * static_cast<size_t>(grid_size_x_) * static_cast<size_t>(grid_size_y_ + 1) * static_cast<size_t>(grid_size_z_ + 1), 0.0);
    electric_field_y_.resize(3 * static_cast<size_t>(grid_size_x_ + 1) * static_cast<size_t>(grid_size_y_) * static_cast<size_t>(grid_size_z_ + 1), 0.0);
    electric_field_z_.resize(3 * static_cast<size_t>(grid_size_x_ + 1) * static_cast<size_t>(grid_size_y_ + 1) * static_cast<size_t>(grid_size_z_), 0.0);
    connectivity_x_.resize(order_ * (order_ + 1) * (order_ + 1) * static_cast<size_t>(grid_size_x_) * static_cast<size_t>(grid_size_y_) * static_cast<size_t>(grid_size_z_), 0);
    connectivity_y_.resize(order_ * (order_ + 1) * (order_ + 1) * static_cast<size_t>(grid_size_x_) * static_cast<size_t>(grid_size_y_) * static_cast<size_t>(grid_size_z_), 0);
    connectivity_z_.resize(order_ * (order_ + 1) * (order_ + 1) * static_cast<size_t>(grid_size_x_) * static_cast<size_t>(grid_size_y_) * static_cast<size_t>(grid_size_z_), 0);
    send_buf_x_plane_0_y_.resize(static_cast<size_t>(grid_size_y_) * static_cast<size_t>(grid_size_z_ + 1), 0.0);
    recv_buf_x_plane_0_y_.resize(static_cast<size_t>(grid_size_y_) * static_cast<size_t>(grid_size_z_ + 1), 0.0);
    send_buf_x_plane_1_y_.resize(static_cast<size_t>(grid_size_y_) * static_cast<size_t>(grid_size_z_ + 1), 0.0);
    recv_buf_x_plane_1_y_.resize(static_cast<size_t>(grid_size_y_) * static_cast<size_t>(grid_size_z_ + 1), 0.0);
    send_buf_x_plane_0_z_.resize(static_cast<size_t>(grid_size_y_ + 1) * static_cast<size_t>(grid_size_z_), 0.0);
    recv_buf_x_plane_0_z_.resize(static_cast<size_t>(grid_size_y_ + 1) * static_cast<size_t>(grid_size_z_), 0.0);
    send_buf_x_plane_1_z_.resize(static_cast<size_t>(grid_size_y_ + 1) * static_cast<size_t>(grid_size_z_), 0.0);
    recv_buf_x_plane_1_z_.resize(static_cast<size_t>(grid_size_y_ + 1) * static_cast<size_t>(grid_size_z_), 0.0);
    send_buf_y_plane_0_x_.resize(static_cast<size_t>(grid_size_x_) * static_cast<size_t>(grid_size_z_ + 1), 0.0);
    recv_buf_y_plane_0_x_.resize(static_cast<size_t>(grid_size_x_) * static_cast<size_t>(grid_size_z_ + 1), 0.0);
    send_buf_y_plane_1_x_.resize(static_cast<size_t>(grid_size_x_) * static_cast<size_t>(grid_size_z_ + 1), 0.0);
    recv_buf_y_plane_1_x_.resize(static_cast<size_t>(grid_size_x_) * static_cast<size_t>(grid_size_z_ + 1), 0.0);
    send_buf_y_plane_0_z_.resize(static_cast<size_t>(grid_size_x_ + 1) * static_cast<size_t>(grid_size_z_), 0.0);
    recv_buf_y_plane_0_z_.resize(static_cast<size_t>(grid_size_x_ + 1) * static_cast<size_t>(grid_size_z_), 0.0);
    send_buf_y_plane_1_z_.resize(static_cast<size_t>(grid_size_x_ + 1) * static_cast<size_t>(grid_size_z_), 0.0);
    recv_buf_y_plane_1_z_.resize(static_cast<size_t>(grid_size_x_ + 1) * static_cast<size_t>(grid_size_z_), 0.0);
    send_buf_z_plane_0_x_.resize(static_cast<size_t>(grid_size_x_) * static_cast<size_t>(grid_size_y_ + 1), 0.0);
    recv_buf_z_plane_0_x_.resize(static_cast<size_t>(grid_size_x_) * static_cast<size_t>(grid_size_y_ + 1), 0.0);
    send_buf_z_plane_1_x_.resize(static_cast<size_t>(grid_size_x_) * static_cast<size_t>(grid_size_y_ + 1), 0.0);
    recv_buf_z_plane_1_x_.resize(static_cast<size_t>(grid_size_x_) * static_cast<size_t>(grid_size_y_ + 1), 0.0);
    send_buf_z_plane_0_y_.resize(static_cast<size_t>(grid_size_x_ + 1) * static_cast<size_t>(grid_size_y_), 0.0);
    recv_buf_z_plane_0_y_.resize(static_cast<size_t>(grid_size_x_ + 1) * static_cast<size_t>(grid_size_y_), 0.0);
    send_buf_z_plane_1_y_.resize(static_cast<size_t>(grid_size_x_ + 1) * static_cast<size_t>(grid_size_y_), 0.0);
    recv_buf_z_plane_1_y_.resize(static_cast<size_t>(grid_size_x_ + 1) * static_cast<size_t>(grid_size_y_), 0.0);
    send_buf_x_line_0_.resize(static_cast<size_t>(grid_size_x_), 0.0);
    recv_buf_x_line_0_.resize(static_cast<size_t>(grid_size_x_), 0.0);
    send_buf_x_line_1_.resize(static_cast<size_t>(grid_size_x_), 0.0);
    recv_buf_x_line_1_.resize(static_cast<size_t>(grid_size_x_), 0.0);
    send_buf_x_line_2_.resize(static_cast<size_t>(grid_size_x_), 0.0);
    recv_buf_x_line_2_.resize(static_cast<size_t>(grid_size_x_), 0.0);
    send_buf_x_line_3_.resize(static_cast<size_t>(grid_size_x_), 0.0);
    recv_buf_x_line_3_.resize(static_cast<size_t>(grid_size_x_), 0.0);
    send_buf_y_line_0_.resize(static_cast<size_t>(grid_size_y_), 0.0);
    recv_buf_y_line_0_.resize(static_cast<size_t>(grid_size_y_), 0.0);
    send_buf_y_line_1_.resize(static_cast<size_t>(grid_size_y_), 0.0);
    recv_buf_y_line_1_.resize(static_cast<size_t>(grid_size_y_), 0.0);
    send_buf_y_line_2_.resize(static_cast<size_t>(grid_size_y_), 0.0);
    recv_buf_y_line_2_.resize(static_cast<size_t>(grid_size_y_), 0.0);
    send_buf_y_line_3_.resize(static_cast<size_t>(grid_size_y_), 0.0);
    recv_buf_y_line_3_.resize(static_cast<size_t>(grid_size_y_), 0.0);
    send_buf_z_line_0_.resize(static_cast<size_t>(grid_size_z_), 0.0);
    recv_buf_z_line_0_.resize(static_cast<size_t>(grid_size_z_), 0.0);
    send_buf_z_line_1_.resize(static_cast<size_t>(grid_size_z_), 0.0);
    recv_buf_z_line_1_.resize(static_cast<size_t>(grid_size_z_), 0.0);
    send_buf_z_line_2_.resize(static_cast<size_t>(grid_size_z_), 0.0);
    recv_buf_z_line_2_.resize(static_cast<size_t>(grid_size_z_), 0.0);
    send_buf_z_line_3_.resize(static_cast<size_t>(grid_size_z_), 0.0);
    recv_buf_z_line_3_.resize(static_cast<size_t>(grid_size_z_), 0.0);
    outer_elems_.resize(2 * (static_cast<size_t>(grid_size_x_) / order_ * static_cast<size_t>(grid_size_y_) / order_ + static_cast<size_t>(grid_size_y_) / order_ * static_cast<size_t>(grid_size_z_) / order_ + static_cast<size_t>(grid_size_z_) / order_ * static_cast<size_t>(grid_size_x_) / order_) - 4 * (static_cast<size_t>(grid_size_x_) / order_ + static_cast<size_t>(grid_size_y_) / order_ + static_cast<size_t>(grid_size_z_) / order_) + 8, 0);

    ef_x_ptr_ = electric_field_x_.data();
    ef_y_ptr_ = electric_field_y_.data();
    ef_z_ptr_ = electric_field_z_.data();
    conn_x_ptr_ = connectivity_x_.data();
    conn_y_ptr_ = connectivity_y_.data();
    conn_z_ptr_ = connectivity_z_.data();
    send_buf_x_plane_0_y_ptr_ = send_buf_x_plane_0_y_.data();
    recv_buf_x_plane_0_y_ptr_ = recv_buf_x_plane_0_y_.data();
    send_buf_x_plane_1_y_ptr_ = send_buf_x_plane_1_y_.data();
    recv_buf_x_plane_1_y_ptr_ = recv_buf_x_plane_1_y_.data();
    send_buf_x_plane_0_z_ptr_ = send_buf_x_plane_0_z_.data();
    recv_buf_x_plane_0_z_ptr_ = recv_buf_x_plane_0_z_.data();
    send_buf_x_plane_1_z_ptr_ = send_buf_x_plane_1_z_.data();
    recv_buf_x_plane_1_z_ptr_ = recv_buf_x_plane_1_z_.data();
    send_buf_y_plane_0_x_ptr_ = send_buf_y_plane_0_x_.data();
    recv_buf_y_plane_0_x_ptr_ = recv_buf_y_plane_0_x_.data();
    send_buf_y_plane_1_x_ptr_ = send_buf_y_plane_1_x_.data();
    recv_buf_y_plane_1_x_ptr_ = recv_buf_y_plane_1_x_.data();
    send_buf_y_plane_0_z_ptr_ = send_buf_y_plane_0_z_.data();
    recv_buf_y_plane_0_z_ptr_ = recv_buf_y_plane_0_z_.data();
    send_buf_y_plane_1_z_ptr_ = send_buf_y_plane_1_z_.data();
    recv_buf_y_plane_1_z_ptr_ = recv_buf_y_plane_1_z_.data();
    send_buf_z_plane_0_x_ptr_ = send_buf_z_plane_0_x_.data();
    recv_buf_z_plane_0_x_ptr_ = recv_buf_z_plane_0_x_.data();
    send_buf_z_plane_1_x_ptr_ = send_buf_z_plane_1_x_.data();
    recv_buf_z_plane_1_x_ptr_ = recv_buf_z_plane_1_x_.data();
    send_buf_z_plane_0_y_ptr_ = send_buf_z_plane_0_y_.data();
    recv_buf_z_plane_0_y_ptr_ = recv_buf_z_plane_0_y_.data();
    send_buf_z_plane_1_y_ptr_ = send_buf_z_plane_1_y_.data();
    recv_buf_z_plane_1_y_ptr_ = recv_buf_z_plane_1_y_.data();
    send_buf_x_line_0_ptr_ = send_buf_x_line_0_.data();
    recv_buf_x_line_0_ptr_ = recv_buf_x_line_0_.data();
    send_buf_x_line_1_ptr_ = send_buf_x_line_1_.data();
    recv_buf_x_line_1_ptr_ = recv_buf_x_line_1_.data();
    send_buf_x_line_2_ptr_ = send_buf_x_line_2_.data();
    recv_buf_x_line_2_ptr_ = recv_buf_x_line_2_.data();
    send_buf_x_line_3_ptr_ = send_buf_x_line_3_.data();
    recv_buf_x_line_3_ptr_ = recv_buf_x_line_3_.data();
    send_buf_y_line_0_ptr_ = send_buf_y_line_0_.data();
    recv_buf_y_line_0_ptr_ = recv_buf_y_line_0_.data();
    send_buf_y_line_1_ptr_ = send_buf_y_line_1_.data();
    recv_buf_y_line_1_ptr_ = recv_buf_y_line_1_.data();
    send_buf_y_line_2_ptr_ = send_buf_y_line_2_.data();
    recv_buf_y_line_2_ptr_ = recv_buf_y_line_2_.data();
    send_buf_y_line_3_ptr_ = send_buf_y_line_3_.data();
    recv_buf_y_line_3_ptr_ = recv_buf_y_line_3_.data();
    send_buf_z_line_0_ptr_ = send_buf_z_line_0_.data();
    recv_buf_z_line_0_ptr_ = recv_buf_z_line_0_.data();
    send_buf_z_line_1_ptr_ = send_buf_z_line_1_.data();
    recv_buf_z_line_1_ptr_ = recv_buf_z_line_1_.data();
    send_buf_z_line_2_ptr_ = send_buf_z_line_2_.data();
    recv_buf_z_line_2_ptr_ = recv_buf_z_line_2_.data();
    send_buf_z_line_3_ptr_ = send_buf_z_line_3_.data();
    recv_buf_z_line_3_ptr_ = recv_buf_z_line_3_.data();
    outer_elems_ptr_ = outer_elems_.data();

    ENx = electric_field_x_.size();
    ENy = electric_field_y_.size();
    ENz = electric_field_z_.size();
    CNx = connectivity_x_.size();
    CNy = connectivity_y_.size();
    CNz = connectivity_z_.size();
    BNx0y = send_buf_x_plane_0_y_.size();
    BNx1y = send_buf_x_plane_1_y_.size();
    BNx0z = send_buf_x_plane_0_z_.size();
    BNx1z = send_buf_x_plane_1_z_.size();
    BNy0x = send_buf_y_plane_0_x_.size();
    BNy1x = send_buf_y_plane_1_x_.size();
    BNy0z = send_buf_y_plane_0_z_.size();
    BNy1z = send_buf_y_plane_1_z_.size();
    BNz0x = send_buf_z_plane_0_x_.size();
    BNz1x = send_buf_z_plane_1_x_.size();
    BNz0y = send_buf_z_plane_0_y_.size();
    BNz1y = send_buf_z_plane_1_y_.size();
    BLx0 = send_buf_x_line_0_.size();
    BLx1 = send_buf_x_line_1_.size();
    BLx2 = send_buf_x_line_2_.size();
    BLx3 = send_buf_x_line_3_.size();
    BLy0 = send_buf_y_line_0_.size();
    BLy1 = send_buf_y_line_1_.size();
    BLy2 = send_buf_y_line_2_.size();
    BLy3 = send_buf_y_line_3_.size();
    BLz0 = send_buf_z_line_0_.size();
    BLz1 = send_buf_z_line_1_.size();
    BLz2 = send_buf_z_line_2_.size();
    BLz3 = send_buf_z_line_3_.size();
    OE = outer_elems_.size();

#pragma acc data copyin(ef_x_ptr_[0 : ENx])
#pragma acc data copyin(ef_y_ptr_[0 : ENy])
#pragma acc data copyin(ef_z_ptr_[0 : ENz])
#pragma acc data copyin(conn_x_ptr_[0 : CNx])
#pragma acc data copyin(conn_y_ptr_[0 : CNy])
#pragma acc data copyin(conn_z_ptr_[0 : CNz])
#pragma acc data copyin(send_buf_x_plane_0_y_ptr_[0 : BNx0y])
#pragma acc data copyin(recv_buf_x_plane_0_y_ptr_[0 : BNx0y])
#pragma acc data copyin(send_buf_x_plane_1_y_ptr_[0 : BNx1y])
#pragma acc data copyin(recv_buf_x_plane_1_y_ptr_[0 : BNx1y])
#pragma acc data copyin(send_buf_x_plane_0_z_ptr_[0 : BNx0z])
#pragma acc data copyin(recv_buf_x_plane_0_z_ptr_[0 : BNx0z])
#pragma acc data copyin(send_buf_x_plane_1_z_ptr_[0 : BNx1z])
#pragma acc data copyin(recv_buf_x_plane_1_z_ptr_[0 : BNx1z])
#pragma acc data copyin(send_buf_y_plane_0_x_ptr_[0 : BNy0x])
#pragma acc data copyin(recv_buf_y_plane_0_x_ptr_[0 : BNy0x])
#pragma acc data copyin(send_buf_y_plane_1_x_ptr_[0 : BNy1x])
#pragma acc data copyin(recv_buf_y_plane_1_x_ptr_[0 : BNy1x])
#pragma acc data copyin(send_buf_y_plane_0_z_ptr_[0 : BNy0z])
#pragma acc data copyin(recv_buf_y_plane_0_z_ptr_[0 : BNy0z])
#pragma acc data copyin(send_buf_y_plane_1_z_ptr_[0 : BNy1z])
#pragma acc data copyin(recv_buf_y_plane_1_z_ptr_[0 : BNy1z])
#pragma acc data copyin(send_buf_z_plane_0_x_ptr_[0 : BNz0x])
#pragma acc data copyin(recv_buf_z_plane_0_x_ptr_[0 : BNz0x])
#pragma acc data copyin(send_buf_z_plane_1_x_ptr_[0 : BNz1x])
#pragma acc data copyin(recv_buf_z_plane_1_x_ptr_[0 : BNz1x])
#pragma acc data copyin(send_buf_z_plane_0_y_ptr_[0 : BNz0y])
#pragma acc data copyin(recv_buf_z_plane_0_y_ptr_[0 : BNz0y])
#pragma acc data copyin(send_buf_z_plane_1_y_ptr_[0 : BNz1y])
#pragma acc data copyin(recv_buf_z_plane_1_y_ptr_[0 : BNz1y])
#pragma acc data copyin(send_buf_x_line_0_ptr_[0 : BLx0])
#pragma acc data copyin(recv_buf_x_line_0_ptr_[0 : BLx0])
#pragma acc data copyin(send_buf_x_line_1_ptr_[0 : BLx1])
#pragma acc data copyin(recv_buf_x_line_1_ptr_[0 : BLx1])
#pragma acc data copyin(send_buf_x_line_2_ptr_[0 : BLx2])
#pragma acc data copyin(recv_buf_x_line_2_ptr_[0 : BLx2])
#pragma acc data copyin(send_buf_x_line_3_ptr_[0 : BLx3])
#pragma acc data copyin(recv_buf_x_line_3_ptr_[0 : BLx3])
#pragma acc data copyin(send_buf_y_line_0_ptr_[0 : BLy0])
#pragma acc data copyin(recv_buf_y_line_0_ptr_[0 : BLy0])
#pragma acc data copyin(send_buf_y_line_1_ptr_[0 : BLy1])
#pragma acc data copyin(recv_buf_y_line_1_ptr_[0 : BLy1])
#pragma acc data copyin(send_buf_y_line_2_ptr_[0 : BLy2])
#pragma acc data copyin(recv_buf_y_line_2_ptr_[0 : BLy2])
#pragma acc data copyin(send_buf_y_line_3_ptr_[0 : BLy3])
#pragma acc data copyin(recv_buf_y_line_3_ptr_[0 : BLy3])
#pragma acc data copyin(send_buf_z_line_0_ptr_[0 : BLz0])
#pragma acc data copyin(recv_buf_z_line_0_ptr_[0 : BLz0])
#pragma acc data copyin(send_buf_z_line_1_ptr_[0 : BLz1])
#pragma acc data copyin(recv_buf_z_line_1_ptr_[0 : BLz1])
#pragma acc data copyin(send_buf_z_line_2_ptr_[0 : BLz2])
#pragma acc data copyin(recv_buf_z_line_2_ptr_[0 : BLz2])
#pragma acc data copyin(send_buf_z_line_3_ptr_[0 : BLz3])
#pragma acc data copyin(recv_buf_z_line_3_ptr_[0 : BLz3])
#pragma acc data copyin(outer_elems_ptr_[0 : OE])

#pragma acc parallel loop collapse(3) private(idx, idx_element, l, m, n)
    for (int k = 0; k < grid_size_z_ / order_; ++k)
    {
        for (int j = 0; j < grid_size_y_ / order_; ++j)
        {
            for (int i = 0; i < grid_size_x_ / order_; ++i)
            {
                idx = (i + j * grid_size_x_ / order_ + k * grid_size_x_ / order_ * grid_size_y_ / order_) * order_ * (order_ + 1) * (order_ + 1);
                idx_element =
                    order_ * (i + j * grid_size_x_ + k * grid_size_x_ * grid_size_y_);
#pragma acc loop seq
                for (int n = 0; n < order_ + 1; ++n)
                {
#pragma acc loop seq
                    for (int m = 0; m < order_ + 1; ++m)
                    {
#pragma acc loop seq
                        for (int l = 0; l < order_; ++l)
                        {
                            conn_x_ptr_[idx + l + m * order_ + n * order_ * (order_ + 1)] =
                                idx_element + k * order_ * grid_size_x_ + l + m * grid_size_x_ + n * (grid_size_y_ + 1) * grid_size_x_;
                        }
                    }
                }
#pragma acc loop seq
                for (int n = 0; n < order_ + 1; ++n)
                {
#pragma acc loop seq
                    for (int m = 0; m < order_; ++m)
                    {
#pragma acc loop seq
                        for (int l = 0; l < order_ + 1; ++l)
                        {
                            conn_y_ptr_[idx + l + m * (order_ + 1) + n * order_ * (order_ + 1)] =
                                idx_element + j * order_ + k * order_ * grid_size_y_ + l + m * (grid_size_x_ + 1) + n * grid_size_y_ * (grid_size_x_ + 1);
                        }
                    }
                }
#pragma acc loop seq
                for (int n = 0; n < order_; ++n)
                {
#pragma acc loop seq
                    for (int m = 0; m < order_ + 1; ++m)
                    {
#pragma acc loop seq
                        for (int l = 0; l < order_ + 1; ++l)
                        {
                            conn_z_ptr_[idx + l + m * (order_ + 1) + n * (order_ + 1) * (order_ + 1)] =
                                idx_element + j * order_ + k * order_ * (grid_size_x_ + grid_size_y_ + 1) + l + m * (grid_size_x_ + 1) + n * (grid_size_y_ + 1) * (grid_size_x_ + 1);
                        }
                    }
                }
            }
        }
    }
#pragma acc parallel loop collapse(2)
    for (int j = 0; j < grid_size_y_ / order_; ++j)
    {
        for (int i = 0; i < grid_size_x_ / order_; ++i)
        {
            outer_elems_ptr_[i + j * grid_size_x_ / order_] = i + j * grid_size_x_ / order_;
            outer_elems_ptr_[grid_size_x_ / order_ * grid_size_y_ / order_ + i + j * grid_size_x_ / order_] =
                grid_size_x_ / order_ * grid_size_y_ / order_ * (grid_size_z_ / order_ - 1) + i + j * grid_size_x_ / order_;
        }
    }
#pragma acc parallel loop collapse(2)
    for (int k = 1; k < grid_size_z_ / order_ - 1; ++k)
    {
        for (int j = 0; j < grid_size_y_ / order_; ++j)
        {
            outer_elems_ptr_[2 * grid_size_x_ / order_ * grid_size_y_ / order_ + j + (k - 1) * grid_size_y_ / order_] =
                j * grid_size_x_ / order_ + k * grid_size_x_ / order_ * grid_size_y_ / order_;
            outer_elems_ptr_[2 * grid_size_x_ / order_ * grid_size_y_ / order_ + grid_size_y_ / order_ * (grid_size_z_ / order_ - 2) + j + (k - 1) * grid_size_y_ / order_] =
                (grid_size_x_ / order_ - 1) + j * grid_size_x_ / order_ + k * grid_size_x_ / order_ * grid_size_y_ / order_;
        }
    }
#pragma acc parallel loop collapse(2)
    for (int i = 1; i < grid_size_x_ / order_ - 1; ++i)
    {
        for (int k = 1; k < grid_size_z_ / order_ - 1; ++k)
        {
            outer_elems_ptr_[2 * grid_size_x_ / order_ * grid_size_y_ / order_ + 2 * grid_size_y_ / order_ * (grid_size_z_ / order_ - 2) + k - 1 + (i - 1) * (grid_size_z_ / order_ - 2)] =
                i + k * grid_size_x_ / order_ * grid_size_y_ / order_;
            outer_elems_ptr_[2 * grid_size_x_ / order_ * grid_size_y_ / order_ + 2 * grid_size_y_ / order_ * (grid_size_z_ / order_ - 2) + (grid_size_z_ / order_ - 2) * (grid_size_x_ / order_ - 2) + k - 1 + (i - 1) * (grid_size_z_ / order_ - 2)] =
                grid_size_x_ / order_ * (grid_size_y_ / order_ - 1) + i + k * grid_size_x_ / order_ * grid_size_y_ / order_;
        }
    }
}

void FemSimulation::calculateElementStiffnessMatrix()
{
    mat_size_ = order_ * (order_ + 1) * (order_ + 1) * 3;
    std::string filename = "kmat/" + std::to_string(order_) + ".csv";
    if (use_ofem_ == 0)
    {
        filename = "kmat/lamped" + std::to_string(order_) + ".csv";
    }
    auto tempMat = loadMatrixFromCSV(filename);

    Eigen::MatrixXd Kmat_Eigen(order_ * (order_ + 1) * (order_ + 1) * 3, order_ * (order_ + 1) * (order_ + 1) * 3);
    for (int i = 0; i < order_ * (order_ + 1) * (order_ + 1) * 3; ++i)
    {
        for (int j = 0; j < order_ * (order_ + 1) * (order_ + 1) * 3; ++j)
        {
            Kmat_Eigen(i, j) = tempMat[i][j];
        }
    }
    Kmat_Eigen = 4.0 / (domain_size_ * order_) / (domain_size_ * order_) / permeability_ / permittivity_ * Kmat_Eigen;

    // 1次元配列として確保
    element_stiffness_matrix_.resize(mat_size_ * mat_size_);
    for (int i = 0; i < mat_size_; i++)
    {
        for (int j = 0; j < mat_size_; j++)
        {
            element_stiffness_matrix_[i * mat_size_ + j] = Kmat_Eigen(i, j);
        }
    }

    element_stiffness_matrix_ptr_ = element_stiffness_matrix_.data();
#pragma acc data copyin(element_stiffness_matrix_ptr_[0 : mat_size_ * mat_size_])
}

void FemSimulation::setSource_x(const std::array<int, 3> &position)
{
    if (position[0] >= grid_size_x_ * coord_x_ && position[0] <= grid_size_x_ * (coord_x_ + 1) &&
        position[1] >= grid_size_y_ * coord_y_ && position[1] <= grid_size_y_ * (coord_y_ + 1) &&
        position[2] >= grid_size_z_ * coord_z_ && position[2] <= grid_size_z_ * (coord_z_ + 1))
    {
        int i = position[0] - grid_size_x_ * coord_x_;
        int j = position[1] - grid_size_y_ * coord_y_;
        int k = position[2] - grid_size_z_ * coord_z_;

        // Ez配列のインデックスを直接計算
        int idx = i + j * grid_size_x_ + k * grid_size_x_ * (grid_size_y_ + 1);
        source_position_x_.push_back(idx);
    }
}

void FemSimulation::setSource_y(const std::array<int, 3> &position)
{
    if (position[0] >= grid_size_x_ * coord_x_ && position[0] <= grid_size_x_ * (coord_x_ + 1) &&
        position[1] >= grid_size_y_ * coord_y_ && position[1] <= grid_size_y_ * (coord_y_ + 1) &&
        position[2] >= grid_size_z_ * coord_z_ && position[2] <= grid_size_z_ * (coord_z_ + 1))
    {
        int i = position[0] - grid_size_x_ * coord_x_;
        int j = position[1] - grid_size_y_ * coord_y_;
        int k = position[2] - grid_size_z_ * coord_z_;

        // Ez配列のインデックスを直接計算
        int idx = i + j * (grid_size_x_ + 1) + k * (grid_size_x_ + 1) * grid_size_y_;
        source_position_y_.push_back(idx);
    }
}

void FemSimulation::setSource_z(const std::array<int, 3> &position)
{
    if (position[0] >= grid_size_x_ * coord_x_ && position[0] <= grid_size_x_ * (coord_x_ + 1) &&
        position[1] >= grid_size_y_ * coord_y_ && position[1] <= grid_size_y_ * (coord_y_ + 1) &&
        position[2] >= grid_size_z_ * coord_z_ && position[2] <= grid_size_z_ * (coord_z_ + 1))
    {
        int i = position[0] - grid_size_x_ * coord_x_;
        int j = position[1] - grid_size_y_ * coord_y_;
        int k = position[2] - grid_size_z_ * coord_z_;

        // Ez配列のインデックスを直接計算
        int idx = i + j * (grid_size_x_ + 1) + k * (grid_size_x_ + 1) * (grid_size_y_ + 1);
        source_position_z_.push_back(idx);
    }
}

void FemSimulation::setObservationPoint(const std::array<int, 3> &position)
{
    if (position[0] >= grid_size_x_ * coord_x_ && position[0] <= grid_size_x_ * (coord_x_ + 1) &&
        position[1] >= grid_size_y_ * coord_y_ && position[1] <= grid_size_y_ * (coord_y_ + 1) &&
        position[2] >= grid_size_z_ * coord_z_ && position[2] <= grid_size_z_ * (coord_z_ + 1))
    {
        int i = position[0] - grid_size_x_ * coord_x_;
        int j = position[1] - grid_size_y_ * coord_y_;
        int k = position[2] - grid_size_z_ * coord_z_;

        std::array<int, 3> obs;
        obs[0] = i + j * grid_size_x_ + k * grid_size_x_ * (grid_size_y_ + 1);
        obs[1] = i + j * (grid_size_x_ + 1) + k * (grid_size_x_ + 1) * grid_size_y_;
        obs[2] = i + j * (grid_size_x_ + 1) + k * (grid_size_x_ + 1) * (grid_size_y_ + 1);
        observation_points_.push_back(obs);

        saved_electric_field_.push_back(std::vector<std::array<double, 4>>());
    }
}

void FemSimulation::updateTimeStep(const double deltat, const double offset)
{
    int idx;
    double temp;
    int l, m, n;

    // 初期化
    // Uが0,Pが1,Pの更新に使うのが2
#pragma acc parallel loop
    for (int i = ef_x_idx_2_; i < ef_x_idx_2_ + grid_size_x_ * (grid_size_y_ + 1) * (grid_size_z_ + 1); ++i)
    {
        ef_x_ptr_[i] = 0.0;
    }
#pragma acc parallel loop
    for (int i = ef_y_idx_2_; i < ef_y_idx_2_ + (grid_size_x_ + 1) * grid_size_y_ * (grid_size_z_ + 1); ++i)
    {
        ef_y_ptr_[i] = 0.0;
    }
#pragma acc parallel loop
    for (int i = ef_z_idx_2_; i < ef_z_idx_2_ + (grid_size_x_ + 1) * (grid_size_y_ + 1) * grid_size_z_; ++i)
    {
        ef_z_ptr_[i] = 0.0;
    }

    // 2次精度シンプレクティック

    // U(E)の更新
#pragma acc parallel loop
    for (int i = 0; i < grid_size_x_ * (grid_size_y_ + 1) * (grid_size_z_ + 1); ++i)
    {
        ef_x_ptr_[ef_x_idx_0_ + i] +=
            0.5 * deltat * ef_x_ptr_[ef_x_idx_1_ + i] / (8.0 / order_ / order_ / order_);
    }
#pragma acc parallel loop
    for (int i = 0; i < (grid_size_x_ + 1) * grid_size_y_ * (grid_size_z_ + 1); ++i)
    {
        ef_y_ptr_[ef_y_idx_0_ + i] +=
            0.5 * deltat * ef_y_ptr_[ef_y_idx_1_ + i] / (8.0 / order_ / order_ / order_);
    }
#pragma acc parallel loop
    for (int i = 0; i < (grid_size_x_ + 1) * (grid_size_y_ + 1) * grid_size_z_; ++i)
    {
        ef_z_ptr_[ef_z_idx_0_ + i] +=
            0.5 * deltat * ef_z_ptr_[ef_z_idx_1_ + i] / (8.0 / order_ / order_ / order_);
    }

    // P(M・dE/dt)の更新
    // 要素ごとの剛性行列の計算(外側要素のみ)
    dispatchOuterElements(
        order_,
        ef_x_ptr_, ef_y_ptr_, ef_z_ptr_,
        conn_x_ptr_, conn_y_ptr_, conn_z_ptr_,
        outer_elems_ptr_, element_stiffness_matrix_ptr_,
        ef_x_idx_0_, ef_x_idx_2_,
        ef_y_idx_0_, ef_y_idx_2_,
        ef_z_idx_0_, ef_z_idx_2_,
        OE, mat_size_);

    // 外力項の計算
#pragma acc parallel loop
    for (size_t i = 0; i < source_position_z_.size(); ++i)
    {
        double temp = source_function(current_time_ + offset + 0.5 * deltat, permeability_, domain_size_) * 8.0 / permittivity_ / (domain_size_ * order_) / (domain_size_ * order_) / (domain_size_ * order_);
        ef_z_ptr_[ef_z_idx_2_ + src_pos_z_ptr_[i]] += temp;
    }

    // MPI通信
    start_communication_time_ = MPI_Wtime();
    startExchangeElectricField();
    end_communication_time_ = MPI_Wtime();
    total_communication_time_ += end_communication_time_ - start_communication_time_;

    // 要素ごとの剛性行列の計算(内側要素のみ)
    dispatchInnerElements(
        order_,
        ef_x_ptr_, ef_y_ptr_, ef_z_ptr_,
        conn_x_ptr_, conn_y_ptr_, conn_z_ptr_,
        element_stiffness_matrix_ptr_,
        ef_x_idx_0_, ef_x_idx_2_,
        ef_y_idx_0_, ef_y_idx_2_,
        ef_z_idx_0_, ef_z_idx_2_,
        grid_size_x_, grid_size_y_, grid_size_z_,
        mat_size_);

    // MPI通信完了待ち
    start_communication_time_ = MPI_Wtime();
    finishExchangeElectricField();
    end_communication_time_ = MPI_Wtime();
    total_communication_time_ += end_communication_time_ - start_communication_time_;

    // U(E)の更新(Pの更新も兼ねる)
#pragma acc parallel loop
    for (int i = 0; i < grid_size_x_ * (grid_size_y_ + 1) * (grid_size_z_ + 1); ++i)
    {
        ef_x_ptr_[ef_x_idx_1_ + i] -= deltat * ef_x_ptr_[ef_x_idx_2_ + i];
        ef_x_ptr_[ef_x_idx_0_ + i] +=
            0.5 * deltat * ef_x_ptr_[ef_x_idx_1_ + i] / (8.0 / order_ / order_ / order_);
    }
#pragma acc parallel loop
    for (int i = 0; i < (grid_size_x_ + 1) * grid_size_y_ * (grid_size_z_ + 1); ++i)
    {
        ef_y_ptr_[ef_y_idx_1_ + i] -= deltat * ef_y_ptr_[ef_y_idx_2_ + i];
        ef_y_ptr_[ef_y_idx_0_ + i] +=
            0.5 * deltat * ef_y_ptr_[ef_y_idx_1_ + i] / (8.0 / order_ / order_ / order_);
    }
#pragma acc parallel loop
    for (int i = 0; i < (grid_size_x_ + 1) * (grid_size_y_ + 1) * grid_size_z_; ++i)
    {
        ef_z_ptr_[ef_z_idx_1_ + i] -= deltat * ef_z_ptr_[ef_z_idx_2_ + i];
        ef_z_ptr_[ef_z_idx_0_ + i] +=
            0.5 * deltat * ef_z_ptr_[ef_z_idx_1_ + i] / (8.0 / order_ / order_ / order_);
    }

    // 境界条件
    applyBoundaryConditions();
}

void FemSimulation::startExchangeElectricField()
{
    MPI_Comm comm_0_, comm_1_, comm_2_, comm_3_;

    comm_0_ = coord_x_ % 2 == 0 ? comm_x_plane_0_ : comm_x_plane_1_;
    comm_1_ = coord_x_ % 2 == 0 ? comm_x_plane_1_ : comm_x_plane_0_;
#pragma acc parallel loop collapse(2)
    for (int i = 0; i < grid_size_y_; ++i)
    {
        for (int j = 0; j < grid_size_z_ + 1; ++j)
        {
            send_buf_x_plane_0_y_ptr_[i + grid_size_y_ * j] =
                ef_y_ptr_[ef_y_idx_2_ + i * (grid_size_x_ + 1) + j * (grid_size_x_ + 1) * grid_size_y_];
            send_buf_x_plane_1_y_ptr_[i + grid_size_y_ * j] =
                ef_y_ptr_[ef_y_idx_2_ + grid_size_x_ + i * (grid_size_x_ + 1) + j * (grid_size_x_ + 1) * grid_size_y_];
        }
    }
#pragma acc host_data use_device(send_buf_x_plane_0_y_ptr_, send_buf_x_plane_1_y_ptr_, recv_buf_x_plane_0_y_ptr_, recv_buf_x_plane_1_y_ptr_)
    {
        if (coord_x_ != 0)
        {
            MPI_Iallreduce(send_buf_x_plane_0_y_ptr_, recv_buf_x_plane_0_y_ptr_,
                           BNx0y, MPI_DOUBLE, MPI_SUM, comm_0_, &reqs_[0]);
        }
        if (coord_x_ != dim_x_ - 1)
        {
            MPI_Iallreduce(send_buf_x_plane_1_y_ptr_, recv_buf_x_plane_1_y_ptr_,
                           BNx1y, MPI_DOUBLE, MPI_SUM, comm_1_, &reqs_[1]);
        }
    }
#pragma acc parallel loop collapse(2)
    for (int i = 0; i < grid_size_y_ + 1; ++i)
    {
        for (int j = 0; j < grid_size_z_; ++j)
        {
            send_buf_x_plane_0_z_ptr_[i + (grid_size_y_ + 1) * j] =
                ef_z_ptr_[ef_z_idx_2_ + i * (grid_size_x_ + 1) + j * (grid_size_x_ + 1) * (grid_size_y_ + 1)];
            send_buf_x_plane_1_z_ptr_[i + (grid_size_y_ + 1) * j] =
                ef_z_ptr_[ef_z_idx_2_ + grid_size_x_ + i * (grid_size_x_ + 1) + j * (grid_size_x_ + 1) * (grid_size_y_ + 1)];
        }
    }
#pragma acc host_data use_device(send_buf_x_plane_0_z_ptr_, send_buf_x_plane_1_z_ptr_, recv_buf_x_plane_0_z_ptr_, recv_buf_x_plane_1_z_ptr_)
    {
        if (coord_x_ != 0)
        {
            MPI_Iallreduce(send_buf_x_plane_0_z_ptr_, recv_buf_x_plane_0_z_ptr_,
                           BNx0z, MPI_DOUBLE, MPI_SUM, comm_0_, &reqs_[2]);
        }
        if (coord_x_ != dim_x_ - 1)
        {
            MPI_Iallreduce(send_buf_x_plane_1_z_ptr_, recv_buf_x_plane_1_z_ptr_,
                           BNx1z, MPI_DOUBLE, MPI_SUM, comm_1_, &reqs_[3]);
        }
    }

    comm_0_ = coord_y_ % 2 == 0 ? comm_y_plane_0_ : comm_y_plane_1_;
    comm_1_ = coord_y_ % 2 == 0 ? comm_y_plane_1_ : comm_y_plane_0_;
#pragma acc parallel loop collapse(2)
    for (int i = 0; i < grid_size_x_ + 1; ++i)
    {
        for (int j = 0; j < grid_size_z_; ++j)
        {
            send_buf_y_plane_0_z_ptr_[i + (grid_size_x_ + 1) * j] =
                ef_z_ptr_[ef_z_idx_2_ + i + j * (grid_size_x_ + 1) * (grid_size_y_ + 1)];
            send_buf_y_plane_1_z_ptr_[i + (grid_size_x_ + 1) * j] =
                ef_z_ptr_[ef_z_idx_2_ + i + grid_size_y_ * (grid_size_x_ + 1) + j * (grid_size_x_ + 1) * (grid_size_y_ + 1)];
        }
    }
#pragma acc host_data use_device(send_buf_y_plane_0_z_ptr_, send_buf_y_plane_1_z_ptr_, recv_buf_y_plane_0_z_ptr_, recv_buf_y_plane_1_z_ptr_)
    {
        if (coord_y_ != 0)
        {
            MPI_Iallreduce(send_buf_y_plane_0_z_ptr_, recv_buf_y_plane_0_z_ptr_,
                           BNy0z, MPI_DOUBLE, MPI_SUM, comm_0_, &reqs_[4]);
        }
        if (coord_y_ != dim_y_ - 1)
        {
            MPI_Iallreduce(send_buf_y_plane_1_z_ptr_, recv_buf_y_plane_1_z_ptr_,
                           BNy1z, MPI_DOUBLE, MPI_SUM, comm_1_, &reqs_[5]);
        }
    }
#pragma acc parallel loop collapse(2)
    for (int i = 0; i < grid_size_x_; ++i)
    {
        for (int j = 0; j < grid_size_z_ + 1; ++j)
        {
            send_buf_y_plane_0_x_ptr_[i + grid_size_x_ * j] =
                ef_x_ptr_[ef_x_idx_2_ + i + j * grid_size_x_ * (grid_size_y_ + 1)];
            send_buf_y_plane_1_x_ptr_[i + grid_size_x_ * j] =
                ef_x_ptr_[ef_x_idx_2_ + i + grid_size_y_ * grid_size_x_ + j * grid_size_x_ * (grid_size_y_ + 1)];
        }
    }
#pragma acc host_data use_device(send_buf_y_plane_0_x_ptr_, send_buf_y_plane_1_x_ptr_, recv_buf_y_plane_0_x_ptr_, recv_buf_y_plane_1_x_ptr_)
    {
        if (coord_y_ != 0)
        {
            MPI_Iallreduce(send_buf_y_plane_0_x_ptr_, recv_buf_y_plane_0_x_ptr_,
                           BNy0x, MPI_DOUBLE, MPI_SUM, comm_0_, &reqs_[6]);
        }
        if (coord_y_ != dim_y_ - 1)
        {
            MPI_Iallreduce(send_buf_y_plane_1_x_ptr_, recv_buf_y_plane_1_x_ptr_,
                           BNy1x, MPI_DOUBLE, MPI_SUM, comm_1_, &reqs_[7]);
        }
    }

    comm_0_ = coord_z_ % 2 == 0 ? comm_z_plane_0_ : comm_z_plane_1_;
    comm_1_ = coord_z_ % 2 == 0 ? comm_z_plane_1_ : comm_z_plane_0_;
#pragma acc parallel loop collapse(2)
    for (int i = 0; i < grid_size_x_; ++i)
    {
        for (int j = 0; j < grid_size_y_ + 1; ++j)
        {
            send_buf_z_plane_0_x_ptr_[i + grid_size_x_ * j] =
                ef_x_ptr_[ef_x_idx_2_ + i + j * grid_size_x_];
            send_buf_z_plane_1_x_ptr_[i + grid_size_x_ * j] =
                ef_x_ptr_[ef_x_idx_2_ + i + j * grid_size_x_ + grid_size_x_ * (grid_size_y_ + 1) * grid_size_z_];
        }
    }
#pragma acc host_data use_device(send_buf_z_plane_0_x_ptr_, send_buf_z_plane_1_x_ptr_, recv_buf_z_plane_0_x_ptr_, recv_buf_z_plane_1_x_ptr_)
    {
        if (coord_z_ != 0)
        {
            MPI_Iallreduce(send_buf_z_plane_0_x_ptr_, recv_buf_z_plane_0_x_ptr_,
                           BNz0x, MPI_DOUBLE, MPI_SUM, comm_0_, &reqs_[8]);
        }
        if (coord_z_ != dim_z_ - 1)
        {
            MPI_Iallreduce(send_buf_z_plane_1_x_ptr_, recv_buf_z_plane_1_x_ptr_,
                           BNz1x, MPI_DOUBLE, MPI_SUM, comm_1_, &reqs_[9]);
        }
    }
#pragma acc parallel loop collapse(2)
    for (int i = 0; i < grid_size_x_ + 1; ++i)
    {
        for (int j = 0; j < grid_size_y_; ++j)
        {
            send_buf_z_plane_0_y_ptr_[i + (grid_size_x_ + 1) * j] =
                ef_y_ptr_[ef_y_idx_2_ + i + j * (grid_size_x_ + 1)];
            send_buf_z_plane_1_y_ptr_[i + (grid_size_x_ + 1) * j] =
                ef_y_ptr_[ef_y_idx_2_ + i + j * (grid_size_x_ + 1) + (grid_size_x_ + 1) * grid_size_y_ * grid_size_z_];
        }
    }
#pragma acc host_data use_device(send_buf_z_plane_0_y_ptr_, send_buf_z_plane_1_y_ptr_, recv_buf_z_plane_0_y_ptr_, recv_buf_z_plane_1_y_ptr_)
    {
        if (coord_z_ != 0)
        {
            MPI_Iallreduce(send_buf_z_plane_0_y_ptr_, recv_buf_z_plane_0_y_ptr_,
                           BNz0y, MPI_DOUBLE, MPI_SUM, comm_0_, &reqs_[10]);
        }
        if (coord_z_ != dim_z_ - 1)
        {
            MPI_Iallreduce(send_buf_z_plane_1_y_ptr_, recv_buf_z_plane_1_y_ptr_,
                           BNz1y, MPI_DOUBLE, MPI_SUM, comm_1_, &reqs_[11]);
        }
    }

    if (coord_y_ % 2 == 0 && coord_z_ % 2 == 0)
    {
        comm_0_ = comm_x_line_0_;
        comm_1_ = comm_x_line_1_;
        comm_2_ = comm_x_line_2_;
        comm_3_ = comm_x_line_3_;
    }
    else if (coord_y_ % 2 == 1 && coord_z_ % 2 == 0)
    {
        comm_0_ = comm_x_line_1_;
        comm_1_ = comm_x_line_0_;
        comm_2_ = comm_x_line_3_;
        comm_3_ = comm_x_line_2_;
    }
    else if (coord_y_ % 2 == 0 && coord_z_ % 2 == 1)
    {
        comm_0_ = comm_x_line_2_;
        comm_1_ = comm_x_line_3_;
        comm_2_ = comm_x_line_0_;
        comm_3_ = comm_x_line_1_;
    }
    else
    {
        comm_0_ = comm_x_line_3_;
        comm_1_ = comm_x_line_2_;
        comm_2_ = comm_x_line_1_;
        comm_3_ = comm_x_line_0_;
    }
#pragma acc parallel loop
    for (int i = 0; i < grid_size_x_; ++i)
    {
        send_buf_x_line_0_ptr_[i] =
            ef_x_ptr_[ef_x_idx_2_ + i];
        send_buf_x_line_1_ptr_[i] =
            ef_x_ptr_[ef_x_idx_2_ + i + grid_size_x_ * grid_size_y_];
        send_buf_x_line_2_ptr_[i] =
            ef_x_ptr_[ef_x_idx_2_ + i + grid_size_x_ * (grid_size_y_ + 1) * grid_size_z_];
        send_buf_x_line_3_ptr_[i] =
            ef_x_ptr_[ef_x_idx_2_ + i + grid_size_x_ * (grid_size_y_ + 1) * grid_size_z_ + grid_size_x_ * grid_size_y_];
    }
#pragma acc host_data use_device(send_buf_x_line_0_ptr_, send_buf_x_line_1_ptr_, send_buf_x_line_2_ptr_, send_buf_x_line_3_ptr_, recv_buf_x_line_0_ptr_, recv_buf_x_line_1_ptr_, recv_buf_x_line_2_ptr_, recv_buf_x_line_3_ptr_)
    {
        if (coord_y_ != 0 && coord_z_ != 0)
        {
            MPI_Iallreduce(send_buf_x_line_0_ptr_, recv_buf_x_line_0_ptr_,
                           BLx0, MPI_DOUBLE, MPI_SUM, comm_0_, &reqs_[12]);
        }
        if (coord_y_ != dim_y_ - 1 && coord_z_ != 0)
        {
            MPI_Iallreduce(send_buf_x_line_1_ptr_, recv_buf_x_line_1_ptr_,
                           BLx1, MPI_DOUBLE, MPI_SUM, comm_1_, &reqs_[13]);
        }
        if (coord_y_ != 0 && coord_z_ != dim_z_ - 1)
        {
            MPI_Iallreduce(send_buf_x_line_2_ptr_, recv_buf_x_line_2_ptr_,
                           BLx2, MPI_DOUBLE, MPI_SUM, comm_2_, &reqs_[14]);
        }
        if (coord_y_ != dim_y_ - 1 && coord_z_ != dim_z_ - 1)
        {
            MPI_Iallreduce(send_buf_x_line_3_ptr_, recv_buf_x_line_3_ptr_,
                           BLx3, MPI_DOUBLE, MPI_SUM, comm_3_, &reqs_[15]);
        }
    }

    if (coord_z_ % 2 == 0 && coord_x_ % 2 == 0)
    {
        comm_0_ = comm_y_line_0_;
        comm_1_ = comm_y_line_1_;
        comm_2_ = comm_y_line_2_;
        comm_3_ = comm_y_line_3_;
    }
    else if (coord_z_ % 2 == 1 && coord_x_ % 2 == 0)
    {
        comm_0_ = comm_y_line_1_;
        comm_1_ = comm_y_line_0_;
        comm_2_ = comm_y_line_3_;
        comm_3_ = comm_y_line_2_;
    }
    else if (coord_z_ % 2 == 0 && coord_x_ % 2 == 1)
    {
        comm_0_ = comm_y_line_2_;
        comm_1_ = comm_y_line_3_;
        comm_2_ = comm_y_line_0_;
        comm_3_ = comm_y_line_1_;
    }
    else
    {
        comm_0_ = comm_y_line_3_;
        comm_1_ = comm_y_line_2_;
        comm_2_ = comm_y_line_1_;
        comm_3_ = comm_y_line_0_;
    }
#pragma acc parallel loop
    for (int i = 0; i < grid_size_y_; ++i)
    {
        send_buf_y_line_0_ptr_[i] =
            ef_y_ptr_[ef_y_idx_2_ + i * (grid_size_x_ + 1)];
        send_buf_y_line_1_ptr_[i] =
            ef_y_ptr_[ef_y_idx_2_ + i * (grid_size_x_ + 1) + (grid_size_x_ + 1) * grid_size_y_ * grid_size_z_];
        send_buf_y_line_2_ptr_[i] =
            ef_y_ptr_[ef_y_idx_2_ + i * (grid_size_x_ + 1) + grid_size_x_];
        send_buf_y_line_3_ptr_[i] =
            ef_y_ptr_[ef_y_idx_2_ + i * (grid_size_x_ + 1) + (grid_size_x_ + 1) * grid_size_y_ * grid_size_z_ + grid_size_x_];
    }
#pragma acc host_data use_device(send_buf_y_line_0_ptr_, send_buf_y_line_1_ptr_, send_buf_y_line_2_ptr_, send_buf_y_line_3_ptr_, recv_buf_y_line_0_ptr_, recv_buf_y_line_1_ptr_, recv_buf_y_line_2_ptr_, recv_buf_y_line_3_ptr_)
    {
        if (coord_z_ != 0 && coord_x_ != 0)
        {
            MPI_Iallreduce(send_buf_y_line_0_ptr_, recv_buf_y_line_0_ptr_,
                           BLy0, MPI_DOUBLE, MPI_SUM, comm_0_, &reqs_[16]);
        }
        if (coord_z_ != dim_z_ - 1 && coord_x_ != 0)
        {
            MPI_Iallreduce(send_buf_y_line_1_ptr_, recv_buf_y_line_1_ptr_,
                           BLy1, MPI_DOUBLE, MPI_SUM, comm_1_, &reqs_[17]);
        }
        if (coord_z_ != 0 && coord_x_ != dim_x_ - 1)
        {
            MPI_Iallreduce(send_buf_y_line_2_ptr_, recv_buf_y_line_2_ptr_,
                           BLy2, MPI_DOUBLE, MPI_SUM, comm_2_, &reqs_[18]);
        }
        if (coord_z_ != dim_z_ - 1 && coord_x_ != dim_x_ - 1)
        {
            MPI_Iallreduce(send_buf_y_line_3_ptr_, recv_buf_y_line_3_ptr_,
                           BLy3, MPI_DOUBLE, MPI_SUM, comm_3_, &reqs_[19]);
        }
    }

    if (coord_x_ % 2 == 0 && coord_y_ % 2 == 0)
    {
        comm_0_ = comm_z_line_0_;
        comm_1_ = comm_z_line_1_;
        comm_2_ = comm_z_line_2_;
        comm_3_ = comm_z_line_3_;
    }
    else if (coord_x_ % 2 == 1 && coord_y_ % 2 == 0)
    {
        comm_0_ = comm_z_line_1_;
        comm_1_ = comm_z_line_0_;
        comm_2_ = comm_z_line_3_;
        comm_3_ = comm_z_line_2_;
    }
    else if (coord_x_ % 2 == 0 && coord_y_ % 2 == 1)
    {
        comm_0_ = comm_z_line_2_;
        comm_1_ = comm_z_line_3_;
        comm_2_ = comm_z_line_0_;
        comm_3_ = comm_z_line_1_;
    }
    else
    {
        comm_0_ = comm_z_line_3_;
        comm_1_ = comm_z_line_2_;
        comm_2_ = comm_z_line_1_;
        comm_3_ = comm_z_line_0_;
    }
#pragma acc parallel loop
    for (int i = 0; i < grid_size_z_; ++i)
    {
        send_buf_z_line_0_ptr_[i] =
            ef_z_ptr_[ef_z_idx_2_ + i * (grid_size_x_ + 1) * (grid_size_y_ + 1)];
        send_buf_z_line_1_ptr_[i] =
            ef_z_ptr_[ef_z_idx_2_ + i * (grid_size_x_ + 1) * (grid_size_y_ + 1) + grid_size_x_];
        send_buf_z_line_2_ptr_[i] =
            ef_z_ptr_[ef_z_idx_2_ + i * (grid_size_x_ + 1) * (grid_size_y_ + 1) + (grid_size_x_ + 1) * grid_size_y_];
        send_buf_z_line_3_ptr_[i] =
            ef_z_ptr_[ef_z_idx_2_ + i * (grid_size_x_ + 1) * (grid_size_y_ + 1) + (grid_size_x_ + 1) * grid_size_y_ + grid_size_x_];
    }
#pragma acc host_data use_device(send_buf_z_line_0_ptr_, send_buf_z_line_1_ptr_, send_buf_z_line_2_ptr_, send_buf_z_line_3_ptr_, recv_buf_z_line_0_ptr_, recv_buf_z_line_1_ptr_, recv_buf_z_line_2_ptr_, recv_buf_z_line_3_ptr_)
    {
        if (coord_x_ != 0 && coord_y_ != 0)
        {
            MPI_Iallreduce(send_buf_z_line_0_ptr_, recv_buf_z_line_0_ptr_,
                           BLz0, MPI_DOUBLE, MPI_SUM, comm_0_, &reqs_[20]);
        }
        if (coord_x_ != dim_x_ - 1 && coord_y_ != 0)
        {
            MPI_Iallreduce(send_buf_z_line_1_ptr_, recv_buf_z_line_1_ptr_,
                           BLz1, MPI_DOUBLE, MPI_SUM, comm_1_, &reqs_[21]);
        }
        if (coord_x_ != 0 && coord_y_ != dim_y_ - 1)
        {
            MPI_Iallreduce(send_buf_z_line_2_ptr_, recv_buf_z_line_2_ptr_,
                           BLz2, MPI_DOUBLE, MPI_SUM, comm_2_, &reqs_[22]);
        }
        if (coord_x_ != dim_x_ - 1 && coord_y_ != dim_y_ - 1)
        {
            MPI_Iallreduce(send_buf_z_line_3_ptr_, recv_buf_z_line_3_ptr_,
                           BLz3, MPI_DOUBLE, MPI_SUM, comm_3_, &reqs_[23]);
        }
    }
}

void FemSimulation::finishExchangeElectricField()
{
    MPI_Waitall(24, reqs_, MPI_STATUSES_IGNORE);

#pragma acc parallel loop collapse(2)
    for (int i = 0; i < grid_size_y_; ++i)
    {
        for (int j = 1; j < grid_size_z_; ++j)
        {
            ef_y_ptr_[ef_y_idx_2_ + i * (grid_size_x_ + 1) + j * (grid_size_x_ + 1) * grid_size_y_] =
                recv_buf_x_plane_0_y_ptr_[i + grid_size_y_ * j];
            ef_y_ptr_[ef_y_idx_2_ + grid_size_x_ + i * (grid_size_x_ + 1) + j * (grid_size_x_ + 1) * grid_size_y_] =
                recv_buf_x_plane_1_y_ptr_[i + grid_size_y_ * j];
        }
    }
#pragma acc parallel loop collapse(2)
    for (int i = 1; i < grid_size_y_; ++i)
    {
        for (int j = 0; j < grid_size_z_; ++j)
        {
            ef_z_ptr_[ef_z_idx_2_ + i * (grid_size_x_ + 1) + j * (grid_size_x_ + 1) * (grid_size_y_ + 1)] =
                recv_buf_x_plane_0_z_ptr_[i + (grid_size_y_ + 1) * j];
            ef_z_ptr_[ef_z_idx_2_ + grid_size_x_ + i * (grid_size_x_ + 1) + j * (grid_size_x_ + 1) * (grid_size_y_ + 1)] =
                recv_buf_x_plane_1_z_ptr_[i + (grid_size_y_ + 1) * j];
        }
    }
#pragma acc parallel loop collapse(2)
    for (int i = 1; i < grid_size_x_; ++i)
    {
        for (int j = 0; j < grid_size_z_; ++j)
        {
            ef_z_ptr_[ef_z_idx_2_ + i + j * (grid_size_x_ + 1) * (grid_size_y_ + 1)] =
                recv_buf_y_plane_0_z_ptr_[i + (grid_size_x_ + 1) * j];
            ef_z_ptr_[ef_z_idx_2_ + i + grid_size_y_ * (grid_size_x_ + 1) + j * (grid_size_x_ + 1) * (grid_size_y_ + 1)] =
                recv_buf_y_plane_1_z_ptr_[i + (grid_size_x_ + 1) * j];
        }
    }
#pragma acc parallel loop collapse(2)
    for (int i = 0; i < grid_size_x_; ++i)
    {
        for (int j = 1; j < grid_size_z_; ++j)
        {
            ef_x_ptr_[ef_x_idx_2_ + i + j * grid_size_x_ * (grid_size_y_ + 1)] =
                recv_buf_y_plane_0_x_ptr_[i + grid_size_x_ * j];
            ef_x_ptr_[ef_x_idx_2_ + i + grid_size_y_ * grid_size_x_ + j * grid_size_x_ * (grid_size_y_ + 1)] =
                recv_buf_y_plane_1_x_ptr_[i + grid_size_x_ * j];
        }
    }
#pragma acc parallel loop collapse(2)
    for (int i = 0; i < grid_size_x_; ++i)
    {
        for (int j = 1; j < grid_size_y_; ++j)
        {
            ef_x_ptr_[ef_x_idx_2_ + i + j * grid_size_x_] =
                recv_buf_z_plane_0_x_ptr_[i + grid_size_x_ * j];
            ef_x_ptr_[ef_x_idx_2_ + i + j * grid_size_x_ + grid_size_x_ * (grid_size_y_ + 1) * grid_size_z_] =
                recv_buf_z_plane_1_x_ptr_[i + grid_size_x_ * j];
        }
    }
#pragma acc parallel loop collapse(2)
    for (int i = 1; i < grid_size_x_; ++i)
    {
        for (int j = 0; j < grid_size_y_; ++j)
        {
            ef_y_ptr_[ef_y_idx_2_ + i + j * (grid_size_x_ + 1)] =
                recv_buf_z_plane_0_y_ptr_[i + (grid_size_x_ + 1) * j];
            ef_y_ptr_[ef_y_idx_2_ + i + j * (grid_size_x_ + 1) + (grid_size_x_ + 1) * grid_size_y_ * grid_size_z_] =
                recv_buf_z_plane_1_y_ptr_[i + (grid_size_x_ + 1) * j];
        }
    }
#pragma acc parallel loop
    for (int i = 0; i < grid_size_x_; ++i)
    {
        ef_x_ptr_[ef_x_idx_2_ + i] =
            recv_buf_x_line_0_ptr_[i];
        ef_x_ptr_[ef_x_idx_2_ + i + grid_size_x_ * grid_size_y_] =
            recv_buf_x_line_1_ptr_[i];
        ef_x_ptr_[ef_x_idx_2_ + i + grid_size_x_ * (grid_size_y_ + 1) * grid_size_z_] =
            recv_buf_x_line_2_ptr_[i];
        ef_x_ptr_[ef_x_idx_2_ + i + grid_size_x_ * (grid_size_y_ + 1) * grid_size_z_ + grid_size_x_ * grid_size_y_] =
            recv_buf_x_line_3_ptr_[i];
    }
#pragma acc parallel loop
    for (int i = 0; i < grid_size_y_; ++i)
    {
        ef_y_ptr_[ef_y_idx_2_ + i * (grid_size_x_ + 1)] =
            recv_buf_y_line_0_ptr_[i];
        ef_y_ptr_[ef_y_idx_2_ + i * (grid_size_x_ + 1) + (grid_size_x_ + 1) * grid_size_y_ * grid_size_z_] =
            recv_buf_y_line_1_ptr_[i];
        ef_y_ptr_[ef_y_idx_2_ + i * (grid_size_x_ + 1) + grid_size_x_] =
            recv_buf_y_line_2_ptr_[i];
        ef_y_ptr_[ef_y_idx_2_ + i * (grid_size_x_ + 1) + (grid_size_x_ + 1) * grid_size_y_ * grid_size_z_ + grid_size_x_] =
            recv_buf_y_line_3_ptr_[i];
    }
#pragma acc parallel loop
    for (int i = 0; i < grid_size_z_; ++i)
    {
        ef_z_ptr_[ef_z_idx_2_ + i * (grid_size_x_ + 1) * (grid_size_y_ + 1)] =
            recv_buf_z_line_0_ptr_[i];
        ef_z_ptr_[ef_z_idx_2_ + i * (grid_size_x_ + 1) * (grid_size_y_ + 1) + grid_size_x_] =
            recv_buf_z_line_1_ptr_[i];
        ef_z_ptr_[ef_z_idx_2_ + i * (grid_size_x_ + 1) * (grid_size_y_ + 1) + (grid_size_x_ + 1) * grid_size_y_] =
            recv_buf_z_line_2_ptr_[i];
        ef_z_ptr_[ef_z_idx_2_ + i * (grid_size_x_ + 1) * (grid_size_y_ + 1) + (grid_size_x_ + 1) * grid_size_y_ + grid_size_x_] =
            recv_buf_z_line_3_ptr_[i];
    }
}

void FemSimulation::applyBoundaryConditions()
{
    // 完全導体境界条件（U,Pの接線成分を0に）
    int idx_0, idx_1;
    if (coord_x_ == 0)
    {
#pragma acc parallel loop collapse(2) private(idx_0, idx_1)
        for (int j = 0; j < grid_size_y_; ++j)
        {
            for (int k = 0; k < grid_size_z_ + 1; ++k)
            {
                idx_0 = j * (grid_size_x_ + 1) + k * (grid_size_x_ + 1) * grid_size_y_;
                ef_y_ptr_[ef_y_idx_0_ + idx_0] = 0.0;
                ef_y_ptr_[ef_y_idx_1_ + idx_0] = 0.0;
            }
        }
#pragma acc parallel loop collapse(2) private(idx_0, idx_1)
        for (int k = 0; k < grid_size_z_; ++k)
        {
            for (int j = 0; j < grid_size_y_ + 1; ++j)
            {
                idx_0 = j * (grid_size_x_ + 1) + k * (grid_size_x_ + 1) * (grid_size_y_ + 1);
                ef_z_ptr_[ef_z_idx_0_ + idx_0] = 0.0;
                ef_z_ptr_[ef_z_idx_1_ + idx_0] = 0.0;
            }
        }
    }
    if (coord_x_ == dim_x_ - 1)
    {
#pragma acc parallel loop collapse(2) private(idx_0, idx_1)
        for (int j = 0; j < grid_size_y_; ++j)
        {
            for (int k = 0; k < grid_size_z_ + 1; ++k)
            {
                idx_1 =
                    j * (grid_size_x_ + 1) + k * (grid_size_x_ + 1) * grid_size_y_ + grid_size_x_;
                ef_y_ptr_[ef_y_idx_0_ + idx_1] = 0.0;
                ef_y_ptr_[ef_y_idx_1_ + idx_1] = 0.0;
            }
        }
#pragma acc parallel loop collapse(2) private(idx_0, idx_1)
        for (int k = 0; k < grid_size_z_; ++k)
        {
            for (int j = 0; j < grid_size_y_ + 1; ++j)
            {
                idx_1 =
                    j * (grid_size_x_ + 1) + k * (grid_size_x_ + 1) * (grid_size_y_ + 1) + grid_size_x_;
                ef_z_ptr_[ef_z_idx_0_ + idx_1] = 0.0;
                ef_z_ptr_[ef_z_idx_1_ + idx_1] = 0.0;
            }
        }
    }
    if (coord_y_ == 0)
    {
#pragma acc parallel loop collapse(2) private(idx_0, idx_1)
        for (int k = 0; k < grid_size_z_; ++k)
        {
            for (int i = 0; i < grid_size_x_ + 1; ++i)
            {
                idx_0 = i + k * (grid_size_x_ + 1) * (grid_size_y_ + 1);
                ef_z_ptr_[ef_z_idx_0_ + idx_0] = 0.0;
                ef_z_ptr_[ef_z_idx_1_ + idx_0] = 0.0;
            }
        }
#pragma acc parallel loop collapse(2) private(idx_0, idx_1)
        for (int i = 0; i < grid_size_x_; ++i)
        {
            for (int k = 0; k < grid_size_z_ + 1; ++k)
            {
                idx_0 = i + k * grid_size_x_ * (grid_size_y_ + 1);
                ef_x_ptr_[ef_x_idx_0_ + idx_0] = 0.0;
                ef_x_ptr_[ef_x_idx_1_ + idx_0] = 0.0;
            }
        }
    }
    if (coord_y_ == dim_y_ - 1)
    {
#pragma acc parallel loop collapse(2) private(idx_0, idx_1)
        for (int k = 0; k < grid_size_z_; ++k)
        {
            for (int i = 0; i < grid_size_x_ + 1; ++i)
            {
                idx_1 = i + k * (grid_size_x_ + 1) * (grid_size_y_ + 1) + (grid_size_x_ + 1) * grid_size_y_;
                ef_z_ptr_[ef_z_idx_0_ + idx_1] = 0.0;
                ef_z_ptr_[ef_z_idx_1_ + idx_1] = 0.0;
            }
        }
#pragma acc parallel loop collapse(2) private(idx_0, idx_1)
        for (int i = 0; i < grid_size_x_; ++i)
        {
            for (int k = 0; k < grid_size_z_ + 1; ++k)
            {
                idx_1 = i + k * grid_size_x_ * (grid_size_y_ + 1) + grid_size_x_ * grid_size_y_;
                ef_x_ptr_[ef_x_idx_0_ + idx_1] = 0.0;
                ef_x_ptr_[ef_x_idx_1_ + idx_1] = 0.0;
            }
        }
    }
    if (coord_z_ == 0)
    {
#pragma acc parallel loop collapse(2) private(idx_0, idx_1)
        for (int i = 0; i < grid_size_x_; ++i)
        {
            for (int j = 0; j < grid_size_y_ + 1; ++j)
            {
                idx_0 = i + j * grid_size_x_;
                ef_x_ptr_[ef_x_idx_0_ + idx_0] = 0.0;
                ef_x_ptr_[ef_x_idx_1_ + idx_0] = 0.0;
            }
        }
#pragma acc parallel loop collapse(2) private(idx_0, idx_1)
        for (int j = 0; j < grid_size_y_; ++j)
        {
            for (int i = 0; i < grid_size_x_ + 1; ++i)
            {
                idx_0 = i + j * (grid_size_x_ + 1);
                ef_y_ptr_[ef_y_idx_0_ + idx_0] = 0.0;
                ef_y_ptr_[ef_y_idx_1_ + idx_0] = 0.0;
            }
        }
    }
    if (coord_z_ == dim_z_ - 1)
    {
#pragma acc parallel loop collapse(2) private(idx_0, idx_1)
        for (int i = 0; i < grid_size_x_; ++i)
        {
            for (int j = 0; j < grid_size_y_ + 1; ++j)
            {
                idx_1 = i + j * grid_size_x_ + grid_size_x_ * (grid_size_y_ + 1) * grid_size_z_;
                ef_x_ptr_[ef_x_idx_0_ + idx_1] = 0.0;
                ef_x_ptr_[ef_x_idx_1_ + idx_1] = 0.0;
            }
        }
#pragma acc parallel loop collapse(2) private(idx_0, idx_1)
        for (int j = 0; j < grid_size_y_; ++j)
        {
            for (int i = 0; i < grid_size_x_ + 1; ++i)
            {
                idx_1 = i + j * (grid_size_x_ + 1) + (grid_size_x_ + 1) * grid_size_y_ * grid_size_z_;
                ef_y_ptr_[ef_y_idx_0_ + idx_1] = 0.0;
                ef_y_ptr_[ef_y_idx_1_ + idx_1] = 0.0;
            }
        }
    }
}

void FemSimulation::run(int num_steps)
{
    saved_electric_field_.resize(observation_points_.size());
    for (auto &point : saved_electric_field_)
    {
        point.resize(static_cast<int>(std::round((num_steps - 1) / time_frequency_)) + 1);
    }
    src_pos_x_ptr_ = source_position_x_.data();
    src_pos_y_ptr_ = source_position_y_.data();
    src_pos_z_ptr_ = source_position_z_.data();
#pragma acc data copyin(src_pos_x_ptr_[0 : source_position_x_.size()])
#pragma acc data copyin(src_pos_y_ptr_[0 : source_position_y_.size()])
#pragma acc data copyin(src_pos_z_ptr_[0 : source_position_z_.size()])

    for (int step = 0; step < num_steps; ++step)
    {
        if (step % 100 == 0)
        {
            std::cout << "rank: " << rank_ << " step: " << step << std::endl;
        }

        // 観測点での値を保存
        if (step % time_frequency_ == 0)
        {
            for (size_t i = 0; i < observation_points_.size(); ++i)
            {
                int conn_x_0 = observation_points_[i][0] - 1;
                int conn_x_1 = observation_points_[i][0];
                int conn_x_2 = observation_points_[i][0] + grid_size_x_ * (grid_size_y_ + 1) - 1;
                int conn_x_3 = observation_points_[i][0] + grid_size_x_ * (grid_size_y_ + 1);
                int conn_y_0 = observation_points_[i][1] - (grid_size_x_ + 1);
                int conn_y_1 = observation_points_[i][1];
                int conn_y_2 = observation_points_[i][1] + (grid_size_x_ + 1) * (grid_size_y_ - 1);
                int conn_y_3 = observation_points_[i][1] + (grid_size_x_ + 1) * grid_size_y_;
                int conn_z_0 = observation_points_[i][2];
#pragma acc update host(ef_x_ptr_[ef_x_idx_0_ + conn_x_0])
#pragma acc update host(ef_x_ptr_[ef_x_idx_0_ + conn_x_1])
#pragma acc update host(ef_x_ptr_[ef_x_idx_0_ + conn_x_2])
#pragma acc update host(ef_x_ptr_[ef_x_idx_0_ + conn_x_3])
#pragma acc update host(ef_y_ptr_[ef_y_idx_0_ + conn_y_0])
#pragma acc update host(ef_y_ptr_[ef_y_idx_0_ + conn_y_1])
#pragma acc update host(ef_y_ptr_[ef_y_idx_0_ + conn_y_2])
#pragma acc update host(ef_y_ptr_[ef_y_idx_0_ + conn_y_3])
#pragma acc update host(ef_z_ptr_[ef_z_idx_0_ + conn_z_0])
                saved_electric_field_[i][static_cast<int>(std::round(step / time_frequency_))][0] =
                    current_time_;
                saved_electric_field_[i][static_cast<int>(std::round(step / time_frequency_))][1] =
                    (ef_x_ptr_[ef_x_idx_0_ + conn_x_0] +
                     ef_x_ptr_[ef_x_idx_0_ + conn_x_1] +
                     ef_x_ptr_[ef_x_idx_0_ + conn_x_2] +
                     ef_x_ptr_[ef_x_idx_0_ + conn_x_3]) /
                    4.0;
                saved_electric_field_[i][static_cast<int>(std::round(step / time_frequency_))][2] =
                    (ef_y_ptr_[ef_y_idx_0_ + conn_y_0] +
                     ef_y_ptr_[ef_y_idx_0_ + conn_y_1] +
                     ef_y_ptr_[ef_y_idx_0_ + conn_y_2] +
                     ef_y_ptr_[ef_y_idx_0_ + conn_y_3]) /
                    4.0;
                saved_electric_field_[i][static_cast<int>(std::round(step / time_frequency_))][3] =
                    ef_z_ptr_[ef_z_idx_0_ + conn_z_0];
            }
        }

        // シンプレクティックで更新(2次)
        updateTimeStep(time_step_, 0.0);

        // 時間を進める
        current_time_ += time_step_;
    }
#pragma acc exit data delete (src_pos_x_ptr_[0 : source_position_x_.size()])
#pragma acc exit data delete (src_pos_y_ptr_[0 : source_position_y_.size()])
#pragma acc exit data delete (src_pos_z_ptr_[0 : source_position_z_.size()])
    std::cout << "rank: " << rank_ << " Total communication time: " << total_communication_time_ << " seconds" << std::endl;
}

void FemSimulation::saveResults(int num_steps, const std::string &filename)
{
    for (size_t i = 0; i < saved_electric_field_.size(); ++i)
    {
        std::ofstream file(filename);
        if (!file.is_open())
        {
            std::cerr << "rank: " << rank_ << " Error: Could not open file " << filename << std::endl;
            return;
        }

        // 結果をCSV形式で保存
        file << "time,Ex,Ey,Ez" << std::endl;
        for (size_t j = 0; j < saved_electric_field_[i].size(); ++j)
        {
            file << saved_electric_field_[i][j][0] << "," << saved_electric_field_[i][j][1] << ","
                 << saved_electric_field_[i][j][2] << "," << saved_electric_field_[i][j][3]
                 << std::endl;
        }
        file.close();
        std::cout << "rank: " << rank_ << " Results saved to " << filename << std::endl;
    }
}

FemSimulation::~FemSimulation()
{
#pragma acc exit data delete (ef_x_ptr_[0 : ENx])
#pragma acc exit data delete (ef_y_ptr_[0 : ENy])
#pragma acc exit data delete (ef_z_ptr_[0 : ENz])
#pragma acc exit data delete (conn_x_ptr_[0 : CNx])
#pragma acc exit data delete (conn_y_ptr_[0 : CNy])
#pragma acc exit data delete (conn_z_ptr_[0 : CNz])
#pragma acc exit data delete (element_stiffness_matrix_ptr_[0 : mat_size_ * mat_size_])
#pragma acc exit data delete (outer_elems_ptr_[0 : OE])
#pragma acc exit data delete (send_buf_x_plane_0_y_ptr_[0 : BNx0y])
#pragma acc exit data delete (recv_buf_x_plane_0_y_ptr_[0 : BNx0y])
#pragma acc exit data delete (send_buf_x_plane_1_y_ptr_[0 : BNx1y])
#pragma acc exit data delete (recv_buf_x_plane_1_y_ptr_[0 : BNx1y])
#pragma acc exit data delete (send_buf_x_plane_0_z_ptr_[0 : BNx0z])
#pragma acc exit data delete (recv_buf_x_plane_0_z_ptr_[0 : BNx0z])
#pragma acc exit data delete (send_buf_x_plane_1_z_ptr_[0 : BNx1z])
#pragma acc exit data delete (recv_buf_x_plane_1_z_ptr_[0 : BNx1z])
#pragma acc exit data delete (send_buf_y_plane_0_x_ptr_[0 : BNy0x])
#pragma acc exit data delete (recv_buf_y_plane_0_x_ptr_[0 : BNy0x])
#pragma acc exit data delete (send_buf_y_plane_1_x_ptr_[0 : BNy1x])
#pragma acc exit data delete (recv_buf_y_plane_1_x_ptr_[0 : BNy1x])
#pragma acc exit data delete (send_buf_y_plane_0_z_ptr_[0 : BNy0z])
#pragma acc exit data delete (recv_buf_y_plane_0_z_ptr_[0 : BNy0z])
#pragma acc exit data delete (send_buf_y_plane_1_z_ptr_[0 : BNy1z])
#pragma acc exit data delete (recv_buf_y_plane_1_z_ptr_[0 : BNy1z])
#pragma acc exit data delete (send_buf_z_plane_0_x_ptr_[0 : BNz0x])
#pragma acc exit data delete (recv_buf_z_plane_0_x_ptr_[0 : BNz0x])
#pragma acc exit data delete (send_buf_z_plane_1_x_ptr_[0 : BNz1x])
#pragma acc exit data delete (recv_buf_z_plane_1_x_ptr_[0 : BNz1x])
#pragma acc exit data delete (send_buf_z_plane_0_y_ptr_[0 : BNz0y])
#pragma acc exit data delete (recv_buf_z_plane_0_y_ptr_[0 : BNz0y])
#pragma acc exit data delete (send_buf_z_plane_1_y_ptr_[0 : BNz1y])
#pragma acc exit data delete (recv_buf_z_plane_1_y_ptr_[0 : BNz1y])
#pragma acc exit data delete (send_buf_x_line_0_ptr_[0 : BLx0])
#pragma acc exit data delete (recv_buf_x_line_0_ptr_[0 : BLx0])
#pragma acc exit data delete (send_buf_x_line_1_ptr_[0 : BLx1])
#pragma acc exit data delete (recv_buf_x_line_1_ptr_[0 : BLx1])
#pragma acc exit data delete (send_buf_x_line_2_ptr_[0 : BLx2])
#pragma acc exit data delete (recv_buf_x_line_2_ptr_[0 : BLx2])
#pragma acc exit data delete (send_buf_x_line_3_ptr_[0 : BLx3])
#pragma acc exit data delete (recv_buf_x_line_3_ptr_[0 : BLx3])
#pragma acc exit data delete (send_buf_y_line_0_ptr_[0 : BLy0])
#pragma acc exit data delete (recv_buf_y_line_0_ptr_[0 : BLy0])
#pragma acc exit data delete (send_buf_y_line_1_ptr_[0 : BLy1])
#pragma acc exit data delete (recv_buf_y_line_1_ptr_[0 : BLy1])
#pragma acc exit data delete (send_buf_y_line_2_ptr_[0 : BLy2])
#pragma acc exit data delete (recv_buf_y_line_2_ptr_[0 : BLy2])
#pragma acc exit data delete (send_buf_y_line_3_ptr_[0 : BLy3])
#pragma acc exit data delete (recv_buf_y_line_3_ptr_[0 : BLy3])
#pragma acc exit data delete (send_buf_z_line_0_ptr_[0 : BLz0])
#pragma acc exit data delete (recv_buf_z_line_0_ptr_[0 : BLz0])
#pragma acc exit data delete (send_buf_z_line_1_ptr_[0 : BLz1])
#pragma acc exit data delete (recv_buf_z_line_1_ptr_[0 : BLz1])
#pragma acc exit data delete (send_buf_z_line_2_ptr_[0 : BLz2])
#pragma acc exit data delete (recv_buf_z_line_2_ptr_[0 : BLz2])
#pragma acc exit data delete (send_buf_z_line_3_ptr_[0 : BLz3])
#pragma acc exit data delete (recv_buf_z_line_3_ptr_[0 : BLz3])
}

bool check_params(const std::array<double, 3> &domain_sizes, const std::array<int, 3> &grid_num, double duration,
                  double time_step, double domain_size, double c)
{
    if (domain_sizes[0] <= 0 || domain_sizes[1] <= 0 || domain_sizes[2] <= 0)
    {
        return false;
    }
    if (grid_num[0] <= 0 || grid_num[1] <= 0 || grid_num[2] <= 0)
    {
        return false;
    }
    if (duration <= 0)
    {
        return false;
    }
    if (time_step <= 0)
    {
        return false;
    }
    if (std::round(domain_sizes[0] / grid_num[0]) != std::round(domain_sizes[1] / grid_num[1]) ||
        std::round(domain_sizes[0] / grid_num[0]) != std::round(domain_sizes[2] / grid_num[2]))
    {
        std::cerr << "Invalid grid size" << std::endl;
        return false;
    }
    if (c * time_step >= domain_size)
    {
        std::cerr << "Invalid time step" << std::endl;
        std::cerr << "c * time_step: " << c * time_step << std::endl;
        std::cerr << "domain_size: " << domain_size << std::endl;
        return false;
    }
    return true;
};

std::string compress(double value)
{
    std::ostringstream oss;
    oss << std::scientific << std::setprecision(6) << value;
    std::string str = oss.str();

    // 'e' または 'E' の位置を探す
    size_t e_pos = str.find_first_of("eE");
    if (e_pos == std::string::npos)
    {
        return str; // e表記でない場合はそのまま
    }

    // 仮数部と指数部に分割
    std::string mantissa = str.substr(0, e_pos);
    std::string exponent = str.substr(e_pos); // "e+00" だけは除く
    if (exponent == "e+00" || exponent == "E+00")
    {
        exponent = "";
    }

    // 仮数部の末尾のゼロを削除
    mantissa.erase(mantissa.find_last_not_of('0') + 1, std::string::npos);
    if (mantissa.back() == '.')
    {
        mantissa.pop_back();
    }

    return mantissa + exponent;
}

double parseValue(const std::string &str)
{
    size_t start = str.find_first_not_of(" \t\r\n\"");
    size_t end = str.find_last_not_of(" \t\r\n\"");
    if (start == std::string::npos)
        return 0.0;
    std::string trimmed = str.substr(start, end - start + 1);

    size_t slashPos = trimmed.find('/');
    if (slashPos != std::string::npos)
    {
        double num = std::stod(trimmed.substr(0, slashPos));
        double den = std::stod(trimmed.substr(slashPos + 1));
        return num / den;
    }
    return std::stod(trimmed);
}

std::vector<std::vector<double>> loadMatrixFromCSV(const std::string &filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    std::vector<std::vector<double>> matrix;
    std::string line;

    while (std::getline(file, line))
    {
        std::vector<double> row;
        std::stringstream ss(line);
        std::string cell;

        while (std::getline(ss, cell, ','))
        {
            row.push_back(parseValue(cell));
        }

        if (!row.empty())
        {
            matrix.push_back(row);
        }
    }

    return matrix;
}