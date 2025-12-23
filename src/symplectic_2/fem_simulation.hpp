#pragma once

#include <array>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <mpi.h>

class FemSimulation
{
public:
    // コンストラクタ
    FemSimulation(int order, const std::array<int, 3> &grid_size, double domain_size, double time_step,
                  double permittivity, double permeability, int time_frequency,
                  int use_ofem, const std::array<int, 3> &dims, const std::array<int, 3> &coords, int rank,
                  MPI_Comm comm_x_plane_0, MPI_Comm comm_x_plane_1, MPI_Comm comm_y_plane_0, MPI_Comm comm_y_plane_1, MPI_Comm comm_z_plane_0, MPI_Comm comm_z_plane_1,
                  MPI_Comm comm_x_line_0, MPI_Comm comm_x_line_1, MPI_Comm comm_x_line_2, MPI_Comm comm_x_line_3,
                  MPI_Comm comm_y_line_0, MPI_Comm comm_y_line_1, MPI_Comm comm_y_line_2, MPI_Comm comm_y_line_3,
                  MPI_Comm comm_z_line_0, MPI_Comm comm_z_line_1, MPI_Comm comm_z_line_2, MPI_Comm comm_z_line_3,
                  int rank_x_plane_0, int rank_x_plane_1, int rank_y_plane_0, int rank_y_plane_1, int rank_z_plane_0, int rank_z_plane_1,
                  int rank_x_line_0, int rank_x_line_1, int rank_x_line_2, int rank_x_line_3,
                  int rank_y_line_0, int rank_y_line_1, int rank_y_line_2, int rank_y_line_3,
                  int rank_z_line_0, int rank_z_line_1, int rank_z_line_2, int rank_z_line_3);

    // シミュレーションの実行
    void run(int num_steps);

    // 外力の設定
    void setSource_x(const std::array<int, 3> &position);
    void setSource_y(const std::array<int, 3> &position);
    void setSource_z(const std::array<int, 3> &position);

    // 観測点の設定
    void setObservationPoint(const std::array<int, 3> &position);

    // 結果の保存
    void saveResults(int num_steps, const std::string &filename);

    ~FemSimulation();

private:
    // メッシュの初期化
    void initializeMesh();

    // 時間ステップの更新
    void updateTimeStep(const double deltat, const double offset);

    // 境界条件の適用
    void applyBoundaryConditions();

    // メンバ変数
    int order_;
    int grid_size_x_;
    int grid_size_y_;
    int grid_size_z_;
    double domain_size_;
    double time_step_;
    double permittivity_;
    double permeability_;
    int time_frequency_;
    int use_ofem_;
    int dim_x_;
    int dim_y_;
    int dim_z_;
    int coord_x_;
    int coord_y_;
    int coord_z_;
    int rank_;
    MPI_Comm comm_x_plane_0_, comm_x_plane_1_, comm_y_plane_0_, comm_y_plane_1_, comm_z_plane_0_, comm_z_plane_1_;
    MPI_Comm comm_x_line_0_, comm_x_line_1_, comm_x_line_2_, comm_x_line_3_;
    MPI_Comm comm_y_line_0_, comm_y_line_1_, comm_y_line_2_, comm_y_line_3_;
    MPI_Comm comm_z_line_0_, comm_z_line_1_, comm_z_line_2_, comm_z_line_3_;
    int rank_x_plane_0_, rank_x_plane_1_, rank_y_plane_0_, rank_y_plane_1_, rank_z_plane_0_, rank_z_plane_1_;
    int rank_x_line_0_, rank_x_line_1_, rank_x_line_2_, rank_x_line_3_;
    int rank_y_line_0_, rank_y_line_1_, rank_y_line_2_, rank_y_line_3_;
    int rank_z_line_0_, rank_z_line_1_, rank_z_line_2_, rank_z_line_3_;

    // 電場の配列
    std::vector<double> electric_field_x_;
    std::vector<double> electric_field_y_;
    std::vector<double> electric_field_z_;

    // コネクティビティ
    std::vector<int> connectivity_x_;
    std::vector<int> connectivity_y_;
    std::vector<int> connectivity_z_;

    // bufferの配列
    std::vector<double> send_buf_x_plane_0_y_;
    std::vector<double> recv_buf_x_plane_0_y_;
    std::vector<double> send_buf_x_plane_1_y_;
    std::vector<double> recv_buf_x_plane_1_y_;
    std::vector<double> send_buf_x_plane_0_z_;
    std::vector<double> recv_buf_x_plane_0_z_;
    std::vector<double> send_buf_x_plane_1_z_;
    std::vector<double> recv_buf_x_plane_1_z_;
    std::vector<double> send_buf_y_plane_0_x_;
    std::vector<double> recv_buf_y_plane_0_x_;
    std::vector<double> send_buf_y_plane_1_x_;
    std::vector<double> recv_buf_y_plane_1_x_;
    std::vector<double> send_buf_y_plane_0_z_;
    std::vector<double> recv_buf_y_plane_0_z_;
    std::vector<double> send_buf_y_plane_1_z_;
    std::vector<double> recv_buf_y_plane_1_z_;
    std::vector<double> send_buf_z_plane_0_x_;
    std::vector<double> recv_buf_z_plane_0_x_;
    std::vector<double> send_buf_z_plane_1_x_;
    std::vector<double> recv_buf_z_plane_1_x_;
    std::vector<double> send_buf_z_plane_0_y_;
    std::vector<double> recv_buf_z_plane_0_y_;
    std::vector<double> send_buf_z_plane_1_y_;
    std::vector<double> recv_buf_z_plane_1_y_;
    std::vector<double> send_buf_x_line_0_;
    std::vector<double> recv_buf_x_line_0_;
    std::vector<double> send_buf_x_line_1_;
    std::vector<double> recv_buf_x_line_1_;
    std::vector<double> send_buf_x_line_2_;
    std::vector<double> recv_buf_x_line_2_;
    std::vector<double> send_buf_x_line_3_;
    std::vector<double> recv_buf_x_line_3_;
    std::vector<double> send_buf_y_line_0_;
    std::vector<double> recv_buf_y_line_0_;
    std::vector<double> send_buf_y_line_1_;
    std::vector<double> recv_buf_y_line_1_;
    std::vector<double> send_buf_y_line_2_;
    std::vector<double> recv_buf_y_line_2_;
    std::vector<double> send_buf_y_line_3_;
    std::vector<double> recv_buf_y_line_3_;
    std::vector<double> send_buf_z_line_0_;
    std::vector<double> recv_buf_z_line_0_;
    std::vector<double> send_buf_z_line_1_;
    std::vector<double> recv_buf_z_line_1_;
    std::vector<double> send_buf_z_line_2_;
    std::vector<double> recv_buf_z_line_2_;
    std::vector<double> send_buf_z_line_3_;
    std::vector<double> recv_buf_z_line_3_;
    std::vector<int> outer_elems_;

    // 要素数
    int ENx;
    int ENy;
    int ENz;
    int CNx;
    int CNy;
    int CNz;
    int BNx0y;
    int BNx1y;
    int BNx0z;
    int BNx1z;
    int BNy0x;
    int BNy1x;
    int BNy0z;
    int BNy1z;
    int BNz0x;
    int BNz1x;
    int BNz0y;
    int BNz1y;
    int BLx0;
    int BLx1;
    int BLx2;
    int BLx3;
    int BLy0;
    int BLy1;
    int BLy2;
    int BLy3;
    int BLz0;
    int BLz1;
    int BLz2;
    int BLz3;
    int OE;

    // 外力の情報
    std::vector<int> source_position_x_;
    std::vector<int> source_position_y_;
    std::vector<int> source_position_z_;

    // 観測点のリスト
    std::vector<std::array<int, 3>> observation_points_;

    // 時間
    double current_time_;

    // 時間ステップに対応するインデックス
    int ef_x_idx_0_ = 0 * grid_size_x_ * (grid_size_y_ + 1) * (grid_size_z_ + 1);
    int ef_x_idx_1_ = 1 * grid_size_x_ * (grid_size_y_ + 1) * (grid_size_z_ + 1);
    int ef_x_idx_2_ = 2 * grid_size_x_ * (grid_size_y_ + 1) * (grid_size_z_ + 1);
    int ef_y_idx_0_ = 0 * (grid_size_x_ + 1) * grid_size_y_ * (grid_size_z_ + 1);
    int ef_y_idx_1_ = 1 * (grid_size_x_ + 1) * grid_size_y_ * (grid_size_z_ + 1);
    int ef_y_idx_2_ = 2 * (grid_size_x_ + 1) * grid_size_y_ * (grid_size_z_ + 1);
    int ef_z_idx_0_ = 0 * (grid_size_x_ + 1) * (grid_size_y_ + 1) * grid_size_z_;
    int ef_z_idx_1_ = 1 * (grid_size_x_ + 1) * (grid_size_y_ + 1) * grid_size_z_;
    int ef_z_idx_2_ = 2 * (grid_size_x_ + 1) * (grid_size_y_ + 1) * grid_size_z_;

    // 要素剛性行列
    std::vector<double> element_stiffness_matrix_;
    int mat_size_; // = order_ * (order_ + 1) * (order_ + 1) * 3

    // 要素剛性行列の計算
    void calculateElementStiffnessMatrix();

    // MPI通信
    void startExchangeElectricField();
    void finishExchangeElectricField();

    // 保存する時系列計算結果
    std::vector<std::vector<std::array<double, 4>>> saved_electric_field_;

    // 時間計測
    double total_communication_time_ = 0.0;
    double start_communication_time_;
    double end_communication_time_;

    // MPIリクエスト
    MPI_Request reqs_[24] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL,
                             MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL,
                             MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL,
                             MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL,
                             MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL,
                             MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL};
};

bool check_params(const std::array<double, 3> &domain_sizes, const std::array<int, 3> &grid_num, double duration,
                  double time_step, double domain_size, double c);

double source_function(double t);

std::string compress(double value);

double parseValue(const std::string &str);

std::vector<std::vector<double>> loadMatrixFromCSV(const std::string &filename);
