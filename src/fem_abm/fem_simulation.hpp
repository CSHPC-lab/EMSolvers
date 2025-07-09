#pragma once

#include <array>
#include <functional>
#include <memory>
#include <string>
#include <vector>

class FemSimulation
{
public:
    // コンストラクタ
    FemSimulation(const std::array<int, 3> &grid_size, double domain_size, double time_step,
                  double permittivity, double permeability, int time_frequency);

    // シミュレーションの実行
    void run(int num_steps);

    // 外力の設定
    void setSource_x(const std::array<int, 3> &position);
    void setSource_y(const std::array<int, 3> &position);
    void setSource_z(const std::array<int, 3> &position);

    // ヘルツダイポールの位置設定
    void setSourceDipole_x(const std::array<int, 3> &position);
    void setSourceDipole_y(const std::array<int, 3> &position);
    void setSourceDipole_z(const std::array<int, 3> &position);

    // 観測点の設定
    void setObservationPoint(const std::array<std::vector<std::array<int, 3>>, 3> &position);

    // 結果の保存
    void saveResults(int num_steps);

private:
    // メッシュの初期化
    void initializeMesh();

    // 時間ステップの更新
    void updateTimeStep();

    // 境界条件の適用
    void applyBoundaryConditions();

    // メンバ変数
    int grid_size_x_;
    int grid_size_y_;
    int grid_size_z_;
    double domain_size_;
    double time_step_;
    double permittivity_;
    double permeability_;
    int time_frequency_;
    // 電場の配列
    std::vector<double> electric_field_x_;
    std::vector<double> electric_field_y_;
    std::vector<double> electric_field_z_;
    // 磁場の配列
    std::vector<double> magnetic_field_x_;
    std::vector<double> magnetic_field_y_;
    std::vector<double> magnetic_field_z_;

    // コネクティビティ
    std::vector<int> connectivity_x_;
    std::vector<int> connectivity_y_;
    std::vector<int> connectivity_z_;

    // 外力の情報
    std::vector<int> source_position_x_;
    std::vector<int> source_position_y_;
    std::vector<int> source_position_z_;

    // 外力のヘルツダイポールの位置
    std::vector<std::array<int, 3>> source_dipole_position_x_;
    std::vector<std::array<int, 3>> source_dipole_position_y_;
    std::vector<std::array<int, 3>> source_dipole_position_z_;

    // 観測点のリスト
    std::vector<std::array<std::vector<std::array<int, 3>>, 3>> observation_points_;

    // 時間
    double current_time_;

    // 時間ステップ
    int time_0_;
    int time_1_;
    int time_2_;

    // 時間ステップに対応するインデックス
    int ef_x_idx_0_;
    int ef_x_idx_1_;
    int ef_x_idx_2_;
    int ef_x_idx_3_;
    int ef_y_idx_0_;
    int ef_y_idx_1_;
    int ef_y_idx_2_;
    int ef_y_idx_3_;
    int ef_z_idx_0_;
    int ef_z_idx_1_;
    int ef_z_idx_2_;
    int ef_z_idx_3_;

    // 要素剛性行列
    std::array<std::array<double, 12>, 12> element_stiffness_matrix_;

    // 要素剛性行列の計算
    void calculateElementStiffnessMatrix();

    // 保存する時系列計算結果
    std::vector<std::vector<std::array<double, 3>>> saved_electric_field_;
};

bool check_params(const std::array<double, 3> &domain_sizes, const std::array<int, 3> &grid_num, double duration,
                  double time_step, double domain_size, double c);

double source_function(double t);