#include <cmath>
#include <iostream>

#include "fem_simulation.hpp"

// リッカー波の外力関数
double rickerWavelet(double t) {
    const double f0 = 1.0;  // 中心周波数 [Hz]
    const double t0 = 2.0;  // 時間シフト
    double tau = t - t0;
    return (1.0 - 2.0 * M_PI * M_PI * f0 * f0 * tau * tau) *
           std::exp(-M_PI * M_PI * f0 * f0 * tau *
                    tau);  // 格子点に入力する場合は2辺にまたがるので÷2する
}

int main() {
    // シミュレーションパラメータの設定
    std::array<double, 3> domain_sizes = {5.0, 5.0, 5.0};          // 計算領域のサイズ [m]
    std::array<int, 3> grid_num = {200, 200, 200};                 // グリッド数
    double duration = 4.0;                                         // シミュレーション時間 [s]
    double time_step = 0.0125;                                     // 時間ステップ [s]
    double relative_permittivity = 1.0;                            // 比誘電率
    double relative_permeability = 1.0;                            // 比透磁率
    std::array<double, 3> source_position = {2.5, 2.5, 2.5};       // 入力点の位置
    std::array<double, 3> observation_position = {3.5, 2.5, 2.5};  // 観測点の位置
    int time_frequency = 1;

    double domain_size = domain_sizes[0] / grid_num[0];
    int num_steps = std::round(duration / time_step);
    double permittivity =
        // 8.854187817e-12 * relative_permittivity;  // 真空の誘電率 [F/m] * relative_permittivity
        1.0;
    double permeability =
        // 1.25663706212e-6 * relative_permeability;  // 真空の透磁率 [H/m] * relative_permeability
        1.0;
    std::array<int, 3> source_index_0 = {
        static_cast<int>(std::round(source_position[0] / domain_size)) - 1,
        static_cast<int>(std::round(source_position[1] / domain_size)),
        static_cast<int>(std::round(source_position[2] / domain_size))};  // どの辺に入力するか
    std::array<std::vector<std::array<int, 3>>, 3> observation_index = {
        std::vector<std::array<int, 3>>{
            {{static_cast<int>(std::round(observation_position[0] / domain_size)) - 1,
              static_cast<int>(std::round(observation_position[1] / domain_size)),
              static_cast<int>(std::round(observation_position[2] / domain_size))}}},
        std::vector<std::array<int, 3>>{
            {static_cast<int>(std::round(observation_position[0] / domain_size)) - 1,
             static_cast<int>(std::round(observation_position[1] / domain_size)) - 1,
             static_cast<int>(std::round(observation_position[2] / domain_size))},
            {static_cast<int>(std::round(observation_position[0] / domain_size)),
             static_cast<int>(std::round(observation_position[1] / domain_size)) - 1,
             static_cast<int>(std::round(observation_position[2] / domain_size))},
            {static_cast<int>(std::round(observation_position[0] / domain_size)) - 1,
             static_cast<int>(std::round(observation_position[1] / domain_size)),
             static_cast<int>(std::round(observation_position[2] / domain_size))},
            {static_cast<int>(std::round(observation_position[0] / domain_size)),
             static_cast<int>(std::round(observation_position[1] / domain_size)),
             static_cast<int>(std::round(observation_position[2] / domain_size))},
        },
        std::vector<std::array<int, 3>>{
            {static_cast<int>(std::round(observation_position[0] / domain_size)) - 1,
             static_cast<int>(std::round(observation_position[1] / domain_size)),
             static_cast<int>(std::round(observation_position[2] / domain_size)) - 1},
            {static_cast<int>(std::round(observation_position[0] / domain_size)),
             static_cast<int>(std::round(observation_position[1] / domain_size)),
             static_cast<int>(std::round(observation_position[2] / domain_size)) - 1},
            {static_cast<int>(std::round(observation_position[0] / domain_size)) - 1,
             static_cast<int>(std::round(observation_position[1] / domain_size)),
             static_cast<int>(std::round(observation_position[2] / domain_size))},
            {static_cast<int>(std::round(observation_position[0] / domain_size)),
             static_cast<int>(std::round(observation_position[1] / domain_size)),
             static_cast<int>(std::round(observation_position[2] / domain_size))},
        },
    };  // 観測点座標が始点から数えて何番目の節点か
    double observe_r = std::sqrt(std::pow(observation_position[0] - source_position[0], 2) +
                                 std::pow(observation_position[1] - source_position[1], 2) +
                                 std::pow(observation_position[2] - source_position[2], 2));
    double c = 1.0 / std::sqrt(permittivity * permeability);

    // パラメータのチェック
    if (!check_params(domain_sizes, grid_num, duration, time_step, domain_size, c)) {
        std::cerr << "Invalid parameters" << std::endl;
        return 1;
    }

    // シミュレーションオブジェクトの作成
    FemSimulation simulation(domain_size, grid_num, time_step, permittivity, permeability,
                             time_frequency);

    // 外力の設定
    simulation.setSource_x(source_index_0, rickerWavelet);

    // 観測点の設定
    simulation.setObservationPoint(observation_index);

    // 計算の実行
    std::cout << "Starting simulation..." << std::endl;
    simulation.run(num_steps);

    // 結果の保存
    simulation.saveResults(observe_r, rickerWavelet, num_steps, c);
    std::cout << "Simulation completed. Results saved to results.csv" << std::endl;

    return 0;
}