#include <cmath>
#include <iostream>
#include <omp.h>

#include "fem_simulation.hpp"

// リッカー波を入力としたときの解析解（permeability * d/dt j）
double rickerWaveletAnalytical(double t, double observe_r, double c, double permeability)
{
    const double f0 = 1.0; // 中心周波数 [Hz]
    const double t0 = 2.0; // 時間シフト
    double tau = t - t0 - observe_r / c;
    return -(1.0 - 2.0 * M_PI * M_PI * f0 * f0 * tau * tau) *
           std::exp(-M_PI * M_PI * f0 * f0 * tau *
                    tau) *
           permeability / 4.0 / M_PI / observe_r;
}

// // リッカー波を入力としたときの解析解（permeability * d/dt j(= ricker)）
// double rickerWaveletAnalytical(double t, double observe_r, double c, double permeability)
// {
//     const double f0 = 1.0; // 中心周波数 [Hz]
//     const double t0 = 2.0; // 時間シフト
//     double tau = t - t0 - observe_r / c;
//     return (3.0 - 2.0 * M_PI * M_PI * f0 * f0 * tau * tau) *
//            std::exp(-M_PI * M_PI * f0 * f0 * tau *
//                     tau) *
//            permeability * M_PI * f0 * f0 * tau / 2.0 / observe_r;
// }

int main()
{
    std::cout << "OMP_NUM_THREADS = " << omp_get_max_threads() << std::endl;

    // シミュレーションパラメータの設定
    std::array<double, 3> domain_sizes = {5.0, 5.0, 5.0};         // 計算領域のサイズ [m]
    std::array<int, 3> grid_num = {800, 800, 800};                // グリッド数
    double duration = 5.0;                                        // シミュレーション時間 [s]
    double time_step = 0.0025;                                 // 時間ステップ [s]
    double relative_permittivity = 1.0;                           // 比誘電率
    double relative_permeability = 1.0;                           // 比透磁率
    std::array<double, 3> source_position = {2.0, 2.0, 2.0};      // 入力点の位置
    std::array<double, 3> observation_position = {2.5, 2.5, 2.5}; // 観測点の位置
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
        static_cast<int>(std::round(source_position[2] / domain_size))}; // どの辺に入力するか
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
    }; // 観測点座標が始点から数えて何番目の節点か
    double observe_r = std::sqrt(std::pow(observation_position[0] - source_position[0], 2) +
                                 std::pow(observation_position[1] - source_position[1], 2) +
                                 std::pow(observation_position[2] - source_position[2], 2));
    double c = 1.0 / std::sqrt(permittivity * permeability);

    // パラメータのチェック
    if (!check_params(domain_sizes, grid_num, duration, time_step, domain_size, c))
    {
        std::cerr << "Invalid parameters" << std::endl;
        return 1;
    }

    double start_time = omp_get_wtime();
    // シミュレーションオブジェクトの作成
    FemSimulation simulation(grid_num, domain_size, time_step, permittivity, permeability,
                             time_frequency);
    double end_time = omp_get_wtime();
    std::cout << "Simulation object created in " << end_time - start_time << " seconds" << std::endl;

    // 外力の設定
    simulation.setSource_x(source_index_0);

    // 観測点の設定
    simulation.setObservationPoint(observation_index);

    // 計算の実行
    std::cout << "Starting simulation..." << std::endl;
    start_time = omp_get_wtime();
    simulation.run(num_steps);
    end_time = omp_get_wtime();
    std::cout << "Simulation completed in " << end_time - start_time << " seconds" << std::endl;

    // 結果の保存
    simulation.saveResults(observe_r, rickerWaveletAnalytical, num_steps, c);
    std::cout << "Simulation completed. Results saved to results.csv" << std::endl;

    return 0;
}