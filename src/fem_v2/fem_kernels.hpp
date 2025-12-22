#ifndef FEM_KERNELS_HPP
#define FEM_KERNELS_HPP

#include <openacc.h>

// Order=1: 1*(1+1)*(1+1) = 4,   4*3 = 12
// Order=2: 2*(2+1)*(2+1) = 18,  18*3 = 54
// Order=3: 3*(3+1)*(3+1) = 48,  48*3 = 144

//=============================================================================
// 外側要素
//=============================================================================
template <int Order>
void computeOuterElements(
    double *ef_x_ptr, double *ef_y_ptr, double *ef_z_ptr,
    const int *conn_x_ptr, const int *conn_y_ptr, const int *conn_z_ptr,
    const int *outer_elems_ptr, const double *stiffness_ptr,
    int ef_x_idx_0, int ef_x_idx_2,
    int ef_y_idx_0, int ef_y_idx_2,
    int ef_z_idx_0, int ef_z_idx_2,
    int num_outer_elems, int mat_size)
{
    constexpr int DOF = Order * (Order + 1) * (Order + 1);
    constexpr int TOTAL = 3 * DOF;

    int idx;
    double temp;
    int l, m, n;

#pragma acc parallel loop private(idx, temp, l, m, n)
    for (int i = 0; i < num_outer_elems; ++i)
    {
        double ef_local[TOTAL];
        idx = outer_elems_ptr[i] * DOF;

#pragma acc loop seq
        for (l = 0; l < DOF; ++l)
        {
            ef_local[l] = ef_x_ptr[ef_x_idx_0 + conn_x_ptr[idx + l]];
            ef_local[l + DOF] = ef_y_ptr[ef_y_idx_0 + conn_y_ptr[idx + l]];
            ef_local[l + 2 * DOF] = ef_z_ptr[ef_z_idx_0 + conn_z_ptr[idx + l]];
        }

#pragma acc loop seq
        for (l = 0; l < DOF; ++l)
        {
            temp = 0;
#pragma acc loop seq
            for (m = 0; m < TOTAL; ++m)
            {
                temp += stiffness_ptr[l * mat_size + m] * ef_local[m];
            }
            n = ef_x_idx_2 + conn_x_ptr[idx + l];
#pragma acc atomic update
            ef_x_ptr[n] -= temp;
        }

#pragma acc loop seq
        for (l = DOF; l < 2 * DOF; ++l)
        {
            temp = 0;
#pragma acc loop seq
            for (m = 0; m < TOTAL; ++m)
            {
                temp += stiffness_ptr[l * mat_size + m] * ef_local[m];
            }
            n = ef_y_idx_2 + conn_y_ptr[idx + l - DOF];
#pragma acc atomic update
            ef_y_ptr[n] -= temp;
        }

#pragma acc loop seq
        for (l = 2 * DOF; l < TOTAL; ++l)
        {
            temp = 0;
#pragma acc loop seq
            for (m = 0; m < TOTAL; ++m)
            {
                temp += stiffness_ptr[l * mat_size + m] * ef_local[m];
            }
            n = ef_z_idx_2 + conn_z_ptr[idx + l - 2 * DOF];
#pragma acc atomic update
            ef_z_ptr[n] -= temp;
        }
    }
}

//=============================================================================
// 内側要素
//=============================================================================
template <int Order>
void computeInnerElements(
    double *ef_x_ptr, double *ef_y_ptr, double *ef_z_ptr,
    const int *conn_x_ptr, const int *conn_y_ptr, const int *conn_z_ptr,
    const double *stiffness_ptr,
    int ef_x_idx_0, int ef_x_idx_2,
    int ef_y_idx_0, int ef_y_idx_2,
    int ef_z_idx_0, int ef_z_idx_2,
    int grid_size_x, int grid_size_y, int grid_size_z,
    int mat_size)
{
    constexpr int DOF = Order * (Order + 1) * (Order + 1);
    constexpr int TOTAL = 3 * DOF;

    int idx;
    double temp;
    int l, m, n;

    int num_elem_x = grid_size_x / Order;
    int num_elem_y = grid_size_y / Order;
    int num_elem_z = grid_size_z / Order;

#pragma acc parallel loop private(idx, temp, l, m, n) collapse(3)
    for (int k = 1; k < num_elem_z - 1; ++k)
    {
        for (int j = 1; j < num_elem_y - 1; ++j)
        {
            for (int i = 1; i < num_elem_x - 1; ++i)
            {
                double ef_local[TOTAL];
                idx = (i + j * num_elem_x + k * num_elem_x * num_elem_y) * DOF;

#pragma acc loop seq
                for (l = 0; l < DOF; ++l)
                {
                    ef_local[l] = ef_x_ptr[ef_x_idx_0 + conn_x_ptr[idx + l]];
                    ef_local[l + DOF] = ef_y_ptr[ef_y_idx_0 + conn_y_ptr[idx + l]];
                    ef_local[l + 2 * DOF] = ef_z_ptr[ef_z_idx_0 + conn_z_ptr[idx + l]];
                }

#pragma acc loop seq
                for (l = 0; l < DOF; ++l)
                {
                    temp = 0;
#pragma acc loop seq
                    for (m = 0; m < TOTAL; ++m)
                    {
                        temp += stiffness_ptr[l * mat_size + m] * ef_local[m];
                    }
                    n = ef_x_idx_2 + conn_x_ptr[idx + l];
#pragma acc atomic update
                    ef_x_ptr[n] -= temp;
                }

#pragma acc loop seq
                for (l = DOF; l < 2 * DOF; ++l)
                {
                    temp = 0;
#pragma acc loop seq
                    for (m = 0; m < TOTAL; ++m)
                    {
                        temp += stiffness_ptr[l * mat_size + m] * ef_local[m];
                    }
                    n = ef_y_idx_2 + conn_y_ptr[idx + l - DOF];
#pragma acc atomic update
                    ef_y_ptr[n] -= temp;
                }

#pragma acc loop seq
                for (l = 2 * DOF; l < TOTAL; ++l)
                {
                    temp = 0;
#pragma acc loop seq
                    for (m = 0; m < TOTAL; ++m)
                    {
                        temp += stiffness_ptr[l * mat_size + m] * ef_local[m];
                    }
                    n = ef_z_idx_2 + conn_z_ptr[idx + l - 2 * DOF];
#pragma acc atomic update
                    ef_z_ptr[n] -= temp;
                }
            }
        }
    }
}

//=============================================================================
// ディスパッチャー関数
//=============================================================================
inline void dispatchOuterElements(
    int order,
    double *ef_x_ptr, double *ef_y_ptr, double *ef_z_ptr,
    const int *conn_x_ptr, const int *conn_y_ptr, const int *conn_z_ptr,
    const int *outer_elems_ptr, const double *stiffness_ptr,
    int ef_x_idx_0, int ef_x_idx_2,
    int ef_y_idx_0, int ef_y_idx_2,
    int ef_z_idx_0, int ef_z_idx_2,
    int num_outer_elems, int mat_size)
{
    switch (order)
    {
    case 1:
        computeOuterElements<1>(ef_x_ptr, ef_y_ptr, ef_z_ptr,
                                conn_x_ptr, conn_y_ptr, conn_z_ptr,
                                outer_elems_ptr, stiffness_ptr,
                                ef_x_idx_0, ef_x_idx_2, ef_y_idx_0, ef_y_idx_2, ef_z_idx_0, ef_z_idx_2,
                                num_outer_elems, mat_size);
        break;
    case 2:
        computeOuterElements<2>(ef_x_ptr, ef_y_ptr, ef_z_ptr,
                                conn_x_ptr, conn_y_ptr, conn_z_ptr,
                                outer_elems_ptr, stiffness_ptr,
                                ef_x_idx_0, ef_x_idx_2, ef_y_idx_0, ef_y_idx_2, ef_z_idx_0, ef_z_idx_2,
                                num_outer_elems, mat_size);
        break;
    case 3:
        computeOuterElements<3>(ef_x_ptr, ef_y_ptr, ef_z_ptr,
                                conn_x_ptr, conn_y_ptr, conn_z_ptr,
                                outer_elems_ptr, stiffness_ptr,
                                ef_x_idx_0, ef_x_idx_2, ef_y_idx_0, ef_y_idx_2, ef_z_idx_0, ef_z_idx_2,
                                num_outer_elems, mat_size);
        break;
    }
}

inline void dispatchInnerElements(
    int order,
    double *ef_x_ptr, double *ef_y_ptr, double *ef_z_ptr,
    const int *conn_x_ptr, const int *conn_y_ptr, const int *conn_z_ptr,
    const double *stiffness_ptr,
    int ef_x_idx_0, int ef_x_idx_2,
    int ef_y_idx_0, int ef_y_idx_2,
    int ef_z_idx_0, int ef_z_idx_2,
    int grid_size_x, int grid_size_y, int grid_size_z,
    int mat_size)
{
    switch (order)
    {
    case 1:
        computeInnerElements<1>(ef_x_ptr, ef_y_ptr, ef_z_ptr,
                                conn_x_ptr, conn_y_ptr, conn_z_ptr, stiffness_ptr,
                                ef_x_idx_0, ef_x_idx_2, ef_y_idx_0, ef_y_idx_2, ef_z_idx_0, ef_z_idx_2,
                                grid_size_x, grid_size_y, grid_size_z, mat_size);
        break;
    case 2:
        computeInnerElements<2>(ef_x_ptr, ef_y_ptr, ef_z_ptr,
                                conn_x_ptr, conn_y_ptr, conn_z_ptr, stiffness_ptr,
                                ef_x_idx_0, ef_x_idx_2, ef_y_idx_0, ef_y_idx_2, ef_z_idx_0, ef_z_idx_2,
                                grid_size_x, grid_size_y, grid_size_z, mat_size);
        break;
    case 3:
        computeInnerElements<3>(ef_x_ptr, ef_y_ptr, ef_z_ptr,
                                conn_x_ptr, conn_y_ptr, conn_z_ptr, stiffness_ptr,
                                ef_x_idx_0, ef_x_idx_2, ef_y_idx_0, ef_y_idx_2, ef_z_idx_0, ef_z_idx_2,
                                grid_size_x, grid_size_y, grid_size_z, mat_size);
        break;
    }
}

#endif