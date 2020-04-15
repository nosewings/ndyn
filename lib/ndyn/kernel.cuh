#pragma once

#include <ndyn/util.cuh>
#include <ndyn/vec.cuh>

namespace ndyn {

namespace kernel {

///
/// Iterate an m-dimensional function over an arbitrary sampling of an
/// n-dimensional parameter space.
///
/// @param[out] out Output array with shape `(m2, NX, dims...)`.
///
template<typename P, size_t NP, typename X, size_t NX, typename F>
__global__
void iterate_param(
        F f,
        P const * in,
        size_t dim,
        vec<X, NX> x,
        size_t m1,
        size_t m2,
        X * out)
{
    auto tidx = size_t(blockIdx.x * blockDim.x + threadIdx.x);
    if (tidx >= dim) {
        return;
    }
    vec<P, NP> p;
    vec<size_t, 2> rec_dim_in { NP, dim };
    for (size_t n = 0; n < NP; n++) {
        vec<size_t, 2> rec_idx_in { n, tidx };
        auto lin_idx = ndyn::util::index::hyperrectangular_to_linear(
                rec_dim_in,
                rec_idx_in);
        p[n] = in[lin_idx];
    }

    for (size_t i = 0; i < m1; i++) {
        x = f(p, x);
    }
    vec<size_t, 3> rec_dim_out { m2, NX, dim };
    for (size_t i = 0; i < m2; i++) {
        for (size_t n = 0; n < NX; n++) {
            vec<size_t, 3> rec_idx_out { i, n, tidx };
            auto out_idx = ndyn::util::index::hyperrectangular_to_linear(
                    rec_dim_out,
                    rec_idx_out);
            out[out_idx] = x[n];
        }
        x = f(p, x);
    }
}

///
/// Iterate an m-dimensional function over a uniform sampling of an
/// n-dimensional parameter space, looking for cycles in powers of two.
///
/// This is useful for generating bifurcation diagrams of higher-dimensional
/// functions.
///
/// @param[out] out Output array with shape `(m2, NX, dims...)`.
///
template<typename P, size_t NP, typename X, size_t NX, typename F>
__global__
void iterate_param_uniform_find_powers_of_two_cycles(
        F f,
        vec<P, NP> start,
        vec<P, NP> end,
        vec<size_t, NP> dim,
        vec<X, NX> x,
        X tol,
        size_t m1,
        size_t m2,
        uint8_t * out)
{
    auto prod = dim.product();
    auto lin_tidx = size_t(blockIdx.x * blockDim.x + threadIdx.x);
    if (lin_tidx >= prod) {
        return;
    }
    auto rec_tidx = ndyn::util::index::linear_to_hyperrectangular(dim, lin_tidx);
    vec<P, NP> p = ndyn::util::hyperlinspace(dim, rec_tidx);

    for (size_t i = 0; i < m1; i++) {
        x = f(p, x);
    }

    auto x0 = x;
//    for (size_t i = 0; i < m2; i++) {
//        auto nloops = 1 << i;
//        for (size_t j = 0; j < nloops; j++) {
//            x = f(p, x);
//        }
//        if (x.distance(x0) < tol) {
//            out
//            return;
//        }
//    }
}

///
/// Iterate an n-dimensional function over an arbitrary sampling of an
/// n-dimensional phase space.
///
/// @param[in]  in  Input array with shape `(N, dim)`.
/// @param[out] out Output array with shape `(m2, N, dim)`.
///
template<typename T, size_t N, typename F>
__global__
void iterate_phase(
        F f,
        T const * in,
        size_t dim,
        size_t m1,
        size_t m2,
        T * out)
{
    auto tidx = size_t(blockIdx.x * blockDim.x + threadIdx.x);
    if (tidx >= dim) {
        return;
    }
    vec<T, N> x;
    vec<size_t, 2> rec_dim_in { N, dim };
    for (size_t n = 0; n < N; n++) {
        vec<size_t, 2> rec_idx_in { n, tidx };
        auto lin_idx = ndyn::util::index::hyperrectangular_to_linear(
                rec_dim_in,
                rec_idx_in);
        x[n] = in[lin_idx];
    }

    for (size_t i = 0; i < m1; i++) {
        x = f(x);
    }
    vec<size_t, 3> rec_dim_out { m2, N, dim };
    for (size_t i = 0; i < m2; i++) {
        for (size_t n = 0; n < N; n++) {
            vec<size_t, 3> rec_idx_out { i, n, tidx };
            auto out_idx = ndyn::util::index::hyperrectangular_to_linear(
                    rec_dim_out,
                    rec_idx_out);
            out[out_idx] = x[n];
        }
        x = f(x);
    }
}

///
/// Iterate an n-dimensional function over a uniform sampling of an
/// n-dimensional phase space.
///
/// @param[out] out Output array with shape `(m2, N, dims...)`.
///
template<typename T, size_t N, typename F>
__global__
void iterate_phase_uniform(
        F f,
        ndyn::vec<T, N> start,
        ndyn::vec<T, N> end,
        ndyn::vec<size_t, N> dims,
        size_t m1,
        size_t m2,
        T * out)
{
    auto prod = dims.product();
    auto lin_tidx = size_t(blockIdx.x * blockDim.x + threadIdx.x);
    if (lin_tidx >= prod) {
        return;
    }
    auto rec_tidx = ndyn::util::index::linear_to_hyperrectangular(
            dims,
            lin_tidx);
    auto x = ndyn::util::hyperlinspace(start, end, dims, rec_tidx);

    for (size_t i = 0; i < m1; i++) {
        x = f(x);
    }
    vec<size_t, 3> rec_dim { m2, N, prod };
    for (size_t i = 0; i < m2; i++) {
        for (size_t n = 0; n < N; n++) {
            vec<size_t, 3> rec_idx { i, n, lin_tidx };
            auto out_idx = ndyn::util::index::hyperrectangular_to_linear(
                    rec_dim,
                    rec_idx);
            out[out_idx] = x[n];
        }
        x = f(x);
    }
}

} // namespace kernel

} // namespace ndyn
