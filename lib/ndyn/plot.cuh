#pragma once

#include <memory>
#include <vector>

#include <cuda/api_wrappers.hpp>
#include <matplotlibcpp.h>

#include <ndyn/kernel.cuh>
#include <ndyn/util.cuh>
#include <ndyn/vec.cuh>

namespace ndyn {

namespace plot {

///
/// @brief Plot the x- and y-components of a one-parameter function in two
/// dimensions.
///
template<typename P, typename X, typename F, bool AssumedCurrent =
        cuda::detail::do_not_assume_device_is_current>
void bifurcation_1d_2d(
        F f,
        P const & start,
        P const & end,
        size_t dim,
        vec<X, 2> x0,
        size_t m1,
        size_t m2,
        cuda::device_t<AssumedCurrent> & device)
{
    auto npoints = dim;
    auto nelems_in = npoints;
    auto nbytes_in = nelems_in * sizeof(P);
    auto nelems_out = m2 * 2 * npoints;
    auto nbytes_out = nelems_out * sizeof(X);

    P hmem_in[npoints];
    for (size_t n = 0; n < npoints; n++) {
        hmem_in[n] = ndyn::util::linspace(start, end, npoints, n);
    }
    auto dmem_in = cuda::memory::device::make_unique<P[], AssumedCurrent>(
            device,
            nelems_in);
    cuda::memory::copy(dmem_in.get(), hmem_in, nbytes_in);

    auto dmem_out = cuda::memory::device::make_unique<X[], AssumedCurrent>(
            device,
            nelems_out);

    constexpr size_t thread = 32;
    auto block = ndyn::util::ceil_div(nelems_in, thread);
    cuda::launch_configuration_t config { block, thread };
    device.launch(
            ndyn::kernel::iterate_param<P, 1, X, 2, F>,
            config,
            f,
            dmem_in.get(),
            dim,
            x0,
            m1,
            m2,
            dmem_out.get());
    X hmem_out[m2][2][npoints];
    cuda::memory::copy(hmem_out, dmem_out.get(), nbytes_out);

    std::vector<P> ps;
    std::vector<X> xs;
    std::vector<X> ys;
    ps.reserve(m2 * npoints);
    xs.reserve(m2 * npoints);
    ys.reserve(m2 * npoints);
    for (size_t i = 0; i < npoints; i++) {
        auto p = ndyn::util::linspace(start, end, npoints, i);
        for (size_t j = 0; j < m2; j++) {
            auto x = hmem_out[j][0][i];
            auto y = hmem_out[j][1][i];
            ps.push_back(p);
            xs.push_back(x);
            ys.push_back(y);
        }
    }
    matplotlibcpp::figure();
    matplotlibcpp::subplot(1, 2, 1);
    matplotlibcpp::scatter(ps, xs);
    matplotlibcpp::subplot(1, 2, 2);
    matplotlibcpp::scatter(ps, ys);
}

///
/// @brief Plot the bifurcation diagram of a two-parameter function in two
/// dimensions.
///
template<typename P, typename X, typename F, typename Colors,
        bool AssumedCurrent = cuda::detail::do_not_assume_device_is_current>
void bifurcation_2d_2d(
        F f,
        vec<P, 2> const & start,
        vec<P, 2> const & end,
        vec<size_t, 2> const & dim,
        vec<X, 2> x0,
        size_t m1,
        Colors colors,
        cuda::device_t<AssumedCurrent> & device)
{
    auto nelems = dim.product();
    auto nbytes = nelems * sizeof(uint8_t);
    auto dmem = cuda::memory::device::make_unique<uint8_t[], AssumedCurrent>(
            device,
            nelems);
    constexpr size_t thread = 32;
    auto block = ndyn::util::ceil_div(nelems, thread);
    cuda::launch_configuration_t config { block, thread };
    device.launch(
            ndyn::kernel::iterate_param_uniform_find_powers_of_two_cycles,
            config,
            f,
            start,
            end,
            dim,
            x0,
            m1,
            colors.size(),
            dmem.get());
    uint8_t hmem[dim[0]][dim[1]];
    cuda::memory::copy(hmem, dmem.get(), nbytes);
}

template<typename T, typename Container, typename F, bool AssumedCurrent =
        cuda::detail::do_not_assume_device_is_current>
void trajectories_2d(
        F f,
        Container points,
        size_t m2,
        cuda::device_t<AssumedCurrent> & device)
{
    auto npoints = points.size();
    auto nelems_in = 2 * npoints;
    auto nbytes_in = nelems_in * sizeof(T);
    auto nelems_out = m2 * 2 * npoints;
    auto nbytes_out = nelems_out * sizeof(T);

    T hmem_in[2][npoints];
    for (size_t n = 0; n < npoints; n++) {
        auto point = points[n];
        hmem_in[0][n] = point[0];
        hmem_in[1][n] = point[1];
    }
    auto dmem_in = cuda::memory::device::make_unique<T[], AssumedCurrent>(
            device,
            nelems_in);
    cuda::memory::copy(dmem_in.get(), hmem_in, nbytes_in);

    auto dmem_out = cuda::memory::device::make_unique<T[], AssumedCurrent>(
            device,
            nelems_out);

    constexpr size_t thread = 32;
    auto block = ndyn::util::ceil_div(nelems_in, thread);
    cuda::launch_configuration_t config { block, thread };
    device.launch(
            ndyn::kernel::iterate_phase<T, 2, F>,
            config,
            f,
            dmem_in.get(),
            npoints,
            0,
            m2,
            dmem_out.get());
    T hmem_out[m2][2][npoints];
    cuda::memory::copy(hmem_out, dmem_out.get(), nbytes_out);

    std::vector<T> x, y;
    x.reserve(m2);
    y.reserve(m2);
    for (size_t i = 0; i < npoints; i++) {
        for (size_t j = 0; j < m2; j++) {
            x.push_back(hmem_out[j][0][i]);
            y.push_back(hmem_out[j][1][i]);
        }
        matplotlibcpp::plot(x, y);
        x.clear();
        y.clear();
    }
}

template<typename T, typename F, bool AssumedCurrent =
        cuda::detail::do_not_assume_device_is_current>
void vector_field_2d(
        F f,
        ndyn::vec<T, 2> const & start,
        ndyn::vec<T, 2> const & end,
        ndyn::vec<size_t, 2> const & dims,
        cuda::device_t<AssumedCurrent> & device)
{
    auto npoints = dims.product();
    auto nelems = 2 * npoints;
    auto nbytes = nelems * sizeof(T);

    auto dmem = cuda::memory::device::make_unique<T[], AssumedCurrent>(
            device,
            nelems);

    constexpr size_t thread = 32;
    auto block = ndyn::util::ceil_div(nelems, thread);
    cuda::launch_configuration_t config { block, thread };
    device.launch(
            ndyn::kernel::iterate_phase_uniform<T, 2, F>,
            config,
            f,
            start,
            end,
            dims,
            1,
            1,
            dmem.get());
    T hmem[2][dims[0]][dims[1]];
    cuda::memory::copy(hmem, dmem.get(), nbytes);

    std::vector<T> x, y, u, v;
    x.reserve(npoints);
    y.reserve(npoints);
    u.reserve(npoints);
    v.reserve(npoints);
    for (size_t i = 0; i < dims[0]; i++) {
        for (size_t j = 0; j < dims[1]; j++) {
            auto x0y0 = ndyn::util::hyperlinspace(start, end, dims, { i, j });
            auto x0 = x0y0[0];
            auto y0 = x0y0[1];
            auto x1 = hmem[0][i][j];
            auto y1 = hmem[1][i][j];
            auto dx = x1 - x0;
            auto dy = y1 - y0;
            auto norm = vec<T, 2>( { dx, dy }).norm();
            x.push_back(x0);
            y.push_back(y0);
            u.push_back(dx / norm);
            v.push_back(dy / norm);
        }
    }
    matplotlibcpp::quiver(x, y, u, v);
}

} // namespace plot

} // namespace ndyn
