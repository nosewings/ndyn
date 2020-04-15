#pragma once

#include <ndyn/vec.cuh>

namespace ndyn {

namespace util {

///////////////////////////////////////////////////////////////////////////////
// Declarations
///////////////////////////////////////////////////////////////////////////////

///
/// `constexpr` ceiling function.
///
template<typename T>
constexpr
T constexpr_ceil(
        T const &);

template<typename R, typename I>
constexpr
R linspace(
        R const &,
        R const &,
        I const &,
        I const &);

///
/// n-dimensional analogue of linspace().
///
template<typename R, typename I, size_t N>
constexpr
__host__ __device__
ndyn::vec<R, N> hyperlinspace(
        ndyn::vec<R, N> const &,
        ndyn::vec<R, N> const &,
        ndyn::vec<I, N> const &,
        ndyn::vec<I, N> const &);

///
/// Integer division that rounds up (toward positive infinity).
///
template<typename T>
constexpr
T ceil_div(
        T const &,
        T const &);

///////////////////////////////////////////////////////////////////////////////
// Definitions
///////////////////////////////////////////////////////////////////////////////

template<typename T>
constexpr
T constexpr_ceil(
        T const & x)
{
    auto x_ = int64_t(x);
    return (x_ == x) ? (x) : (x + 1);
}

template<typename R, typename I>
constexpr
R linspace(
        R const & start,
        R const & end,
        I const & n,
        I const & i)
{
    return start + i * ((end - start) / (n - 1));
}

template<typename R, typename I, size_t N>
constexpr
__host__ __device__
ndyn::vec<R, N> hyperlinspace(
        ndyn::vec<R, N> const & start,
        ndyn::vec<R, N> const & end,
        ndyn::vec<I, N> const & dims,
        ndyn::vec<I, N> const & i)
{
    ndyn::vec<R, N> ret;
    for (size_t n = 0; n < N; n++) {
        ret[n] = ndyn::util::linspace(start[n], end[n], dims[n], i[n]);
    }
    return ret;
}

template<typename T>
constexpr
T ceil_div(
        T const & n,
        T const & d)
{
    auto n_ = double(n);
    auto d_ = double(d);
    return T(constexpr_ceil(n_ / d_));
}

namespace index {

///////////////////////////////////////////////////////////////////////////////
// Declarations
///////////////////////////////////////////////////////////////////////////////

template<typename T, size_t N>
constexpr
T hyperrectangular_to_linear(
        ndyn::vec<T, N> const &,
        ndyn::vec<T, N> const &);

template<typename T, size_t N>
constexpr
__host__ __device__
ndyn::vec<T, N> linear_to_hyperrectangular(
        ndyn::vec<T, N> const &,
        T const &);

///////////////////////////////////////////////////////////////////////////////
// Definitions
///////////////////////////////////////////////////////////////////////////////

template<typename T, size_t N>
constexpr
T hyperrectangular_to_linear(
        ndyn::vec<T, N> const & dims,
        ndyn::vec<T, N> const & i)
{
    T prod = 1;
    T ret = 0;
    for (size_t n = N - 1; n <= N; n--) {
        ret += i[n] * prod;
        prod *= dims[n];
    }
    return ret;
}

template<typename T, size_t N>
constexpr
__host__ __device__
ndyn::vec<T, N> linear_to_hyperrectangular(
        ndyn::vec<T, N> const & dims,
        T const & i_)
{
    auto i = i_;
    ndyn::vec<T, N> ret;
    for (size_t n = N - 1; n <= N; n--) {
        ret[n] = i % dims[n];
        i /= dims[n];
    }
    return ret;
}

}
// namespace index

}// namespace util

} // namespace ndyn
