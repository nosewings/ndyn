#pragma once

#include <ndyn/vec.cuh>

namespace ndyn {

namespace model {

template<typename T>
struct ricker_2d {
    T a, b, r, s;

    constexpr
    ndyn::vec<T, 2> operator()(
            T const & x,
            T const & y)
    {
        return {x*exp(r - x - a*y), y*exp(s - y - b*x)};
    }

    constexpr
    ndyn::vec<T, 2> operator()(
            ndyn::vec<T, 2> const & xy)
    {
        return (*this)(xy[0], xy[1]);
    }

    constexpr
    ndyn::vec<T, 2> coexistence()
    {
        return (vec<T, 2> { r - a * s, s - b * r }) / (1 - a * b);
    }

};

template<typename T>
struct ricker_3d {
    T a, b, c, d, e, f, r, s, t;

    constexpr
    ndyn::vec<T, 3> operator()(
            T const & x,
            T const & y,
            T const & z)
    {
        return {x*exp(r - x - a*y - b*z), y*exp(s - y - c*x - d*z), z*exp(t - z - e*x - f*y)};
    }

    constexpr
    ndyn::vec<T, 3> operator()(
            ndyn::vec<T, 3> const & xyz)
    {
        return (*this)(xyz[0], xyz[1], xyz[2]);
    }
};

} // namespace model

} // namespace ndyn
