#pragma once

#include <functional>
#include <iterator>

namespace ndyn {

//
// TODO: use <algorithm> and <numeric> functions when they become constexpr.
//
// TODO: determine whether our loops get fused.
//
template<typename T, std::size_t N>
struct vec {
    T _elems[N];

    typedef T value_type;
    typedef std::size_t size_type;
    typedef std::ptrdiff_t difference_type;
    typedef value_type & reference;
    typedef value_type const & const_reference;
    typedef value_type * pointer;
    typedef value_type const * const_pointer;
    typedef value_type * iterator;
    typedef value_type const * const_iterator;
    typedef std::reverse_iterator<iterator> reverse_iterator;
    typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

    constexpr
    reference operator[](
            size_type n)
    {
        return _elems[n];
    }

    constexpr
    const_reference operator[](
            size_type n) const
    {
        return _elems[n];
    }

    constexpr
    pointer data() noexcept
    {
        return pointer(_elems);
    }

    constexpr
    const_pointer data() const noexcept
    {
        return const_pointer(_elems);
    }

    constexpr
    iterator begin() noexcept
    {
        return iterator(data());
    }

    constexpr
    const_iterator begin() const noexcept
    {
        return const_iterator(data());
    }

    constexpr
    const_iterator cbegin() const noexcept
    {
        return const_iterator(data());
    }

    constexpr
    iterator end() noexcept
    {
        return iterator(data() + N);
    }

    constexpr
    const_iterator end() const noexcept
    {
        return const_iterator(data() + N);
    }

    constexpr
    const_iterator cend() const noexcept
    {
        return const_iterator(data() + N);
    }

private:

    template<typename BinaryOp>
    constexpr
    static
    vec<T, N> map_bound(
            BinaryOp op,
            vec<T, N> const & xs,
            value_type const & x)
    {
        vec<T, N> ret;
        for (size_type n = 0; n < N; n++) {
            ret[n] = op(xs[n], x);
        }
        return ret;
    }

    template<typename BinaryOp>
    constexpr
    static
    value_type foldl(
            BinaryOp op,
            value_type const & e,
            vec<T, N> const & xs)
    {
        value_type ret = e;
        for (auto x : xs) {
            ret = op(ret, x);
        }
        return ret;
    }

    template<typename BinaryOp>
    constexpr
    static
    vec<T, N> zip_with(
            BinaryOp op,
            vec<T, N> const & lhs,
            vec<T, N> const & rhs)
    {
        vec<T, N> ret;
        for (size_type n = 0; n < N; n++) {
            ret[n] = op(lhs[n], rhs[n]);
        }
        return ret;
    }

public:

    constexpr
    friend
    vec<T, N> operator-(
            vec<T, N> const & lhs,
            vec<T, N> const & rhs)
    {
        return zip_with(std::minus<T>(), lhs, rhs);
    }

    constexpr
    friend
    vec<T, N> operator*(
            vec<T, N> const & lhs,
            vec<T, N> const & rhs)
    {
        return zip_with(std::multiplies<T>(), lhs, rhs);
    }

    constexpr
    friend
    vec<T, N> operator/(
            vec<T, N> const & lhs,
            T const & rhs)
    {
        return map_bound(std::divides<T>(), lhs, rhs);
    }

    constexpr
    value_type sum() const
    {
        return foldl(std::plus<T>(), 0, *this);
    }

    constexpr
    value_type product() const
    {
        return foldl(std::multiplies<T>(), 1, *this);
    }

    constexpr
    value_type inner_product(
            vec<T, N> const & rhs) const
    {
        return (*this * rhs).sum();
    }

    constexpr
    value_type norm() const
    {
        return sqrt(inner_product(*this));
    }

    constexpr
    value_type distance(
            vec<T, N> const & rhs) const
    {
        return (*this - rhs).norm();
    }
};

} // namespace ndyn
