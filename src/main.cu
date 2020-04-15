#include <vector>
using namespace std;

#include <cuda/api_wrappers.hpp>
#include <matplotlibcpp.h>
namespace plt = matplotlibcpp;

#include <ndyn/model.cuh>
#include <ndyn/plot.cuh>
using namespace ndyn;
using namespace ndyn::model;

void plot_trajectories()
{
    auto device = cuda::device::current::get();
    ricker_2d<double> model { 0.5, 0.5, 2.0, 2.0 };
    vec<double, 2> start { 0.0, 0.0 };
    vec<double, 2> end { 2.0, 2.0 };
    vec<size_t, 2> dims { 21, 21 };
    plot::vector_field_2d(model, start, end, dims, device);
    auto fp = model.coexistence();
//    vector<vec<double, 2>> init_points { { 0.4, 2.0 }, { 0.8, 2.0 },
//            { 1.2, 2.0 }, { 1.6, 2.0 }, { 2.0, 1.6 }, { 2.0, 1.2 },
//            { 2.0, 0.8 }, { 2.0, 0.4 } };
    vector<vec<double, 2>> init_points { { 0.8, 2.0 }, { 1.6, 2.0 },
            { 2.0, 1.2 }, { 2.0, 0.4 }, { 0.1, 1.9 }, { 0.1, 0.3 } };
    plot::trajectories_2d<double>(model, init_points, 7, device);
    plt::title("r = s = 2.0");
    plt::show();
}

void plot_bifurcation_1d_2d()
{
    auto device = cuda::device::current::get();
    double a = 0.5;
    double b = 0.5;
    double r = 5.0;
    auto f = [a, b, r] __host__ __device__ (vec<double, 1> s, vec<double, 2> xy) {return (ricker_2d<double> {a, b, r, s[0]})(xy);};
    double start = 0.0;
    double end = 4.0;
    size_t dim = 1024;
    vec<double, 2> x0 { 0.1, 0.5 };
    size_t m1 = 32;
    size_t m2 = 128;
    plot::bifurcation_1d_2d(f, start, end, dim, x0, m1, m2, device);
    plt::suptitle("r = 5.0");
    plt::show();
}

int main()
{
    plot_bifurcation_1d_2d();
}
