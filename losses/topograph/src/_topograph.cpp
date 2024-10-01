#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "topograph.hpp"

namespace py = pybind11;
using namespace pybind11::literals;

// struct S {
//     int x;
//     explicit S(int x): x{x} {
//         std::cout << "construct S(" << x << ")" << std::endl;
//     }
//     S(const S &s): x{s.x} {
//         std::cout << "copy S(const S &s)" << std::endl;
//     }
// };

// S make_value(int x) {
//     return S(x);
// }

// S make_double_value(int x) {
//     S s = make_value(x);
//     s.x = 2*x;
//     return s;
// }

// std::tuple<int, int> copy_test() {
//     S s1 = make_double_value(21);
//     S s2 = make_double_value(23);
//     return std::make_tuple(s1.x, s2.x);
// }

PYBIND11_MODULE(Topograph, m) {
    m.doc() = "Highly Efficient C++ Topograph implementation"; // optional module docstring

    //m.def("compute_batch_loss", &compute_batch_loss, "Computes the loss for a batch of prediction and ground truth pair", py::return_value_policy::take_ownership);
    m.def("compute_single_loss", &compute_single_loss, py::arg("argmax_pred").noconvert(), py::arg("argmax_gt").noconvert(),py::arg("num_classes"), py::call_guard<py::gil_scoped_release>(), py::return_value_policy::take_ownership);
    m.def("get_relabel_indices", &get_relabel_indices, py::arg("labelled_comps").noconvert(), py::arg("critical_nodes").noconvert(), py::arg("cluster_lengths").noconvert(), py::call_guard<py::gil_scoped_release>(), py::return_value_policy::take_ownership);
}