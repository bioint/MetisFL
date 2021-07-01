//
// MIT License
//
// Copyright (c) 2021 Project Metis
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
using namespace std;
namespace py = pybind11;

namespace projectmetis::controller {

// An simple function to add two integers.
int pybindaddints(int i, int j) { return i + j; }

// A simple function to add two arrays.
py::array_t<double> pybindaddarrs(py::array_t<double> &input1,
                                  py::array_t<double> &input2) {
  py::buffer_info buf1 = input1.request(), buf2 = input2.request();

  if (buf1.ndim != 1 || buf2.ndim != 1)
    throw std::runtime_error("Number of dimensions must be one");

  if (buf1.size != buf2.size)
    throw std::runtime_error("Input shapes must match");

  /* No pointer is passed, so NumPy will allocate the buffer */
  auto result = py::array_t<double>(buf1.size);

  py::buffer_info buf3 = result.request();

  auto *ptr1 = static_cast<double *>(buf1.ptr);
  auto *ptr2 = static_cast<double *>(buf2.ptr);
  auto *ptr3 = static_cast<double *>(buf3.ptr);

  for (size_t idx = 0; idx < buf1.shape[0]; idx++)
    ptr3[idx] = ptr1[idx] + ptr2[idx];

  return result;
}

} // namespace projectmetis::controller

PYBIND11_MODULE(pybind_example, m) {
  m.doc() = "pybind11 simple example plugin";
  m.def("add_int", &projectmetis::controller::pybindaddints,
        "A function that adds two numbers");
  m.def("add_arr", &projectmetis::controller::pybindaddarrs,
        "A function that adds two arrays");
}
