#include "piano_transcriber/audio.hpp"
#include "piano_transcriber/dsp.hpp"

#include <cstddef>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace {

std::vector<float> as_vector(const py::array_t<float, py::array::c_style | py::array::forcecast>& a) {
    const auto view = a.request();
    if (view.ndim != 1) {
        throw py::value_error("samples must be a one-dimensional array");
    }
    const auto* begin = static_cast<const float*>(view.ptr);
    return {begin, begin + view.size};
}
py::array_t<float> as_array(const std::vector<float>& values) {
    py::array_t<float> result(values.size());
    auto output = result.mutable_unchecked<1>();
    for (py::ssize_t index = 0; index < output.shape(0); ++index) {
        output(index) = values[static_cast<std::size_t>(index)];
    }
    return result;
}

}  // namespace

PYBIND11_MODULE(_native, module) {
    module.doc() = "Native DSP helpers for piano-transcriber";
    module.def("stereo_to_mono", [](const py::array_t<float, py::array::c_style |
                                                             py::array::forcecast>& samples) {
        return as_array(piano_transcriber::interleaved_to_mono(as_vector(samples), 2));
    });
    module.def("interleaved_to_mono",
               [](const py::array_t<float, py::array::c_style | py::array::forcecast>& samples,
                  const std::size_t channels) {
                   return as_array(
                       piano_transcriber::interleaved_to_mono(as_vector(samples), channels));
               });
    module.def("rms", [](const py::array_t<float, py::array::c_style | py::array::forcecast>& samples) {
        return piano_transcriber::rms(as_vector(samples));
    });
    module.def("peak_normalize",
               [](const py::array_t<float, py::array::c_style | py::array::forcecast>& samples,
                  const float target_peak) {
                   return as_array(
                       piano_transcriber::peak_normalize(as_vector(samples), target_peak));
               },
               py::arg("samples"), py::arg("target_peak") = 1.0F);
    module.def("frame_signal",
               [](const py::array_t<float, py::array::c_style | py::array::forcecast>& samples,
                  const std::size_t frame_size, const std::size_t hop_size, const bool pad_end) {
                   return piano_transcriber::frame_signal(as_vector(samples), frame_size, hop_size,
                                                           pad_end);
               },
               py::arg("samples"), py::arg("frame_size"), py::arg("hop_size"),
               py::arg("pad_end") = false);
}
