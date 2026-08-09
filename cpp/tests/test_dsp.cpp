#include "piano_transcriber/audio.hpp"
#include "piano_transcriber/dsp.hpp"

#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void require(const bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

bool close(const double left, const double right, const double tolerance = 1e-6) {
    return std::abs(left - right) <= tolerance;
}

void test_stereo_to_mono() {
    const auto mono = piano_transcriber::interleaved_to_mono({1.0F, -1.0F, 0.5F, 0.25F}, 2);
    require(mono.size() == 2, "stereo conversion returned incorrect size");
    require(close(mono[0], 0.0), "first mono frame is incorrect");
    require(close(mono[1], 0.375), "second mono frame is incorrect");
}

void test_rms() {
    require(close(piano_transcriber::rms({1.0F, -1.0F}), 1.0), "RMS is incorrect");
    require(close(piano_transcriber::rms({}), 0.0), "empty RMS must be zero");
}

void test_normalization() {
    const auto normalized = piano_transcriber::peak_normalize({-0.5F, 0.25F}, 1.0F);
    require(close(normalized[0], -1.0), "normalization negative peak is incorrect");
    require(close(normalized[1], 0.5), "normalization scale is incorrect");
    require(piano_transcriber::peak_normalize({0.0F, 0.0F}).size() == 2,
            "silence normalization changed size");
}

void test_framing() {
    const std::vector<float> signal{1, 2, 3, 4, 5};
    const auto unpadded = piano_transcriber::frame_signal(signal, 3, 2, false);
    require(unpadded.size() == 2, "unpadded frame count is incorrect");
    require(unpadded[1][2] == 5, "last complete frame is incorrect");
    const auto padded = piano_transcriber::frame_signal(signal, 4, 3, true);
    require(padded.size() == 2, "padded frame count is incorrect");
    require(padded[1][0] == 4 && padded[1][2] == 0, "padded frame contents are incorrect");
    require(piano_transcriber::frame_signal({}, 4, 2, true).empty(),
            "empty input must produce no frames");
}

}  // namespace

int main() {
    try {
        test_stereo_to_mono();
        test_rms();
        test_normalization();
        test_framing();
    } catch (const std::exception& error) {
        std::cerr << "FAILED: " << error.what() << '\n';
        return 1;
    }
    std::cout << "All native DSP tests passed\n";
    return 0;
}
