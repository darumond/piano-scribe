#pragma once

#include <cstddef>
#include <vector>

namespace piano_transcriber {

double rms(const std::vector<float>& samples) noexcept;
std::vector<float> peak_normalize(const std::vector<float>& samples,
                                  float target_peak = 1.0F);
std::vector<std::vector<float>> frame_signal(const std::vector<float>& samples,
                                             std::size_t frame_size,
                                             std::size_t hop_size,
                                             bool pad_end = false);

}  // namespace piano_transcriber
