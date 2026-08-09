#pragma once

#include <cstddef>
#include <vector>

namespace piano_transcriber {

std::vector<float> interleaved_to_mono(const std::vector<float>& samples,
                                       std::size_t channels);

}  // namespace piano_transcriber
