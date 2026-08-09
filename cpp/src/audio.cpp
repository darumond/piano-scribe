#include "piano_transcriber/audio.hpp"

#include <stdexcept>

namespace piano_transcriber {

std::vector<float> interleaved_to_mono(const std::vector<float>& samples,
                                       const std::size_t channels) {
    if (channels == 0) {
        throw std::invalid_argument("channels must be greater than zero");
    }
    if (samples.size() % channels != 0) {
        throw std::invalid_argument("interleaved sample count must be divisible by channels");
    }

    const auto frame_count = samples.size() / channels;
    std::vector<float> mono(frame_count, 0.0F);
    for (std::size_t frame = 0; frame < frame_count; ++frame) {
        double sum = 0.0;
        for (std::size_t channel = 0; channel < channels; ++channel) {
            sum += samples[frame * channels + channel];
        }
        mono[frame] = static_cast<float>(sum / static_cast<double>(channels));
    }
    return mono;
}

}  // namespace piano_transcriber
