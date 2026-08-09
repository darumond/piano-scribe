#include "piano_transcriber/dsp.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace piano_transcriber {

double rms(const std::vector<float>& samples) noexcept {
    if (samples.empty()) {
        return 0.0;
    }
    double sum_squares = 0.0;
    for (const auto sample : samples) {
        const auto value = static_cast<double>(sample);
        sum_squares += value * value;
    }
    return std::sqrt(sum_squares / static_cast<double>(samples.size()));
}

std::vector<float> peak_normalize(const std::vector<float>& samples, const float target_peak) {
    if (target_peak <= 0.0F || target_peak > 1.0F) {
        throw std::invalid_argument("target_peak must be in the interval (0, 1]");
    }
    float peak = 0.0F;
    for (const auto sample : samples) {
        peak = std::max(peak, std::abs(sample));
    }
    if (peak == 0.0F) {
        return samples;
    }

    std::vector<float> normalized;
    normalized.reserve(samples.size());
    const auto scale = target_peak / peak;
    for (const auto sample : samples) {
        normalized.push_back(sample * scale);
    }
    return normalized;
}

std::vector<std::vector<float>> frame_signal(const std::vector<float>& samples,
                                             const std::size_t frame_size,
                                             const std::size_t hop_size,
                                             const bool pad_end) {
    if (frame_size == 0 || hop_size == 0) {
        throw std::invalid_argument("frame_size and hop_size must be greater than zero");
    }
    std::vector<std::vector<float>> frames;
    if (samples.empty()) {
        return frames;
    }

    for (std::size_t start = 0; start < samples.size(); start += hop_size) {
        const auto remaining = samples.size() - start;
        if (remaining < frame_size && !pad_end) {
            break;
        }
        std::vector<float> frame(frame_size, 0.0F);
        const auto copy_count = std::min(frame_size, remaining);
        std::copy_n(samples.begin() + static_cast<std::ptrdiff_t>(start), copy_count,
                    frame.begin());
        frames.push_back(std::move(frame));
        if (remaining <= frame_size) {
            break;
        }
    }
    return frames;
}

}  // namespace piano_transcriber
