#include "layer/sigmoid_layer.hpp"

#include <cmath>      // std::exp
#include <stdexcept>  // std::invalid_argument
#include <utility>    // std::move

namespace infer {

std::vector<std::shared_ptr<TensorF32>> SigmoidLayer::Forward(
    const std::vector<std::shared_ptr<TensorF32>>& inputs) {
  std::vector<std::shared_ptr<TensorF32>> outputs;
  outputs.reserve(inputs.size());

  for (const auto& input : inputs) {
    if (input == nullptr) {
      throw std::invalid_argument("SigmoidLayer::Forward: 输入不能为空");
    }

    auto output = std::make_shared<TensorF32>(input->channels(),
                                              input->rows(),
                                              input->cols());

    const float* src = input->raw_ptr();
    float* dst = output->raw_ptr();
    const size_t n = input->size();
    for (size_t i = 0; i < n; ++i) {
      dst[i] = 1.0f / (1.0f + std::exp(-src[i]));
    }

    outputs.push_back(std::move(output));
  }
  return outputs;
}

}  // namespace infer
