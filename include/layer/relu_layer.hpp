// include/layer/relu_layer.hpp
#pragma once                       // 防止重复包含

#include <memory>
#include <string>
#include <utility>                 // std::move 需要
#include <vector>

#include "layer/abstract/layer.hpp" // 用工程内路径，而不是 "./..."

namespace infer {

class ReluLayer : public Layer {
 public:
  explicit ReluLayer(std::string name = "ReLU")
      : Layer(std::move(name)) {}

  std::vector<std::shared_ptr<TensorF32>> Forward(
      const std::vector<std::shared_ptr<TensorF32>>& inputs) override;
};

}  // namespace infer
