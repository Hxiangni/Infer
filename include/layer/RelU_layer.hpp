// include/layer/RelU_layer.hpp
#pragma once                       // 防止重复包含

#include <memory>
#include <string>
#include <utility>                 // std::move 需要
#include <vector>

#include "layer/abstract/layer.hpp" // 用工程内路径，而不是 "./..."

namespace infer {

class ReLU_layer : public Layer {
 public:
  explicit ReLU_layer(std::string name = "ReLU")
      : Layer(std::move(name)) {}

  std::vector<std::shared_ptr<Tensor>> Forward(
      const std::vector<std::shared_ptr<Tensor>>& inputs) override;
};

}  // namespace infer
