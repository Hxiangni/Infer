#pragma once

#include <memory>
#include <string>
#include <utility>  // std::move
#include <vector>

#include "data/tensor.hpp"

namespace infer {

using TensorF32 = Tensor<float>;

class Layer {
 public:
  explicit Layer(std::string name) : name_(std::move(name)) {}

  virtual ~Layer() = default;

  // 输入若干 tensor，返回若干 tensor；子类实现具体计算
  // vector 放的是 TensorF32 的智能指针,为什么用 shared_ptr<TensorF32>?
  // 因为一个 TensorF32 可能同时是多个结点的输入,生命周期不归于单个算子
  // 为什么输出也是vector:ReLU是1进1出，Add等多进多出接口一开是就留好
  virtual std::vector<std::shared_ptr<TensorF32>> Forward(
      const std::vector<std::shared_ptr<TensorF32>>& inputs) = 0;  // 纯虚函数

  const std::string& name() const { return name_; }

 private:
  std::string name_;
};  // class Layer

}  // namespace infer
