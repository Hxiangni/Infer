// source/layer/RelU_layer.cpp
#include "layer/RelU_layer.hpp"

#include <stdexcept>
#include <utility>

namespace infer {

//逻辑就是申请一大片内存inputs.size()
//然后根据ReLU的规则遍历inputs然后赋值到申请的数组中
//大致类比成swap简单交换,output就是一个tensor副本，就是swap的哪个temp
std::vector<std::shared_ptr<Tensor>> ReLU_layer::Forward(
    const std::vector<std::shared_ptr<Tensor>>& inputs) {
  std::vector<std::shared_ptr<Tensor>> outputs;

  //提前开辟好内存
  //因为后面每次for都需要push_back一个tensor到vector中
  //避免反复扩容带来的损耗
  outputs.reserve(inputs.size());

  //const 不能通过input修改元素里的值
  for (const auto& input : inputs) {
    //检查空值
    if (input == nullptr) {
      throw std::invalid_argument("ReLU_layer::Forward: 输入不能为空");
    }

    auto output = std::make_shared<Tensor>(input->channels(),
                                           input->rows(),
                                           input->cols());

    //裸指针是局部的变量出作用域会销毁所以不用担心泄漏
    const float* src = input->raw_ptr();
    float* dst = output->raw_ptr();
    const size_t n = input->size();
    for (size_t i = 0; i < n; ++i) {
      dst[i] = src[i] > 0.0f ? src[i] : 0.0f;
    }

    outputs.push_back(std::move(output));
  }

  return outputs;
}

}  // namespace infer
