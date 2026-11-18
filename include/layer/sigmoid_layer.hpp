#pragma once // 防止重复包含

#include <memory>
#include <string>
#include <utility> // std::move 需要
#include <vector>

#include "layer/abstract/layer.hpp" // 用工程内路径，而不是 "./..."

namespace infer
{
/*
 *只有一两行：构造函数转发、getter/setter、empty()、size()；
 *调用频率极高，希望内联；
 *实现不依赖额外的头文件。
*/
    class SigmoidLayer : public Layer
    {
    public:
        explicit SigmoidLayer(std::string name="Sigmoid") :
            Layer(std::move(name)) {};

        std::vector<std::shared_ptr<TensorF32>> Forward(
            const std::vector<std::shared_ptr<TensorF32>> &inputs) override;
    };

} // namespace infer