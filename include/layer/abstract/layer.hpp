#include <string>
#include <vector>
#include <memory>
#include "data/tensor.hpp"
namespace infer{
using Tensor = tensor<float>;

class Layer{
public:
    explicit Layer(std::string name) : name_(std::move(name)){}

    virtual ~Layer()=default;

    //输入若干tensor，返回若干tensor;子类实现具体计算
    //vector放的是Tensor的智能指针,为什么用shared_ptr<Tensor>?
    //因为一个Tensor可能同时是多个结点的输入,生命周期不归于单个算子
    //为什么输出也是vector:ReLU是1进1出，Add等多进多出接口一开是就留好
    virtual std::vector<std::shared_ptr<Tensor>> Forward(
        const std::vector<std::shared_ptr<Tensor>>& inputs) = 0;//纯虚函数

    const std::string& get_name() const {return name_;}


private:
    std::string name_;
};//class Layer

}//namespace infer