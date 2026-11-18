#include "data/tensor.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace infer {

namespace {// 匿名命名空间

/**
 * 文件内自由函数
 * 它是自由函数，不属于 tensor<T>，没有 this，访问不到 channels_、data_ 这些成员；
 * 匿名命名空间会给里面的符号内部链接效果等价于：
 * static void CheckShape(bool condition, const std::string& message) { ... }
 * -----------------------------------------------------------------------------
 * 两种写法在 C++ 里基本等价，区别只是匿名命名空间对类、变量也适用，更通用，所以是现代 C++ 的惯用写法。
 */
void CheckShape(bool condition, const std::string& message) {
  //throw 之后，当前函数的所有后续语句（包括 return）都不执行；控制权直接交给最近的 catch
  //并且还会销毁从throw点一直到catch之间的所有栈帧,全部逐层展开销毁
  if (!condition) {
    //维度检查失败时统一抛这个异常，避免到处写 throw std::invalid_argument(...)
    //std::invalid_argument传入了一个非法、不合逻辑的参数。
    throw std::invalid_argument(message);
  }
}

}  // namespace匿名命名空间结束

template <typename T>
Tensor<T>::Tensor(size_t channels, size_t rows, size_t cols)
    : channels_(channels), rows_(rows), cols_(cols) {
      //assign全部清理后重新申请内存
  data_.assign(channels * rows * cols, T{});
}

template <typename T>
Tensor<T>::Tensor(const std::vector<size_t>& shapes) {
  CheckShape(shapes.size() == 3,
             "Tensor(const std::vector<size_t>&): shapes 必须是 {channels, rows, cols}");
  channels_ = shapes[0];
  rows_ = shapes[1];
  cols_ = shapes[2];
  // 先销毁原来容器里所有元素，size 变成 0，再添加新元素
  // T{}：值初始化，也就是T 类型的默认零值
  data_.assign(channels_ * rows_ * cols_, T{});
}

template <typename T>
T& Tensor<T>::at(size_t channel, size_t row, size_t col) {
  CheckShape(channel < channels_ && row < rows_ && col < cols_,
             "Tensor::at: 下标越界");
  return data_[Offset(channel, row, col)];
}

//const对象肯定要返回const引用啊不返回const被改变了怎么办
template <typename T>
const T& Tensor<T>::at(size_t channel, size_t row, size_t col) const {
  CheckShape(channel < channels_ && row < rows_ && col < cols_,
             "tensor::at: 下标越界");
  return data_[Offset(channel, row, col)];
}

template <typename T>
T& Tensor<T>::index(size_t offset) {
  CheckShape(offset < data_.size(), "tensor::index: 下标越界");
  return data_[offset];
}

template <typename T>
const T& Tensor<T>::index(size_t offset) const {
  CheckShape(offset < data_.size(), "tensor::index: 下标越界");
  return data_[offset];
}

template <typename T>
T* Tensor<T>::channel_ptr(size_t channel) {
  //如果这个条件不成立，就抛异常
  CheckShape(channel < channels_, "tensor::channel_ptr: 通道下标越界");
  return data_.data() + channel * PlaneSize();
}

template <typename T>
const T* Tensor<T>::channel_ptr(size_t channel) const {
  CheckShape(channel < channels_, "tensor::channel_ptr: 通道下标越界");
  return data_.data() + channel * PlaneSize();
}

template <typename T>
void Tensor<T>::Fill(T value) {
  //std::fill吧一个区域内的所有元素都赋值为value
  std::fill(data_.begin(), data_.end(), value);

  /*
   * 手写下面的循环也可以
   for (auto it = data_.begin(); it != data_.end(); ++it) {
   *it = value;
    }

    std::fill 对连续内存（vector）通常会生成高效实现，甚至被优化成 memset 或 SIMD 指令。
    手写循环经过 -O2 也常常能达到同样效果，但用标准算法更稳妥。

  */

}

template <typename T>
void Tensor<T>::Fill(const std::vector<T>& values) {
  CheckShape(values.size() == data_.size(),
             "tensor::Fill: values 的元素个数必须等于张量元素个数");
  data_ = values;
}

template <typename T>
void Tensor<T>::Padding(size_t pad_row, size_t pad_col, T padding_value) {
  const size_t new_rows = rows_ + 2 * pad_row;
  const size_t new_cols = cols_ + 2 * pad_col;

  //重新申请内存全部初始化成padding_value
  std::vector<T> padded(channels_ * new_rows * new_cols, padding_value);

  //将原始数据复制到新内存中
  for (size_t c = 0; c < channels_; ++c) {
    for (size_t r = 0; r < rows_; ++r) {
      for (size_t col = 0; col < cols_; ++col) {
        padded[(c * new_rows + (r + pad_row)) * new_cols + (col + pad_col)] =
            data_[Offset(c, r, col)];
      }
    }
  }

  data_.swap(padded);
  rows_ = new_rows;
  cols_ = new_cols;
}

template <typename T>
void Tensor<T>::Flatten() {
  channels_ = 1;
  rows_ = 1;
  cols_ = data_.size();
}

//换视角解释vector
template <typename T>
void Tensor<T>::ReView(size_t channels, size_t rows, size_t cols) {
  CheckShape(channels * rows * cols == data_.size(),
             "tensor::ReView: 新的维度乘积必须等于原元素个数");
  channels_ = channels;
  rows_ = rows;
  cols_ = cols;
}

template <typename T>
void Tensor<T>::ReRawShape(const std::vector<size_t>& shapes) {
  CheckShape(shapes.size() == 3,
             "tensor::ReRawShape: shapes 必须是 {channels, rows, cols}");
  ReView(shapes[0], shapes[1], shapes[2]);
}

/**
 * 说明：
 * 本模板类采用声明与实现分离的写法：头文件放声明，实现写在 cpp 文件中。
 * 因此若要在其他 cpp 文件中使用该模板类，需要在 cpp 文件中进行显式实例化。
 */
template class Tensor<float>;
template class Tensor<double>;

}  // namespace infer


