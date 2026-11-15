#pragma once

#include <cstddef>
#include <vector>

namespace infer {

// 三维张量：channel（通道） × row（行） × col（列）
//
// 内存布局：channel-major，即
//   data_[(c * rows_ + r) * cols_ + col]
// 同一通道内是行优先（row-major）排列，和 OpenCV / Armadillo 的 Mat 一致。
//
// 注意：这是模板类，成员函数的实现放在 tensor.cpp 里，
// 并在文件末尾对 float / double 做了显式实例化（explicit instantiation）。
// 如果你想用 tensor<int>，需要在 tensor.cpp 里再补一行 template class tensor<int>;
// 否则会链接报错。等 Tensor 稳定下来，建议改为固定 float，和参考项目一致。
template <typename T>
class tensor {
 public:
  // 默认构造：空张量，shape 为 0×0×0
  tensor() = default;

  // 按维度构造，元素默认值初始化为 T{}
  // 传入int编译器会做隐式整型转换
  tensor(size_t channels, size_t rows, size_t cols);

  // 按 {channels, rows, cols} 构造
  explicit tensor(const std::vector<size_t>& shapes);

  /**
   * 拷贝 / 移动：交给编译器生成即可（vector 成员会正确深拷贝）
   * 编译器生成的拷贝构造函数就是对每个成员调用他自己的拷贝构造
   * -------------------------------------------------------
   * int* 拷贝构造就是复制地址->浅拷贝
   * vector就是调用vector的拷贝构造
   */
  tensor(const tensor& other) = default;
  //默认移动构造
  tensor(tensor&& other) noexcept = default;
  tensor& operator=(const tensor& other) = default;
  tensor& operator=(tensor&& other) noexcept = default;
  ~tensor() = default;

  // ---- 形状 ----
  size_t channels() const { return channels_; }
  size_t rows() const { return rows_; }
  size_t cols() const { return cols_; }
  size_t size() const { return data_.size(); }
  bool empty() const { return data_.empty(); }
  std::vector<size_t> shapes() const { return {channels_, rows_, cols_}; }

  /**---------------------- 元素访问 --------------------
   * at：按 (c, r, col) 访问，带越界检查
   * ---------------------------------------------------
   * 问题：为什么成员函数末尾加个 const 就能重载？
   *
   * 1) 非静态成员函数有一个隐含对象参数：
   *      T& at(...)              -> 隐含参数 tensor<T>&
   *      const T& at(...) const  -> 隐含参数 const tensor<T>&
   *    两个函数的隐含参数类型不同，所以构成重载。
   *    这个隐含参数的作用是,根据隐含参数来选择不同的重载函数
   *
   * 2) 函数体里的 this 是另一回事：
   *      this只是一个表达式，它是语言内置的关键字表达式，在不同成员函数里，this 这个表达式的类型本来就不同！！！
   *      表达式有类型 有值类别--在不同 cv 限定的成员函数里，this 类型不同
   * 
   *      非 const 成员：this 的类型是 tensor<T>*，值类别是 prvalue
   *      const 成员：  this 的类型是 const tensor<T>*，值类别是 prvalue
   * 
   *     this 永远是 prvalue（一个地址值，不是变量），两种情况都一样；
   *     区别只在指向的对象带不带 const；
   *     一旦指向 const 对象，通过它访问到的成员也全是 const 的。
   *    这也说明为啥只有const对象才能调用末尾有const的成员函数
   * 
   *-----------------------------------------------------
   * 右值表达式不标识一个持久对象，所以不能对它取地址
   *-----------------------------------------------------
   * 问题：为啥要重载这俩呢？写一个读不就行了？
   * 
   * 因为如果遇到tensor const a;时只能调用const类型的成员函数 
   * 所以要专门再写一个const版本的成员函数
   *------------------------------------------------------
   **/
  T& at(size_t channel, size_t row, size_t col);
  // 错误：const 对象不能调用非 const 成员函数具体原因查笔记去
  //
  // 为啥上面不加const下面加了const?
  // 因为只有const类才能调用这个重载,const类说明其本身不能被改变
  // 都不能变了你还返回一个可以变的引用干啥？左右脑互博吗？
  const T& at(size_t channel, size_t row, size_t col) const;

  // index：按扁平下标访问，带越界检查
  T& index(size_t offset);
  const T& index(size_t offset) const;


  /*---- 连续内存访问 ----
  * data_.data()返回指向首元素的裸指针
  * 如果data_为空，返回nullptr
  * C++标准不保证空的vector的data()返回的指针是否为nullptr
  */
  T* raw_ptr() noexcept { return data_.empty() ? nullptr : data_.data(); }
  const T* raw_ptr() const noexcept {
    return data_.empty() ? nullptr : data_.data();
  }

  // 返回第 channel 个通道的起始地址，可访问长度为 rows() * cols()。
  // 通道下标越界时抛 std::invalid_argument，因此不能标记 noexcept。
  // 生命周期规则同 raw_ptr()。
  T* channel_ptr(size_t channel);
  const T* channel_ptr(size_t channel) const;

  std::vector<T>& data() { return data_; }
  const std::vector<T>& data() const { return data_; }

  // ---- 填充 ----
  void Fill(T value);//全部赋成value
  void Fill(const std::vector<T>& values);//类似拷贝赋值？
  //定义在类内,默认是inline，编译器会直接内联掉,没有任何函数调用开销
  void Ones() { Fill(T{1}); }
  void Zeros() { Fill(T{0}); }

  // ---- 变形 ----
  // 四周补 padding，shape 变为 (c, rows+2*pad_row, cols+2*pad_col)
  void Padding(size_t pad_row, size_t pad_col, T padding_value = T{0});

  // 展平成 (1, 1, N)，元素顺序不变
  void Flatten();

  // 重新解释维度，元素个数必须保持不变，否则抛 std::invalid_argument
  // 重新维度解释channels=1,rows=1,cols=N即是展平
  void ReView(size_t channels, size_t rows, size_t cols);
  // 根据shapes重新维度解释
  void ReRawShape(const std::vector<size_t>& shapes);

  // 深拷贝
  tensor Cloqne() const { return *this; }

 private:
  size_t Offset(size_t channel, size_t row, size_t col) const {
    return (channel * rows_ + row) * cols_ + col;
  }
  // channel大小
  size_t PlaneSize() const { return rows_ * cols_; }

  size_t channels_ = 0;
  size_t rows_ = 0;
  size_t cols_ = 0;


  /**
   * 内存由vector负责申请、扩容、释放、析构时自动回收
   * 内存归Tensor所有、调用方只是借用
   * vector自己的拷贝构造会重新分配内存并复制元素
   */
  std::vector<T> data_;
};

}  // namespace infer
