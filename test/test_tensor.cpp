#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

//定义标准异常类
#include <stdexcept>
#include <vector>

//tensor在cmake中被设置编译成了静态库
#include "data/tensor.hpp"

/** 先了解一下google test
 * -----------------------------------------------------------------
 * TEST(...)宏
 * TensorTest：测试套件名称
 * ConstructWithShape：测试用例名称，描述测试验证什么
 * ------------------------------------------------------------
 * 链接的GTest::gtest_main提供了main()，程序启动时会自动跑所有注册过的测试。
 *  EXPECT_EQ：精确相等，常用于整数、指针、字符串；
 *  EXPECT_FLOAT_EQ：浮点专用，允许 4 个 ULP 的误差
 * 预处理器按逗号分隔参数。std::vector<size_t>{2, 3, 4} 中间的逗号会被当成宏参数分隔符，
 * 导致编译错误。加一层括号 ( ... ) 把它包成一个整体，预处理器就不会拆开了。
 * 
 */

namespace {

using infer::tensor;
TEST(TensorTest, ConstructWithShape) {
  tensor<float> t(2, 3, 4);

  //u表示unsigned int因为t.channels()返回size_t
  EXPECT_EQ(t.channels(), 2u);
  EXPECT_EQ(t.rows(), 3u);
  EXPECT_EQ(t.cols(), 4u);
  EXPECT_EQ(t.size(), 24u);
  EXPECT_FALSE(t.empty());
    //EXPECT_EQ 是宏
  EXPECT_EQ(t.shapes(), (std::vector<size_t>{2, 3, 4}));

  // 构造后元素应被值初始化为 0
  for (size_t i = 0; i < t.size(); ++i) {
    EXPECT_FLOAT_EQ(t.index(i), 0.0f);
  }
}

TEST(TensorTest, ConstructWithShapes) {
  tensor<float> t(std::vector<size_t>{1, 2, 3});
  EXPECT_EQ(t.shapes(), (std::vector<size_t>{1, 2, 3}));

  // 执行某段代码，必须抛出指定类型的异常
  // 维度个数不对应该抛异常
  EXPECT_THROW(tensor<float>(std::vector<size_t>{1, 2}), std::invalid_argument);
}

TEST(TensorTest, AtAndIndexFollowMemoryLayout) {
  tensor<float> t(2, 2, 3);
  for (size_t c = 0; c < t.channels(); ++c) {
    for (size_t r = 0; r < t.rows(); ++r) {
      for (size_t col = 0; col < t.cols(); ++col) {
        t.at(c, r, col) = static_cast<float>(c * 100 + r * 10 + col);
      }
    }
  }

  // channel-major + row-major：(c * rows + r) * cols + col
  EXPECT_FLOAT_EQ(t.index(0), 0.0f);
  EXPECT_FLOAT_EQ(t.index(2), 2.0f);
  EXPECT_FLOAT_EQ(t.index(3), 10.0f);
  EXPECT_FLOAT_EQ(t.index(6), 100.0f);
  EXPECT_FLOAT_EQ(t.index(11), 112.0f);  // c=1, r=1, col=2 → 1*100 + 1*10 + 2

  // at 和 index 必须指向同一块内存
  EXPECT_FLOAT_EQ(t.at(1, 1, 2), t.index(11));

  // 越界访问抛异常
  EXPECT_THROW(t.at(2, 0, 0), std::invalid_argument);
  EXPECT_THROW(t.index(12), std::invalid_argument);
}

TEST(TensorTest, FillAndChannelPtr) {
  tensor<float> t(2, 2, 2);
  t.Fill(3.0f);
  for (size_t i = 0; i < t.size(); ++i) {
    EXPECT_FLOAT_EQ(t.index(i), 3.0f);
  }

  t.Fill(std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
  EXPECT_FLOAT_EQ(t.channel_ptr(0)[0], 1.0f);
  EXPECT_FLOAT_EQ(t.channel_ptr(0)[3], 4.0f);
  EXPECT_FLOAT_EQ(t.channel_ptr(1)[0], 5.0f);
  EXPECT_FLOAT_EQ(t.channel_ptr(1)[3], 8.0f);

  // 元素个数不匹配应抛异常
  EXPECT_THROW(t.Fill(std::vector<float>{1, 2, 3}), std::invalid_argument);
}

TEST(TensorTest, PaddingAddsBorder) {
  tensor<float> t(1, 2, 2);
  t.Fill(std::vector<float>{1, 2, 3, 4});

  t.Padding(1, 1, -1.0f);

  EXPECT_EQ(t.shapes(), (std::vector<size_t>{1, 4, 4}));

  // 四周是填充值
  EXPECT_FLOAT_EQ(t.at(0, 0, 0), -1.0f);
  EXPECT_FLOAT_EQ(t.at(0, 0, 3), -1.0f);
  EXPECT_FLOAT_EQ(t.at(0, 3, 0), -1.0f);
  EXPECT_FLOAT_EQ(t.at(0, 3, 3), -1.0f);

  // 中间保持原值
  EXPECT_FLOAT_EQ(t.at(0, 1, 1), 1.0f);
  EXPECT_FLOAT_EQ(t.at(0, 1, 2), 2.0f);
  EXPECT_FLOAT_EQ(t.at(0, 2, 1), 3.0f);
  EXPECT_FLOAT_EQ(t.at(0, 2, 2), 4.0f);
}

TEST(TensorTest, FlattenAndReView) {
  tensor<float> t(2, 2, 2);
  t.Fill(std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8});

  t.Flatten();
  EXPECT_EQ(t.shapes(), (std::vector<size_t>{1, 1, 8}));
  for (size_t i = 0; i < t.size(); ++i) {
    EXPECT_FLOAT_EQ(t.index(i), static_cast<float>(i + 1));
  }

  t.ReView(2, 2, 2);
  EXPECT_EQ(t.shapes(), (std::vector<size_t>{2, 2, 2}));
  EXPECT_FLOAT_EQ(t.at(1, 1, 1), 8.0f);

  // 元素个数对不上必须拒绝
  EXPECT_THROW(t.ReView(3, 3, 3), std::invalid_argument);
  EXPECT_THROW(t.ReRawShape(std::vector<size_t>{1, 2}), std::invalid_argument);
}

TEST(TensorTest, RawPtrOfEmptyTensorIsNull) {
  tensor<float> empty;
  EXPECT_TRUE(empty.empty());
  EXPECT_EQ(empty.raw_ptr(), nullptr);

  const tensor<float>& const_empty = empty;
  EXPECT_EQ(const_empty.raw_ptr(), nullptr);
}

TEST(TensorTest, CloneIsDeepCopy) {
  tensor<float> t(1, 1, 3);
  t.Fill(std::vector<float>{1, 2, 3});

  tensor<float> copy = t.Cloqne();
  EXPECT_EQ(copy.shapes(), t.shapes());
  EXPECT_FLOAT_EQ(copy.at(0, 0, 0), 1.0f);

  copy.at(0, 0, 0) = 100.0f;
  EXPECT_FLOAT_EQ(t.at(0, 0, 0), 1.0f);  // 原张量不受影响
  EXPECT_FLOAT_EQ(copy.at(0, 0, 0), 100.0f);
}

// 依赖冒烟测试：确认 spdlog 的头文件和库都能正常编译、链接、运行
TEST(ToolchainTest, SpdlogWorks) {
  spdlog::info("spdlog {}.{}.{} linked", SPDLOG_VER_MAJOR, SPDLOG_VER_MINOR,
               SPDLOG_VER_PATCH);
  SUCCEED();
}

}  // namespace
