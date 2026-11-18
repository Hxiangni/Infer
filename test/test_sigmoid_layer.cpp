#include <gtest/gtest.h>

#include <cmath>      // std::exp
#include <memory>
#include <stdexcept>
#include <vector>

#include "data/tensor.hpp"
#include "layer/sigmoid_layer.hpp"

namespace {

using infer::SigmoidLayer;
using infer::TensorF32;

// 参考实现：与公式 1/(1+e^-x) 一致，用于对拍
float Sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// 基础语义：逐元素应用 sigmoid，形状保持不变
TEST(SigmoidLayerTest, AppliesElementwise) {
  auto input = std::make_shared<TensorF32>(1, 2, 3);
  input->Fill(std::vector<float>{-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f});

  SigmoidLayer sigmoid;
  auto outputs = sigmoid.Forward({input});

  ASSERT_EQ(outputs.size(), 1u);
  ASSERT_NE(outputs[0], nullptr);

  // 形状必须保持不变
  EXPECT_EQ(outputs[0]->shapes(), (std::vector<size_t>{1, 2, 3}));

  EXPECT_FLOAT_EQ(outputs[0]->at(0, 0, 0), Sigmoid(-2.0f));
  EXPECT_FLOAT_EQ(outputs[0]->at(0, 0, 1), Sigmoid(-1.0f));
  EXPECT_FLOAT_EQ(outputs[0]->at(0, 0, 2), Sigmoid(0.0f));
  EXPECT_FLOAT_EQ(outputs[0]->at(0, 1, 0), Sigmoid(1.0f));
  EXPECT_FLOAT_EQ(outputs[0]->at(0, 1, 1), Sigmoid(2.0f));
  EXPECT_FLOAT_EQ(outputs[0]->at(0, 1, 2), Sigmoid(3.0f));

  // 一个可以用解析值直接验证的点
  EXPECT_FLOAT_EQ(outputs[0]->at(0, 0, 2), 0.5f);  // sigmoid(0) = 0.5
}

// 输出必须落在 (0, 1) 区间内，包括极端输入
TEST(SigmoidLayerTest, OutputInRangeZeroToOne) {
  auto input = std::make_shared<TensorF32>(1, 1, 4);
  input->Fill(std::vector<float>{-100.0f, -1e-6f, 1e-6f, 100.0f});

  SigmoidLayer sigmoid;
  auto outputs = sigmoid.Forward({input});

  ASSERT_EQ(outputs.size(), 1u);

  // 极端值：sigmoid(-100) ≈ 0，sigmoid(100) ≈ 1
  EXPECT_FLOAT_EQ(outputs[0]->at(0, 0, 0), 0.0f);
  EXPECT_FLOAT_EQ(outputs[0]->at(0, 0, 3), 1.0f);

  // 所有输出严格落在 [0, 1]
  for (size_t i = 0; i < outputs[0]->size(); ++i) {
    float v = outputs[0]->index(i);
    EXPECT_GE(v, 0.0f);
    EXPECT_LE(v, 1.0f);
  }
}

// 输入不能被修改（图里一个张量可能被多个算子共享）
TEST(SigmoidLayerTest, DoesNotModifyInput) {
  auto input = std::make_shared<TensorF32>(1, 1, 3);
  input->Fill(std::vector<float>{-1.0f, 0.0f, 1.0f});

  SigmoidLayer sigmoid;
  auto outputs = sigmoid.Forward({input});

  EXPECT_FLOAT_EQ(input->at(0, 0, 0), -1.0f);
  EXPECT_FLOAT_EQ(input->at(0, 0, 1), 0.0f);
  EXPECT_FLOAT_EQ(input->at(0, 0, 2), 1.0f);

  // 输出是独立的内存
  EXPECT_NE(outputs[0].get(), input.get());
}

// 单调性：输入越大输出越大
TEST(SigmoidLayerTest, Monotonic) {
  auto input = std::make_shared<TensorF32>(1, 1, 5);
  input->Fill(std::vector<float>{-2.0f, -1.0f, 0.0f, 1.0f, 2.0f});

  SigmoidLayer sigmoid;
  auto outputs = sigmoid.Forward({input});

  float prev = outputs[0]->index(0);
  for (size_t i = 1; i < outputs[0]->size(); ++i) {
    float cur = outputs[0]->index(i);
    EXPECT_GT(cur, prev);  // 严格递增
    prev = cur;
  }
}

// 多通道：每个通道独立处理
TEST(SigmoidLayerTest, HandlesMultipleChannels) {
  auto input = std::make_shared<TensorF32>(2, 1, 2);
  input->Fill(std::vector<float>{-1.0f, 1.0f, 0.0f, 2.0f});

  SigmoidLayer sigmoid;
  auto outputs = sigmoid.Forward({input});

  ASSERT_EQ(outputs.size(), 1u);
  EXPECT_EQ(outputs[0]->shapes(), (std::vector<size_t>{2, 1, 2}));

  EXPECT_FLOAT_EQ(outputs[0]->at(0, 0, 0), Sigmoid(-1.0f));
  EXPECT_FLOAT_EQ(outputs[0]->at(0, 0, 1), Sigmoid(1.0f));
  EXPECT_FLOAT_EQ(outputs[0]->at(1, 0, 0), 0.5f);
  EXPECT_FLOAT_EQ(outputs[0]->at(1, 0, 1), Sigmoid(2.0f));
}

// 一次传入多个张量，输出数量和顺序要对应
TEST(SigmoidLayerTest, HandlesMultipleInputs) {
  auto a = std::make_shared<TensorF32>(1, 1, 2);
  a->Fill(std::vector<float>{-1.0f, 0.0f});

  auto b = std::make_shared<TensorF32>(2, 1, 1);
  b->Fill(std::vector<float>{1.0f, -1.0f});

  SigmoidLayer sigmoid;
  auto outputs = sigmoid.Forward({a, b});

  ASSERT_EQ(outputs.size(), 2u);
  EXPECT_EQ(outputs[0]->shapes(), (std::vector<size_t>{1, 1, 2}));
  EXPECT_EQ(outputs[1]->shapes(), (std::vector<size_t>{2, 1, 1}));

  EXPECT_FLOAT_EQ(outputs[0]->at(0, 0, 0), Sigmoid(-1.0f));
  EXPECT_FLOAT_EQ(outputs[0]->at(0, 0, 1), 0.5f);
  EXPECT_FLOAT_EQ(outputs[1]->at(0, 0, 0), Sigmoid(1.0f));
  EXPECT_FLOAT_EQ(outputs[1]->at(1, 0, 0), Sigmoid(-1.0f));
}

// 空输入：返回空 vector，不崩溃
TEST(SigmoidLayerTest, EmptyInputs) {
  SigmoidLayer sigmoid;
  EXPECT_TRUE(sigmoid.Forward({}).empty());
}

// 空指针：按当前实现应该抛 std::invalid_argument
TEST(SigmoidLayerTest, NullInputThrows) {
  SigmoidLayer sigmoid;
  EXPECT_THROW(sigmoid.Forward({nullptr}), std::invalid_argument);
}

// 名字：默认值和自定义值
TEST(SigmoidLayerTest, Name) {
  SigmoidLayer default_sigmoid;
  EXPECT_EQ(default_sigmoid.name(), "Sigmoid");

  SigmoidLayer custom_sigmoid("my_sigmoid");
  EXPECT_EQ(custom_sigmoid.name(), "my_sigmoid");
}

}  // namespace
