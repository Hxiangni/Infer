#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>
#include <vector>

#include "data/tensor.hpp"
#include "layer/RelU_layer.hpp"

namespace {

using infer::ReLU_layer;
using infer::Tensor;

// 基础语义：正数不变，负数（含 0）变 0
TEST(ReLULayerTest, AppliesElementwise) {
  auto input = std::make_shared<Tensor>(1, 2, 3);
  input->Fill(std::vector<float>{-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f});

  ReLU_layer relu;
  auto outputs = relu.Forward({input});

  ASSERT_EQ(outputs.size(), 1u);
  ASSERT_NE(outputs[0], nullptr);

  // 形状必须保持不变
  EXPECT_EQ(outputs[0]->shapes(), (std::vector<size_t>{1, 2, 3}));

  EXPECT_FLOAT_EQ(outputs[0]->at(0, 0, 0), 0.0f);
  EXPECT_FLOAT_EQ(outputs[0]->at(0, 0, 1), 0.0f);
  EXPECT_FLOAT_EQ(outputs[0]->at(0, 0, 2), 0.0f);
  EXPECT_FLOAT_EQ(outputs[0]->at(0, 1, 0), 1.0f);
  EXPECT_FLOAT_EQ(outputs[0]->at(0, 1, 1), 2.0f);
  EXPECT_FLOAT_EQ(outputs[0]->at(0, 1, 2), 3.0f);
}

// 输入不能被修改（图里一个张量可能被多个算子共享）
TEST(ReLULayerTest, DoesNotModifyInput) {
  auto input = std::make_shared<Tensor>(1, 1, 3);
  input->Fill(std::vector<float>{-1.0f, 2.0f, -3.0f});

  ReLU_layer relu;
  auto outputs = relu.Forward({input});

  EXPECT_FLOAT_EQ(input->at(0, 0, 0), -1.0f);
  EXPECT_FLOAT_EQ(input->at(0, 0, 1), 2.0f);
  EXPECT_FLOAT_EQ(input->at(0, 0, 2), -3.0f);

  // 输出是独立的内存
  EXPECT_NE(outputs[0].get(), input.get());
}

// 多通道：每个通道独立处理
TEST(ReLULayerTest, HandlesMultipleChannels) {
  auto input = std::make_shared<Tensor>(2, 1, 2);
  input->Fill(std::vector<float>{-1.0f, 2.0f, 3.0f, -4.0f});

  ReLU_layer relu;
  auto outputs = relu.Forward({input});

  ASSERT_EQ(outputs.size(), 1u);
  EXPECT_EQ(outputs[0]->shapes(), (std::vector<size_t>{2, 1, 2}));

  EXPECT_FLOAT_EQ(outputs[0]->at(0, 0, 0), 0.0f);
  EXPECT_FLOAT_EQ(outputs[0]->at(0, 0, 1), 2.0f);
  EXPECT_FLOAT_EQ(outputs[0]->at(1, 0, 0), 3.0f);
  EXPECT_FLOAT_EQ(outputs[0]->at(1, 0, 1), 0.0f);
}

// 一次传入多个张量，输出数量和顺序要对应
TEST(ReLULayerTest, HandlesMultipleInputs) {
  auto a = std::make_shared<Tensor>(1, 1, 2);
  a->Fill(std::vector<float>{-1.0f, 2.0f});

  auto b = std::make_shared<Tensor>(2, 1, 1);
  b->Fill(std::vector<float>{-3.0f, 4.0f});

  ReLU_layer relu;
  auto outputs = relu.Forward({a, b});

  ASSERT_EQ(outputs.size(), 2u);
  EXPECT_EQ(outputs[0]->shapes(), (std::vector<size_t>{1, 1, 2}));
  EXPECT_EQ(outputs[1]->shapes(), (std::vector<size_t>{2, 1, 1}));

  EXPECT_FLOAT_EQ(outputs[0]->at(0, 0, 0), 0.0f);
  EXPECT_FLOAT_EQ(outputs[0]->at(0, 0, 1), 2.0f);
  EXPECT_FLOAT_EQ(outputs[1]->at(0, 0, 0), 0.0f);
  EXPECT_FLOAT_EQ(outputs[1]->at(1, 0, 0), 4.0f);
}

// 空输入：返回空 vector，不崩溃
TEST(ReLULayerTest, EmptyInputs) {
  ReLU_layer relu;
  EXPECT_TRUE(relu.Forward({}).empty());
}

// 空指针：按当前实现应该抛 std::invalid_argument
TEST(ReLULayerTest, NullInputThrows) {
  ReLU_layer relu;
  EXPECT_THROW(relu.Forward({nullptr}), std::invalid_argument);
}

// 名字：默认值和自定义值
TEST(ReLULayerTest, Name) {
  ReLU_layer default_relu;
  EXPECT_EQ(default_relu.get_name(), "ReLU");

  ReLU_layer custom_relu("my_relu");
  EXPECT_EQ(custom_relu.get_name(), "my_relu");
}

}  // namespace
