cmake_minimum_required(VERSION 3.16)
project(infer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)   # 给 clangd / CLion 用

# Tensor 的底层容器是 arma::fcube
find_package(Armadillo REQUIRED)

# 日志库使用 spdlog（替代参考项目里的 glog）。
# spdlog 只负责输出日志（spdlog::info/warn/error），没有 glog 的 CHECK 宏；
# 所以 Tensor 的参数校验继续用 std::invalid_argument 异常。
find_package(spdlog REQUIRED)


# add_library(库名 类型 源文件...) 把 tensor.cpp指定的一堆cpp编译成静态库。
# infer_core 是自定义名字：infer 是项目名，core 表示"核心库"。
# 做成库而不是直接编译成可执行文件，是因为之后的 test/demo/benchmark
# 都要复用同一份 Tensor 代码——编译一次，多处链接。
# STATIC 表示生成 build/lib_infer_core.a。
add_library(infer_core STATIC
    source/data/tensor.cpp
    source/layer/relu_layer.cpp
    source/layer/sigmoid_layer.cpp
)

#--------------------------------------------------------------------------
# 编译阶段:cpp->.o目标文件
#   需要头文件路径，来识别类、函数声明、类型定义
#   只看声明不需要库里面的实现代码
# 链接阶段:一堆 .o + 库 → 最终可执行程序
#    找函数/符号的机器实现，只要库文件(.a/.so),不需要头文件
#
# infer_core 是静态库，包含了 tensor.cpp、relu_layer.cpp、sigmoid_layer.cpp 的机器码。库文件需要在链接时才能找到。
#--------------------------------------------------------------------------------
# 给名叫infer_core的库设置头文件搜索路径。
# target_include_directories(目标名 [访问权限] 路径...)
# PUBLIC：
# PRIVATE: 只有infer_core编译的时候能看到这个头文件，demo连接的时候看不到不会继承这个路径
# 

# 在编译infer_core的时候给编译器加上armadillo的头文件路径
# 这个路径**只给 infer_core 自己编译用**；以后别的程序（demo）
# 链接 infer_core，不会自动拿到 armadillo 的头路径。
target_include_directories(infer_core SYSTEM PUBLIC ${ARMADILLO_INCLUDE_DIRS})
target_include_directories(infer_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# infer_core要连接的东西
# 告诉链接器：**把这些库的符号拿进来解析**
# ${ARMADILLO_LIBRARIES} 是 find_package 帮你找到的、给链接器用的库文件完整路径。
# spdlog::spdlog 是 spdlog 导出的 imported target，会自动带上头文件路径和链接选项。
# PUBLIC：谁链接 infer_core，谁就自动链接这两个库。
target_link_libraries(infer_core
    PUBLIC ${ARMADILLO_LIBRARIES} spdlog::spdlog)

# 只给 infer_core 加编译警告选项。
# -Wall 常用警告，-Wextra 更多警告；PRIVATE 表示不传给 infer_tests。
target_compile_options(infer_core PRIVATE -Wall -Wextra)

# 开启 CMake 的测试功能，之后 ctest 才能发现用例，必须写在测试注册之前。
enable_testing()

# GTest 是 GoogleTest 的 CMake 包名（不是 googleTest）。
# 找到后可用 GTest::gtest（库）和 GTest::gtest_main（自带 main 函数）。
find_package(GTest REQUIRED)

# 编译所有测试源文件（Tensor + ReLU），产物为 build/infer_tests
add_executable(infer_tests
    test/test_tensor.cpp
    test/test_relu_layer.cpp
    test/test_sigmoid_layer.cpp
)

# 测试程序需要三样东西：
# 1) infer_core         —— 被测的 Tensor 实现
# 2) GTest::gtest       —— TEST / EXPECT_* 等宏
# 3) GTest::gtest_main  —— gtest 自带 main()，不用自己写
# PRIVATE：这些依赖只服务于 infer_tests，不向外传递。
target_link_libraries(infer_tests PRIVATE infer_core GTest::gtest GTest::gtest_main)

# 让 CTest 自动发现 gtest 里的每个 TEST()，逐个注册为独立用例。
# 构建后会运行一次 ./infer_tests --gtest_list_tests 来枚举用例。
include(GoogleTest)
gtest_discover_tests(infer_tests)
