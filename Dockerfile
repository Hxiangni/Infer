# 基于当前已经装好 Armadillo / GTest 的容器快照
#
# 生成基础镜像（在宿主机执行）：
#   docker commit <容器ID> infer-base:20.04
#
# 然后构建：
#   docker build -t infer-dev:spdlog .
FROM infer-base:20.04

#------------------------------------------------------------------------------
# spdlog：Ubuntu 20.04 源里是 1.5.0
# 同时提供两个 CMake target：
#   spdlog::spdlog              -> 动态库（本项目使用）
#   spdlog::spdlog_header_only  -> 仅头文件
#------------------------------------------------------------------------------
# apt-get update 更新 apt 的软件源索引（包清单）
# -y：所有确认提问自动选 yes，构建的时候不需要人工交互
# --no-install-recommends：只安装必须的依赖，跳过那些推荐但非必需的附属包**，这是做镜像瘦身的关键
# libspdlog-dev：spdlog 的开发包（头文件 + cmake 配置 + 库）
# 删除 apt 下载下来的包索引缓存。
# apt update 下载的索引就放在 `/var/lib/apt/lists/`，这些文件对运行容器完全没用，删掉能显著缩小镜像体积
#------------------------------------------------------------------------------------
# 特别注意！
# 这个方式安装的 spdlog 是系统源自带版本，版本由你的基础 Ubuntu 镜像的源决定，不能自选版本。
# 
RUN apt-get update \
    && apt-get install -y --no-install-recommends libspdlog-dev \
    && rm -rf /var/lib/apt/lists/*
