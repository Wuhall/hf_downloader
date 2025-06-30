# HuggingFace 模型下载器

基于 aria2 实现的高性能 HuggingFace 大模型文件下载工具。特别适用于分片存储的大模型下载。

## 功能特色

- 🔍 自动检测文件命名模式和文件数量
- ✨ 使用 aria2 并行下载
- 🔄 支持断点续传
- 🚀 优化的连接参数
- 📝 详细日志记录
- 🔍 已有文件自动跳过
- ⚡ 单文件多连接下载
- 🎯 灵活的文件命名模式匹配
- 🗂️ 自动目录整理

## 先决条件

1. Python 3.7 及以上
2. 系统已安装 aria2c
3. Python 依赖包：`requests`

### 安装依赖

```bash
# 安装 aria2
sudo apt-get install aria2  # Ubuntu/Debian
brew install aria2         # macOS
# Windows: 从 https://github.com/aria2/aria2/releases 下载

# 安装 Python 依赖
pip install requests
```

## 使用方法

### 基本用法（自动检测）

只需指定仓库 ID 即可：
```python
from hf_model_downloader import HFModelDownloader

# 全自动检测
downloader = HFModelDownloader(
    repo_id="deepseek-ai/DeepSeek-R1"
)
downloader.start_download()
```

下载器会自动完成：
1. 检测仓库中的文件命名模式
2. 统计文件数量
3. 判断起始索引
4. 创建合适的输出目录
5. 并行下载所有文件

### 高级用法

你可以覆盖任何自动检测的参数：

```python
# 示例1：自动检测，指定输出目录
downloader = HFModelDownloader(
    repo_id="deepseek-ai/DeepSeek-R1",
    output_dir="my-model"
)

# 示例2：指定命名模式，自动检测文件数量
downloader = HFModelDownloader(
    repo_id="organization/model-name",
    pattern="pytorch_model-{i:02d}.bin"
)

# 示例3：完全手动配置
downloader = HFModelDownloader(
    repo_id="organization/model-name",
    pattern="model.safetensors.{i:02d}",
    num_files=4,
    start_index=0,
    max_concurrent_downloads=8
)
```

### 支持的文件命名模式

自动检测支持常见的模型文件命名规则：

1. `model-00001-of-00163.safetensors`（如 DeepSeek 模型）
2. `pytorch_model-00001.bin`（标准 PyTorch 检查点）
3. `model.safetensors.00001`（另一种编号方式）

支持的扩展名：`.safetensors`、`.bin`、`.pt`、`.pth`

### 配置选项

- `repo_id`：HuggingFace 仓库 ID（如 "deepseek-ai/DeepSeek-R1"）
- `pattern`：文件命名模式（未指定时自动检测）
- `output_dir`：保存文件的目录（默认取 repo_id 最后一部分）
- `num_files`：文件总数（未指定时自动检测）
- `start_index`：起始索引（未指定时自动检测）
- `max_concurrent_downloads`：并行下载数（默认16）
- `max_connection_per_server`：每服务器最大连接数（默认16）
- `min_split_size`：并行下载的最小分片大小（默认"1M"）

## 生成的文件

- `aria2_urls.txt`：包含下载链接和输出路径
- `aria2_download.log`：详细下载日志

## 生成命令示例

只生成 aria2 命令而不下载：

```python
from hf_model_downloader import HFModelDownloader

downloader = HFModelDownloader(repo_id="organization/model-name")
cmd = downloader.generate_aria2_command()
print(" ".join(cmd))
```

## 常见问题

1. **模式检测失败**：
   - 检查仓库是否公开可访问
   - 确认文件命名是否符合支持的模式
   - 尝试手动指定 pattern

2. **找不到 aria2c**：
   - 确认 aria2 已安装
   - 检查 aria2c 是否在 PATH 路径下

3. **下载中断**：
   - 直接重新运行脚本即可自动断点续传

4. **连接问题**：
   - 尝试降低 `max_concurrent_downloads` 和 `max_connection_per_server`
   - 检查网络连接
   - 确认磁盘空间充足

## 参与贡献

欢迎提交 issue 和功能改进建议！

## 与 AI Agent 协作

本下载器设计为 Agent 友好，便于与如 Cursor 的 Claude 等 AI 助手配合使用。

### 在 Cursor 中使用

1. 在 Cursor 切换到 Composer 模式
2. 将模式从 `normal` 改为 `agent`
3. 在上下文中包含本 README：
   ```
   @README
   ```
4. 用自然语言让 agent 下载模型：
   ```
   请帮我用命令行下载 DeepSeek Coder 模型。
   ```
   或
   ```
   你能帮我下载 CodeLlama 吗？我已经安装了下载器，想用 CLI 下载。
   ```

Agent 会自动：
1. 理解你的请求
2. 判断正确的仓库 ID
3. 生成并执行合适的命令
4. 监控下载进度

### Agent 交互示例

```
用户：你能帮我用命令行下载 DeepSeek-R1 模型吗？URL 是 https://huggingface.co/deepseek-ai/DeepSeek-R1/tree/main

Agent：我来帮你用命令行下载 DeepSeek-R1 模型。
我会用合适的仓库 ID 运行下载器：

运行命令：python3 hf_model_downloader.py --repo-id deepseek-ai/DeepSeek-R1

下载器会：
- 自动检测文件命名和数量
- 设置 aria2 并行下载
- 创建输出目录
- 在终端显示下载进度

你可以随时用 Ctrl+C 中断下载，之后再次运行脚本即可断点续传。

用户：可以下载到指定目录吗？

Agent：我会用自定义输出目录运行下载器：

运行命令：python3 hf_model_downloader.py --repo-id deepseek-ai/DeepSeek-R1 --output-dir ./my-models/deepseek
```

### 命令行选项

Agent 理解以下命令行参数：
```bash
python3 hf_model_downloader.py [选项]

选项：
  --repo-id                  HuggingFace 仓库 ID
  --output-dir              自定义输出目录
  --pattern                 自定义文件命名模式（默认自动检测）
  --num-files               文件数量（默认自动检测）
  --start-index             起始索引（默认自动检测）
  --max-concurrent          最大并行下载数（默认：16）
  --max-connections         每服务器最大连接数（默认：16）
  --min-split-size          最小分片大小（默认："1M"）
```

### Agent 友好特性

- 自然语言理解命令行选项
- 智能默认值，最小化配置需求
- 清晰的终端输出
- 进度报告
- 错误处理
- 可恢复的下载

### Agent 交互技巧

- 指定要下载的模型，并提供 URL
- 提及任何特殊要求（如输出目录）
- 如需要可要求解释命令
- 请求下载状态检查
- 寻求错误信息帮助
