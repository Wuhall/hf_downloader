# HuggingFace Model Downloader

A high-performance utility for downloading large model files from HuggingFace using aria2. Especially useful for models split into multiple files.

## Features

- 🔍 Automatic detection of file patterns and counts
- ✨ Parallel downloading with aria2
- 🔄 Automatic resume of interrupted downloads
- 🚀 Optimized connection settings
- 📝 Detailed logging
- 🔍 Skip existing files
- ⚡ Multiple connections per file
- 🎯 Flexible file pattern matching
- 🗂️ Automatic directory organization

## Prerequisites

1. Python 3.7+
2. aria2c installed on your system
3. Python packages: `requests`

### Installing Requirements

```bash
# Install aria2
sudo apt-get install aria2  # Ubuntu/Debian
brew install aria2         # macOS
# Windows: Download from https://github.com/aria2/aria2/releases

# Install Python dependencies
pip install requests
```

## Usage

### Basic Usage (Automatic Detection)

Simply specify the repository ID:
```python
from hf_model_downloader import HFModelDownloader

# Auto-detect everything
downloader = HFModelDownloader(
    repo_id="deepseek-ai/DeepSeek-R1"
)
downloader.start_download()
```

The downloader will:
1. Detect the file pattern in the repository
2. Count the number of files
3. Determine the starting index
4. Create appropriate output directory
5. Download all files in parallel

### Advanced Usage

You can override any auto-detected values:

```python
# Example 1: Auto-detect with custom output directory
downloader = HFModelDownloader(
    repo_id="deepseek-ai/DeepSeek-R1",
    output_dir="my-model"
)

# Example 2: Specify pattern but auto-detect count
downloader = HFModelDownloader(
    repo_id="organization/model-name",
    pattern="pytorch_model-{i:02d}.bin"
)

# Example 3: Full manual configuration
downloader = HFModelDownloader(
    repo_id="organization/model-name",
    pattern="model.safetensors.{i:02d}",
    num_files=4,
    start_index=0,
    max_concurrent_downloads=8
)
```

### Supported File Patterns

The auto-detection supports common model file patterns:

1. `model-00001-of-00163.safetensors` (e.g., DeepSeek models)
2. `pytorch_model-00001.bin` (e.g., standard PyTorch checkpoints)
3. `model.safetensors.00001` (e.g., alternative numbering)

Extensions supported: `.safetensors`, `.bin`, `.pt`, `.pth`

### Configuration Options

- `repo_id`: HuggingFace repository ID (e.g., "deepseek-ai/DeepSeek-R1")
- `pattern`: Filename pattern (auto-detected if not specified)
- `output_dir`: Directory to save files (defaults to last part of repo_id)
- `num_files`: Total number of files (auto-detected if not specified)
- `start_index`: Starting index (auto-detected if not specified)
- `max_concurrent_downloads`: Number of parallel downloads (default: 16)
- `max_connection_per_server`: Connections per server (default: 16)
- `min_split_size`: Minimum split size for parallel downloading (default: "1M")

## Generated Files

- `aria2_urls.txt`: Contains download URLs and output paths
- `aria2_download.log`: Detailed download log

## Example Command Generation

To generate aria2 commands without downloading:

```python
from hf_model_downloader import HFModelDownloader

downloader = HFModelDownloader(repo_id="organization/model-name")
cmd = downloader.generate_aria2_command()
print(" ".join(cmd))
```

## Troubleshooting

1. **Pattern Detection Failed**:
   - Check if the repository is public and accessible
   - Verify that files follow one of the supported patterns
   - Try specifying the pattern manually

2. **aria2c not found**:
   - Make sure aria2 is installed
   - Check if aria2c is in your PATH

3. **Download interrupted**:
   - Simply run the script again, it will resume automatically

4. **Connection issues**:
   - Try reducing `max_concurrent_downloads` and `max_connection_per_server`
   - Check your internet connection
   - Verify you have enough disk space

## Contributing

Feel free to submit issues and enhancement requests!

## Using with AI Agents

This downloader is designed to be agent-friendly, making it easy to use with AI assistants like Claude in Cursor.

### Using in Cursor

1. Switch to Composer mode in Cursor
2. Change mode from `normal` to `agent`
3. Include this README in the context:
   ```
   @README
   ```
4. Ask the agent to download a model using natural language:
   ```
   Please download the DeepSeek Coder model for me using the command line.
   ```
   or
   ```
   Can you help me download CodeLlama? I have the downloader installed and want to use the CLI.
   ```

The agent will:
1. Understand your request
2. Determine the correct repository ID
3. Generate and execute the appropriate command
4. Monitor the download process

### Example Agent Interactions

```
User: Can you download the DeepSeek-R1 model for me using the command line? The URL is https://huggingface.co/deepseek-ai/DeepSeek-R1/tree/main

Agent: I'll help you download the DeepSeek-R1 model using the command-line interface.
I'll run the downloader with the appropriate repository ID:

Running command: python3 hf_model_downloader.py --repo-id deepseek-ai/DeepSeek-R1

The downloader will:
- Auto-detect the file pattern and count
- Set up parallel downloads with aria2
- Create the output directory
- Show download progress in the terminal

You can interrupt the download anytime with Ctrl+C and resume it later by running the same command.

User: Can you download it to a specific directory?

Agent: I'll run the downloader with a custom output directory:

Running command: python3 hf_model_downloader.py --repo-id deepseek-ai/DeepSeek-R1 --output-dir ./my-models/deepseek
```

### Command-Line Options

The agent understands these command-line arguments:
```bash
python3 hf_model_downloader.py [options]

Options:
  --repo-id                  HuggingFace repository ID
  --output-dir              Custom output directory
  --pattern                 Custom file pattern (auto-detected by default)
  --num-files              Number of files (auto-detected by default)
  --start-index            Starting index (auto-detected by default)
  --max-concurrent         Maximum concurrent downloads (default: 16)
  --max-connections        Maximum connections per server (default: 16)
  --min-split-size         Minimum split size (default: "1M")
```

### Agent-Friendly Features

- Natural language understanding of command-line options
- Smart defaults requiring minimal configuration
- Clear terminal output
- Progress reporting
- Error handling
- Resumable downloads

### Tips for Agent Interaction

- Specify the model you want to download, with URL
- Mention any special requirements (e.g., output directory)
- Ask for command explanations if needed
- Request download status checks
- Ask for help with any error messages
