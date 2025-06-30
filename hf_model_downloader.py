import os
import json
import subprocess
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
import shutil
import requests
from collections import defaultdict
import argparse

class HFModelDownloader:
    # Weight file extensions
    WEIGHT_EXTENSIONS = {'.safetensors', '.bin', '.pt', '.pth'}
    
    # Files to ignore
    IGNORE_FILES = {
        'README.md', 'readme.md', 'LICENSE', 'license', 'LICENSE.txt', 'license.txt',
        '.gitattributes', '.gitignore', 'flax_model.msgpack', 'rust_model.ot', 'tf_model.h5'
    }

    def __init__(self, 
                 repo_id: str,
                 pattern: Optional[str] = None,
                 output_dir: Optional[str] = None,
                 num_files: Optional[int] = None,
                 start_index: Optional[int] = None,
                 max_concurrent_downloads: int = 16,
                 max_connection_per_server: int = 16,
                 min_split_size: str = "1M"):
        """
        Initialize the HuggingFace model downloader.
        
        Args:
            repo_id: HuggingFace repository ID (e.g., "deepseek-ai/DeepSeek-R1")
            pattern: Filename pattern with {i} for index and {total} for total files
                    If None, will try to auto-detect
            output_dir: Directory to save files (defaults to last part of repo_id)
            num_files: Total number of files to download
                      If None, will try to auto-detect
            start_index: Starting index for file numbering
                        If None, will try to auto-detect
            max_concurrent_downloads: Number of parallel downloads
            max_connection_per_server: Connections per server
            min_split_size: Minimum split size for parallel downloading
        """
        self.repo_id = repo_id
        self.output_dir = Path(output_dir or repo_id.split('/')[-1].lower())
        self.max_concurrent_downloads = max_concurrent_downloads
        self.max_connection_per_server = max_connection_per_server
        self.min_split_size = min_split_size
        
        # Get repository contents first
        self.repo_files = self._get_repo_contents()
        if not self.repo_files:
            raise ValueError(f"Could not fetch repository contents for {repo_id}")
        
        # Auto-detect pattern and file count if not provided
        if pattern is None or num_files is None or start_index is None:
            detected_pattern, detected_num_files, detected_start = self._detect_file_pattern()
            self.pattern = pattern or detected_pattern
            self.num_files = num_files or detected_num_files
            self.start_index = start_index or detected_start
        else:
            self.pattern = pattern
            self.num_files = num_files
            self.start_index = start_index
            
        if not self.pattern or not self.num_files:
            raise ValueError(f"Could not detect file pattern in repository {repo_id}")
            
        print(f"Using pattern: {self.pattern}")
        print(f"Number of files: {self.num_files}")
        print(f"Start index: {self.start_index}")

    def _get_repo_contents(self) -> Optional[List[Dict]]:
        """Get repository file listing."""
        api_url = f"https://huggingface.co/api/models/{self.repo_id}/tree/main"
        try:
            response = requests.get(api_url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching repository contents: {str(e)}")
            return None

    def _is_weight_file(self, filename: str) -> bool:
        """Check if a file is a model weight file."""
        return any(filename.endswith(ext) for ext in self.WEIGHT_EXTENSIONS)

    def _detect_file_pattern(self) -> Tuple[Optional[str], Optional[int], Optional[int]]:
        """Detect file pattern and count from repository contents."""
        print(f"Detecting file pattern in {self.repo_id}...")
        
        # First, check for single weight files
        single_weight_files = []
        for file in self.repo_files:
            name = file.get('path', '')
            if self._is_weight_file(name):
                # Check if it's a single file (not matching any shard pattern)
                if not re.match(r'.*-\d+.*\.(safetensors|bin|pt|pth)$', name) and \
                   not re.match(r'.*\.(safetensors|bin|pt|pth)\.\d+$', name):
                    single_weight_files.append(name)
        
        # If we found single weight files, use them
        if single_weight_files:
            if len(single_weight_files) == 1:
                filename = single_weight_files[0]
                print(f"Found single weight file: {filename}")
                # For single files, we'll use a special pattern that just returns the filename
                return filename, 1, 0
            else:
                print(f"Found multiple single weight files: {single_weight_files}")
                # Use the first one
                filename = single_weight_files[0]
                return filename, 1, 0
        
        # Group files by their pattern (for sharded models)
        patterns = defaultdict(list)
        for file in self.repo_files:
            name = file.get('path', '')
            
            # Skip if not a weight file
            if not self._is_weight_file(name):
                continue
                
            # Try different common patterns
            # Pattern 1: model-00001-of-00163.safetensors
            match = re.match(r'(.*?)-(\d+)-of-(\d+)\.(safetensors|bin|pt|pth)$', name)
            if match:
                prefix, idx, total, ext = match.groups()
                pattern = f"{prefix}-{{i:0{len(idx)}d}}-of-{total}.{ext}"
                patterns[pattern].append(int(idx))
                continue
                
            # Pattern 2: pytorch_model-00001.bin
            match = re.match(r'(.*?)-(\d+)\.(safetensors|bin|pt|pth)$', name)
            if match:
                prefix, idx, ext = match.groups()
                pattern = f"{prefix}-{{i:0{len(idx)}d}}.{ext}"
                patterns[pattern].append(int(idx))
                continue
                
            # Pattern 3: model.safetensors.00001
            match = re.match(r'(.*?)\.(safetensors|bin|pt|pth)\.(\d+)$', name)
            if match:
                prefix, ext, idx = match.groups()
                pattern = f"{prefix}.{ext}.{{i:0{len(idx)}d}}"
                patterns[pattern].append(int(idx))

        # Find the pattern with the most files
        if not patterns:
            return None, None, None
            
        best_pattern = max(patterns.items(), key=lambda x: len(x[1]))
        pattern = best_pattern[0]
        indices = sorted(best_pattern[1])
        
        if not indices:
            return None, None, None
            
        return pattern, len(indices), indices[0]

    def _get_auxiliary_files(self) -> List[str]:
        """Get list of auxiliary files (configs, tokenizer files, etc.)."""
        auxiliary_files = []
        for file in self.repo_files:
            name = file.get('path', '')
            
            # Skip weight files and ignored files
            if self._is_weight_file(name) or name in self.IGNORE_FILES:
                continue
                
            # Skip directories
            if file.get('type', '') == 'directory':
                continue
                
            # Skip large binary files that aren't weights
            size = file.get('size', 0)
            if size > 10 * 1024 * 1024:  # Skip files > 10MB
                continue
                
            auxiliary_files.append(name)
            
        return auxiliary_files

    def _download_auxiliary_files(self, urls_file: str) -> Tuple[int, int]:
        """Download auxiliary model files."""
        existing = 0
        total = 0
        
        auxiliary_files = self._get_auxiliary_files()
        if auxiliary_files:
            print("\nFound auxiliary files:")
            for name in auxiliary_files:
                print(f"  - {name}")
        
        with open(urls_file, 'a') as f:
            for name in auxiliary_files:
                output_path = self.output_dir / name
                total += 1
                
                if output_path.exists():
                    existing += 1
                    print(f"Skipping {name} - already exists")
                    continue
                
                url = self.get_file_url(name)
                if url:
                    print(f"Adding auxiliary file: {name}")
                    f.write(f"{url}\n")
                    f.write(f"  out={output_path}\n")
        
        return existing, total

    def get_file_url(self, filename: str) -> Optional[str]:
        """Get the direct download URL for a file from HuggingFace."""
        url = f"https://huggingface.co/{self.repo_id}/resolve/main/{filename}"
        print(f"Getting redirect URL for: {url}")
        
        cmd = ["curl", "-I", "-L", url]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        urls = re.findall(r'location: (.*?)$', result.stdout, re.MULTILINE)
        if urls:
            return urls[-1].strip()
        return None

    def generate_filenames(self) -> List[str]:
        """Generate list of filenames based on the pattern."""
        # If pattern is a single filename (not a format string), return it directly
        if not self.pattern or '{' not in self.pattern:
            return [self.pattern] if self.pattern else []
        
        # For sharded models, use the format pattern
        return [self.pattern.format(i=i, total=self.num_files) 
                for i in range(self.start_index, self.start_index + self.num_files)]

    def generate_aria2_input(self, urls_file: str = "aria2_urls.txt") -> Tuple[int, int]:
        """Generate aria2 input file with URLs and output paths."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # First, handle weight files
        existing_files = 0
        total_files = 0
        
        with open(urls_file, 'w') as f:
            for filename in self.generate_filenames():
                output_path = self.output_dir / filename
                total_files += 1
                
                if output_path.exists():
                    existing_files += 1
                    print(f"Skipping {filename} - already exists")
                    continue
                
                url = self.get_file_url(filename)
                if url:
                    f.write(f"{url}\n")
                    f.write(f"  out={output_path}\n")
        
        # Then, add auxiliary files
        existing_aux, total_aux = self._download_auxiliary_files(urls_file)
        
        return (existing_files + existing_aux, total_files + total_aux)

    def generate_aria2_command(self, 
                             urls_file: str = "aria2_urls.txt",
                             log_file: str = "aria2_download.log") -> List[str]:
        """Generate aria2c command with optimal parameters."""
        return [
            "aria2c",
            "--input-file", urls_file,
            "--max-concurrent-downloads", str(self.max_concurrent_downloads),
            "--max-connection-per-server", str(self.max_connection_per_server),
            "--min-split-size", self.min_split_size,
            "--auto-file-renaming=false",
            "--continue=true",
            "--log", log_file,
            "--log-level=notice",
            "--console-log-level=notice",
            "--summary-interval=1",
            "--show-console-readout=true",
            "--download-result=full"
        ]

    def check_aria2_installed(self) -> bool:
        """Check if aria2 is installed."""
        return shutil.which("aria2c") is not None

    def start_download(self) -> bool:
        """Start the download process using aria2."""
        if not self.check_aria2_installed():
            print("Error: aria2c is not installed. Please install it first.")
            return False
            
        print(f"Downloading {self.repo_id}...")
        print(f"Output directory: {self.output_dir}")
        existing_files, total_files = self.generate_aria2_input()
        
        if existing_files == total_files:
            print("All files already downloaded.")
            return True
            
        print(f"\nStarting download of {total_files - existing_files} files...")
        cmd = self.generate_aria2_command()
        print(f"Running command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd)
            return result.returncode == 0
        except KeyboardInterrupt:
            print("\nDownload interrupted. You can resume by running the script again.")
            return False
        except Exception as e:
            print(f"Error during download: {str(e)}")
            return False

def main():
    parser = argparse.ArgumentParser(description="Download large model files from HuggingFace using aria2")
    parser.add_argument("--repo-id", required=True, help="HuggingFace repository ID (e.g., deepseek-ai/DeepSeek-R1)")
    parser.add_argument("--output-dir", help="Custom output directory")
    parser.add_argument("--pattern", help="Custom file pattern (auto-detected by default)")
    parser.add_argument("--num-files", type=int, help="Number of files (auto-detected by default)")
    parser.add_argument("--start-index", type=int, help="Starting index (auto-detected by default)")
    parser.add_argument("--max-concurrent", type=int, default=16, help="Maximum concurrent downloads (default: 16)")
    parser.add_argument("--max-connections", type=int, default=16, help="Maximum connections per server (default: 16)")
    parser.add_argument("--min-split-size", default="1M", help="Minimum split size (default: 1M)")
    
    args = parser.parse_args()
    
    # Create downloader with command line arguments
    downloader = HFModelDownloader(
        repo_id=args.repo_id,
        pattern=args.pattern,
        output_dir=args.output_dir,
        num_files=args.num_files,
        start_index=args.start_index,
        max_concurrent_downloads=args.max_concurrent,
        max_connection_per_server=args.max_connections,
        min_split_size=args.min_split_size
    )
    
    success = downloader.start_download()
    if success:
        print("\nDownload completed successfully!")
    else:
        print("\nDownload failed or was interrupted.")
        exit(1)

if __name__ == "__main__":
    main() 