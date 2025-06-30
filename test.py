from hf_model_downloader import HFModelDownloader

downloader = HFModelDownloader(
    repo_id="HuggingFaceTB/SmolLM2-135M",
    output_dir="model"
)

downloader.start_download()
