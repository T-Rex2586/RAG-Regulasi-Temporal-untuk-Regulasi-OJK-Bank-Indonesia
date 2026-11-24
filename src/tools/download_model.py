import argparse
from pathlib import Path
from huggingface_hub import snapshot_download


def download_qwen_model(
    repo_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
    output_dir: str = "models/Qwen2.5-1.5B-Instruct",
):
    print(f"=========================================================================")
    print(f"            DOWNLOADING BASE MODEL: {repo_id}                            ")
    print(f"=========================================================================")
    print(f"Target local directory: {output_dir}")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    downloaded_dir = snapshot_download(
        repo_id=repo_id,
        local_dir=str(out_path),
        local_dir_use_symlinks=False,
        resume_download=True,
    )

    print(f"\nModel successfully downloaded to: {downloaded_dir}")
    print("Files in model directory:")
    total_size = 0
    for f in sorted(out_path.iterdir()):
        if f.is_file():
            sz = f.stat().st_size / (1024 * 1024)
            total_size += sz
            print(f"  - {f.name} ({sz:.2f} MB)")
    print(f"Total model size: {total_size:.2f} MB ({total_size / 1024:.2f} GB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Qwen base model from Hugging Face")
    parser.add_argument("--repo_id", default="Qwen/Qwen2.5-1.5B-Instruct", help="Hugging Face repo ID")
    parser.add_argument("--output_dir", default="models/Qwen2.5-1.5B-Instruct", help="Local directory")
    args = parser.parse_args()

    download_qwen_model(args.repo_id, args.output_dir)
