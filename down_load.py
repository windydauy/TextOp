# import argparse
# from huggingface_hub import snapshot_download


# def main():
#     parser = argparse.ArgumentParser(description="Download Yochish/TextOp-Data from Hugging Face")
#     parser.add_argument(
#         "--repo_id",
#         type=str,
#         default="Yochish/TextOp-Data",
#         help="Hugging Face dataset repo id"
#     )
#     parser.add_argument(
#         "--local_dir",
#         type=str,
#         default="./TextOp-Data",
#         help="Local directory to save files"
#     )
#     parser.add_argument(
#         "--subdir",
#         type=str,
#         default=None,
#         help="Optional subdirectory to download only, e.g. TextOpTracker / TextOpDeploy / TextOpRobotMDAR"
#     )
#     parser.add_argument(
#         "--revision",
#         type=str,
#         default="main",
#         help="Repo revision/branch"
#     )
#     parser.add_argument(
#         "--max_workers",
#         type=int,
#         default=8,
#         help="Number of concurrent download workers"
#     )

#     args = parser.parse_args()

#     allow_patterns = None
#     if args.subdir:
#         allow_patterns = [f"{args.subdir}/**"]

#     path = snapshot_download(
#         repo_id=args.repo_id,
#         repo_type="dataset",
#         revision=args.revision,
#         local_dir=args.local_dir,
#         allow_patterns=allow_patterns,
#         max_workers=args.max_workers,
#         resume_download=True,
#     )

#     print(f"Download finished. Files saved under: {path}")


# if __name__ == "__main__":
#     main()
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Yochish/TextOp-Data",
    repo_type="dataset",
    local_dir="./TextOpTracker_partial",
    allow_patterns=[
        "TextOpTracker/logs/**",
        "TextOpTracker/source/**",
    ],
    resume_download=True,
    max_workers=8,
)

print("Download finished.")