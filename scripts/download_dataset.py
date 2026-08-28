#!/usr/bin/env python3

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="DhiaNeifar/Akkurad_Drone_Detection",
    repo_type="dataset",
    local_dir="datasets/drone_merged_package",
    allow_patterns=[
        "shards/*.tar",
        "shard_manifest.json",
        "dataset.yaml",
        "manifest.csv",
        "audit.json",
        "near_duplicate_candidates.csv",
        "annotation_preview.jpg",
        "dataset_card.md",
        "README.md",
    ],
)

print("Download complete.")