"""Reattach face images to DescriptiveFAS/MCIO_public rows from locally cropped frames.

Pipeline assumption (run beforehand):
    1. utils/split_imgs/{BENCH}_split.py  --src ...  --dst ...
    2. python face_crop.py --base-path ... --dest-path <CROPPED_DIR>

Then for each row in MCIO_public, this script locates the corresponding
local cropped frame using (source, original_dir, spoof) and writes a
reconstructed datasets.Dataset to disk.

Matching strategy: per (source, original_dir, spoof) group there are
exactly 2 caption rows. We pick 2 frames using the same selection rule
(index 4 and index 4 + N//2, capped) and assign them to the rows in
id-sorted order. Pair is stable; intra-pair frame<->caption order may
differ from the original randomized pairing, but both frames come from
the same video and the captions are holistic, so the loss is negligible.

OUTPUT POLICY: local disk only. No Hub push.
"""
import argparse
import json
import os
import sys
from collections import defaultdict

from datasets import Dataset, Features, Image as DatasetsImage, Value, load_dataset
from tqdm import tqdm

ALL_CONFIGS = [
    "2frame_inter_demoMCIO_w_GPT_Caption",
    "2frame_inter_demoMIC_w_GPT_Caption",
    "2frame_inter_demoMOC_w_GPT_Caption",
    "2frame_inter_demoMOI_w_GPT_Caption",
    "2frame_inter_demoOCI_w_GPT_Caption",
    "2frame_reasoning_inter_demoMCIO_w_GPT_Caption",
    "2frame_reasoning_inter_demoMIC_w_GPT_Caption",
    "2frame_reasoning_inter_demoMOC_w_GPT_Caption",
    "2frame_reasoning_inter_demoMOI_w_GPT_Caption",
    "2frame_reasoning_inter_demoOCI_w_GPT_Caption",
]

SOURCE_TO_BENCHMARK = {
    "MSU-MFSD": "MSU-MFSD",
    "CASIA": "CASIA-FASD",
    "IDIAP": "IDIAP-REPLAY",
    "OULU-NPU": "OULU-NPU",
}

LICENSE_WARNING = (
    "[reproduce_images] WARNING: The reconstructed dataset contains face images\n"
    "[reproduce_images] derived from MSU-MFSD / CASIA-FASD / IDIAP-REPLAY / OULU-NPU\n"
    "[reproduce_images] videos. DO NOT redistribute. Use locally only, under the\n"
    "[reproduce_images] original benchmark licenses you accepted.\n"
)

OUTPUT_FEATURES = Features({
    "id": Value("string"),
    "source": Value("string"),
    "original_dir": Value("string"),
    "image": DatasetsImage(),
    "spoof": Value("int32"),
    "attack_type": Value("string"),
    "gpt_caption_holistic": Value("string"),
    "prompt": Value("string"),
})


def local_dir_name(source, original_dir, spoof):
    """Map a MCIO_public (source, original_dir, spoof) to the local crop dir basename.

    IDIAP live original_dir has a 'real_' prefix added during data_standardization;
    the local cropped folder name is the underlying video filename without that prefix.
    """
    if source == "IDIAP" and spoof == 0 and original_dir.startswith("real_"):
        return original_dir[len("real_"):]
    return original_dir


def find_crop_dir(cropped_root, source, original_dir, spoof):
    """Search train/ and test/ for the matching local crop directory."""
    benchmark = SOURCE_TO_BENCHMARK.get(source)
    if benchmark is None:
        return None
    label = "live" if spoof == 0 else "spoof"
    name = local_dir_name(source, original_dir, spoof)
    for phase in ("train", "test"):
        candidate = os.path.join(cropped_root, benchmark, phase, label, name)
        if os.path.isdir(candidate):
            return candidate
    return None


def select_two_frames(crop_filenames):
    """Mirror sampled_dataset_gen: index 4 and (4 + N//2), capped at N-1."""
    n = len(crop_filenames)
    if n <= 4:
        return None
    idx1 = 4
    idx2 = 4 + (n // 2)
    if idx2 >= n:
        idx2 = n - 1
    if idx2 <= idx1:
        idx2 = idx1 + 1 if idx1 + 1 < n else idx1
    return [crop_filenames[idx1], crop_filenames[idx2]]


def reproduce_config(cropped_root, config, output_root, hf_token=None):
    print(f"\n=== Loading config: {config} ===", file=sys.stderr)
    ds_pub = load_dataset(
        "DescriptiveFAS/MCIO_public", config, split="train", token=hf_token
    )

    groups = defaultdict(list)
    for idx in range(len(ds_pub)):
        row = ds_pub[idx]
        key = (row["source"], row["original_dir"], row["spoof"])
        groups[key].append(idx)

    image_by_idx = {}
    missing = []
    insufficient = []

    for key in tqdm(sorted(groups.keys()), desc=config):
        source, original_dir, spoof = key
        idxs = groups[key]

        crop_dir = find_crop_dir(cropped_root, source, original_dir, spoof)
        if crop_dir is None:
            missing.append({"source": source, "original_dir": original_dir, "spoof": spoof})
            continue

        crops = sorted(
            f for f in os.listdir(crop_dir)
            if f.startswith("crop") and f.endswith(".jpg")
        )
        selected = select_two_frames(crops)
        if selected is None:
            insufficient.append({
                "source": source,
                "original_dir": original_dir,
                "spoof": spoof,
                "n_crops": len(crops),
                "dir": crop_dir,
            })
            continue

        idxs_sorted = sorted(idxs, key=lambda i: ds_pub[i]["id"])
        for ds_idx, crop_name in zip(idxs_sorted, selected):
            with open(os.path.join(crop_dir, crop_name), "rb") as fh:
                image_by_idx[ds_idx] = fh.read()

    out = {col: [] for col in OUTPUT_FEATURES.keys()}
    skipped_idxs = []
    for idx in range(len(ds_pub)):
        if idx not in image_by_idx:
            skipped_idxs.append(idx)
            continue
        row = ds_pub[idx]
        out["id"].append(row["id"])
        out["source"].append(row["source"])
        out["original_dir"].append(row["original_dir"])
        out["image"].append({"bytes": image_by_idx[idx], "path": None})
        out["spoof"].append(row["spoof"])
        out["attack_type"].append(row["attack_type"])
        out["gpt_caption_holistic"].append(row["gpt_caption_holistic"])
        out["prompt"].append(row["prompt"])

    new_ds = Dataset.from_dict(out, features=OUTPUT_FEATURES)
    save_path = os.path.join(output_root, config)
    new_ds.save_to_disk(save_path)

    report = {
        "config": config,
        "input_rows": len(ds_pub),
        "output_rows": len(new_ds),
        "skipped_rows": len(skipped_idxs),
        "missing_groups": missing,
        "insufficient_groups": insufficient,
    }
    report_path = os.path.join(output_root, f"{config}.report.json")
    with open(report_path, "w") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    print(
        f"[{config}] saved {len(new_ds)}/{len(ds_pub)} rows -> {save_path}",
        file=sys.stderr,
    )
    if missing:
        print(f"[{config}] missing local crops for {len(missing)} groups", file=sys.stderr)
    if insufficient:
        print(f"[{config}] insufficient (<=4) crops for {len(insufficient)} groups", file=sys.stderr)
    print(f"[{config}] report: {report_path}", file=sys.stderr)

    return report


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--cropped-dir", required=True,
        help="Root output of face_crop.py (contains {benchmark}/{train,test}/{live,spoof}/...).",
    )
    parser.add_argument(
        "--config", action="append", default=None,
        help="MCIO_public config name. Repeat to process multiple. Omit to process all 10.",
    )
    parser.add_argument(
        "--output", required=True,
        help="Directory to write reconstructed datasets.",
    )
    parser.add_argument(
        "--hf-token", default=os.environ.get("HF_TOKEN"),
        help="Optional HF token (env HF_TOKEN). MCIO_public is public; usually unnecessary.",
    )
    return parser.parse_args()


def main():
    sys.stderr.write(LICENSE_WARNING)
    sys.stderr.flush()

    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    configs = args.config if args.config else ALL_CONFIGS
    for cfg in configs:
        reproduce_config(args.cropped_dir, cfg, args.output, hf_token=args.hf_token)


if __name__ == "__main__":
    main()
