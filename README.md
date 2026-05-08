# MCIO_public Image Reattachment Pipeline

Companion code for [DescriptiveFAS/MCIO_public](https://huggingface.co/datasets/DescriptiveFAS/MCIO_public).

The published dataset omits face images for privacy/license reasons and ships only captions, labels, and metadata. This repository lets researchers who already have the raw benchmark videos (under their own license) reconstruct a usable training/evaluation dataset on their own machine.

> ⚠️ **The reconstructed dataset MUST NOT be redistributed.** The original benchmark licenses (MSU-MFSD / CASIA-FASD / IDIAP-REPLAY / OULU-NPU) prohibit sharing of derived works that include the source images. Use only on your local machine, for research consistent with the licenses you accepted.

---

## Pipeline

```
raw videos
    │
    ▼
utils/split_imgs/<BENCH>_split.py    # organize {train,test}/{live,spoof}/
    │
    ▼
face_crop.py                          # MTCNN-cropped faces every 5 frames
    │
    ▼
reproduce_images.py                   # attach images to MCIO_public rows
    │
    ▼
local datasets.Dataset (caption + image)
```

## Requirements

- Python ≥ 3.10
- See [requirements.txt](requirements.txt). GPU recommended for `face_crop.py`.

```bash
pip install -r requirements.txt
```

## Required raw benchmark layouts

You must obtain the raw videos directly from each benchmark's official source under their license. Below is the layout each split script expects.

### MSU-MFSD
```
MSU-MFSD/
├── information/
│   ├── train_sub_list.txt   # one client ID per line
│   └── test_sub_list.txt
├── real/                    # client0{ID}_*.{avi,mov,mp4}
└── attack/                  # client0{ID}_*.{avi,mov,mp4}
```

### CASIA-FASD
```
CASIA-FASD/
└── {train,test}/
    └── {user_id}/
        ├── HR_1_*.avi   # live  (HR_1, 1, or 2 prefix)
        ├── HR_2_*.avi   # spoof
        └── ...
```

### IDIAP-REPLAY (Replay-Attack)
```
IDIAP-REPLAY/
└── {train,test}/
    ├── real/*.{avi,mov,mp4}
    └── attack/
        ├── fixed/*.{avi,mov,mp4}
        └── hand/*.{avi,mov,mp4}
```

### OULU-NPU
```
OULU-NPU/
├── Train_files/*_N.avi    # N=1: live, otherwise: spoof
└── Test_files/*_N.avi
```

## Step 1 — Sort videos into live/spoof per phase

Run the split script for each benchmark you have. Output ends up under `<dst>/{train,test}/{live,spoof}/<filename>`. For MTCNN cropping in step 2 we need the parent directory to be `<benchmark>(sorted)`.

```bash
# Pick a working directory; example uses /tmp/MCIO/.
SRC_ROOT=/path/to/raw
SORTED=/tmp/MCIO/sorted

python utils/split_imgs/MSU_MFSD_split.py     --src "$SRC_ROOT/MSU-MFSD"     --dst "$SORTED/MSU-MFSD(sorted)"
python utils/split_imgs/CASIA_FASD_split.py   --src "$SRC_ROOT/CASIA-FASD"   --dst "$SORTED/CASIA-FASD(sorted)"
python utils/split_imgs/IDIAP_REPLAY_split.py --src "$SRC_ROOT/IDIAP-REPLAY" --dst "$SORTED/IDIAP-REPLAY(sorted)"
python utils/split_imgs/OULU_NPU_split.py     --src "$SRC_ROOT/OULU-NPU"     --dst "$SORTED/OULU-NPU(sorted)"
```

## Step 2 — Crop faces with MTCNN every 5 frames

```bash
SORTED=/tmp/MCIO/sorted        # output of Step 1
CROPPED=/tmp/MCIO/cropped

python face_crop.py --base-path "$SORTED" --dest-path "$CROPPED"
```

Output:
```
$CROPPED/<benchmark>/<train|test>/<live|spoof>/<video_name>/crop_NNN.jpg
```

GPU is recommended; CPU works but is slow. The script enables `cudnn.deterministic` and seeds torch/numpy/random to maximize reproducibility on a single machine.

## Step 3 — Reattach images to MCIO_public

```bash
CROPPED=/tmp/MCIO/cropped      # output of Step 2
OUT=/tmp/MCIO/reproduced

# Process all 10 configs (5 cross-domain × {non-reasoning, reasoning})
python reproduce_images.py --cropped-dir "$CROPPED" --output "$OUT"

# Or just one config:
python reproduce_images.py \
    --cropped-dir "$CROPPED" \
    --config 2frame_inter_demoMCIO_w_GPT_Caption \
    --output "$OUT"
```

The output is a `datasets.Dataset` directory per config under `$OUT/<config>/`, plus a `<config>.report.json` that lists any unmatched groups.

Load it from your training code:
```python
from datasets import load_from_disk
ds = load_from_disk("/tmp/MCIO/reproduced/2frame_inter_demoMCIO_w_GPT_Caption")
```

## Step 4 — Verify

```bash
OUT=/tmp/MCIO/reproduced       # output of Step 3

python verify_reproduction.py \
    --reproduced "$OUT/2frame_inter_demoMCIO_w_GPT_Caption" \
    --reference "DescriptiveFAS/MCIO_public:2frame_inter_demoMCIO_w_GPT_Caption" \
    --report verify.json
```

Returns exit 0 only if row counts match, no images are empty, and every (source, original_dir, spoof) group is covered.

## Matching strategy and known limits

- **Match key**: `(source, original_dir, spoof)`. Each group has exactly 2 caption rows (the original pipeline sampled 2 frames per video).
- **Frame selection**: sorted crop filenames, indices 4 and `4 + N//2` (capped at `N-1`). Same rule the published dataset used.
- **Pair assignment**: within a group, two selected frames are assigned to the two caption rows in id-sorted order.
- **What this guarantees**: every (source, original_dir, spoof) group is matched, both images come from the exact source video.
- **What this does not guarantee**: id-level identity with the original publication. The original release used `random.sample` to pair frames with captions inside each group, so a particular `id` may end up with the *other* of the two source-video frames. Both frames are from the same person/scene seconds apart and the captions are holistic, so downstream training behavior is materially equivalent.
- **Why we did not attempt id-level reproduction**: MTCNN detection is mildly hardware/version dependent. A single missed detection in one video shifts the global enumeration counter for every subsequent video, so id strings diverge across machines. Pair-level matching is robust to this because `original_dir` is derived directly from the raw video filename.

## Layout

```
.
├── face_crop.py
├── reproduce_images.py
├── verify_reproduction.py
├── utils/
│   └── split_imgs/
│       ├── CASIA_FASD_split.py
│       ├── IDIAP_REPLAY_split.py
│       ├── MSU_MFSD_split.py
│       └── OULU_NPU_split.py
├── requirements.txt
├── LICENSE
└── README.md
```

## Citation

If you use the captions or this code, please cite the MCIO_public dataset and the corresponding paper.

## License

Code in this repository is released under the MIT License (see [LICENSE](LICENSE)). The MCIO_public captions and the underlying benchmark videos are governed by their respective licenses; this repository's license does not extend to them.
