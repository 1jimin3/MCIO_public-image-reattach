"""Compare a reconstructed dataset against MCIO_public reference.

Checks:
  - row count parity
  - id-set parity (which ids are missing)
  - image column non-empty for every output row
  - (source, original_dir, spoof) coverage

Outputs human summary to stdout and a JSON report to --report path.
"""
import argparse
import json
import os
import sys
from collections import Counter

from datasets import load_dataset, load_from_disk


def parse_reference(ref):
    """Accept 'repo:config' or a local path."""
    if ":" in ref and not os.path.isdir(ref.split(":", 1)[0]):
        repo, config = ref.split(":", 1)
        return ("hub", repo, config)
    return ("local", ref, None)


def load_reference(ref, hf_token=None):
    kind, a, b = parse_reference(ref)
    if kind == "hub":
        return load_dataset(a, b, split="train", token=hf_token)
    return load_from_disk(a)


def check_image_nonempty(ds):
    empty = 0
    for row in ds:
        img = row.get("image")
        if img is None:
            empty += 1
            continue
        # PIL Image when decoded by datasets; fall back to dict shape
        if hasattr(img, "size"):
            if img.size == (0, 0):
                empty += 1
        elif isinstance(img, dict):
            if not img.get("bytes") and not img.get("path"):
                empty += 1
    return empty


def verify(reproduced_path, reference, hf_token=None):
    repro = load_from_disk(reproduced_path)
    ref = load_reference(reference, hf_token=hf_token)

    repro_ids = {r["id"] for r in repro}
    ref_ids = {r["id"] for r in ref}

    missing_in_repro = sorted(ref_ids - repro_ids)
    extra_in_repro = sorted(repro_ids - ref_ids)
    empty_images = check_image_nonempty(repro)

    repro_keys = Counter(
        (r["source"], r["original_dir"], r["spoof"]) for r in repro
    )
    ref_keys = Counter(
        (r["source"], r["original_dir"], r["spoof"]) for r in ref
    )
    missing_keys = sorted(set(ref_keys) - set(repro_keys))

    return {
        "reproduced_rows": len(repro),
        "reference_rows": len(ref),
        "row_parity": len(repro) == len(ref),
        "missing_id_count": len(missing_in_repro),
        "extra_id_count": len(extra_in_repro),
        "empty_image_count": empty_images,
        "missing_group_count": len(missing_keys),
        "missing_ids_sample": missing_in_repro[:20],
        "extra_ids_sample": extra_in_repro[:20],
        "missing_groups_sample": [list(k) for k in missing_keys[:20]],
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--reproduced", required=True,
                        help="Path saved by reproduce_images.py for one config.")
    parser.add_argument("--reference", required=True,
                        help="Either 'DescriptiveFAS/MCIO_public:<config>' or a local dataset path.")
    parser.add_argument("--report", default=None,
                        help="Optional path to write JSON report.")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    return parser.parse_args()


def main():
    args = parse_args()
    summary = verify(args.reproduced, args.reference, hf_token=args.hf_token)

    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.report:
        with open(args.report, "w") as fh:
            json.dump(summary, fh, indent=2, ensure_ascii=False)
        print(f"\nReport written to {args.report}", file=sys.stderr)

    ok = (
        summary["row_parity"]
        and summary["empty_image_count"] == 0
        and summary["missing_group_count"] == 0
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
