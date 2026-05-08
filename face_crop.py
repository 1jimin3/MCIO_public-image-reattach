"""Detect and crop faces from benchmark videos with MTCNN.

Expected input layout (output of utils/split_imgs/*_split.py):
    base_path/{benchmark}(sorted)/{train,test}/{live,spoof}/*.{avi,mov,mp4}

Output layout:
    dest_path/{benchmark}/{train,test}/{live,spoof}/{video_name}/crop_NNN.jpg
"""
import argparse
import glob
import os
import random

import imageio
import numpy as np
import torch
from facenet_pytorch import MTCNN
from PIL import Image
from tqdm import tqdm

BENCHMARKS = ['CASIA-FASD', 'IDIAP-REPLAY', 'MSU-MFSD', 'OULU-NPU']


def crop_faces_mtcnn(base_path, dest_path, benchmarks=BENCHMARKS, min_face_size=70):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Running on device: {device}')
    detector = MTCNN(keep_all=False, min_face_size=min_face_size, device=device)

    for bench in benchmarks:
        for phase in ['train', 'test']:
            for label in ['live', 'spoof']:
                src_dir = os.path.join(base_path, f'{bench}(sorted)', phase, label)
                dest_dir = os.path.join(dest_path, bench, phase, label)
                os.makedirs(dest_dir, exist_ok=True)

                files = sorted(
                    glob.glob(os.path.join(src_dir, '*.avi'))
                    + glob.glob(os.path.join(src_dir, '*.mov'))
                    + glob.glob(os.path.join(src_dir, '*.mp4'))
                )

                for file in tqdm(files, desc=f'{bench}[{phase} - {label}]'):
                    reader = imageio.get_reader(file, 'ffmpeg')

                    file_name = os.path.splitext(os.path.basename(file))[0]
                    os.makedirs(os.path.join(dest_dir, file_name), exist_ok=True)

                    frame_idx = 0
                    frame_checking = False
                    for frame_num, frame in enumerate(reader):
                        if not ((frame_num % 5 == 0 and frame_idx % 5 == 0) or frame_checking):
                            continue

                        frame_rgb = Image.fromarray(frame).convert('RGB')
                        boxes, _ = detector.detect(frame_rgb)

                        if boxes is None:
                            frame_checking = True
                            continue

                        for box in boxes:
                            x1, y1, x2, y2 = (int(v) for v in box)
                            x1 = max(0, x1)
                            y1 = max(0, y1)
                            x2 = min(frame_rgb.width, x2)
                            y2 = min(frame_rgb.height, y2)

                            cropped_img = frame_rgb.crop((x1, y1, x2, y2))
                            dest_img_path = os.path.join(
                                dest_dir, file_name, f'crop_{str(frame_idx).zfill(3)}.jpg'
                            )
                            cropped_img.save(dest_img_path)
                            frame_checking = False
                            frame_idx += 5

                    reader.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--base-path', required=True,
                        help='Root containing {benchmark}(sorted)/ produced by split scripts.')
    parser.add_argument('--dest-path', required=True,
                        help='Destination root for cropped face images.')
    parser.add_argument('--benchmarks', nargs='+', default=BENCHMARKS,
                        help=f'Subset of benchmarks to process. Default: {BENCHMARKS}')
    parser.add_argument('--min-face-size', type=int, default=70,
                        help='MTCNN min_face_size (default: 70).')
    parser.add_argument('--seed', type=int, default=2024,
                        help='Seed for torch/numpy/random to improve reproducibility.')
    return parser.parse_args()


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    crop_faces_mtcnn(
        base_path=args.base_path,
        dest_path=args.dest_path,
        benchmarks=args.benchmarks,
        min_face_size=args.min_face_size,
    )


if __name__ == '__main__':
    main()
