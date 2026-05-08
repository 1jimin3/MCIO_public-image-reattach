"""Split OULU-NPU videos into {train,test}/{live,spoof}/.

Expected raw layout:
    src_path/
      Train_files/*_N.avi
      Test_files/*_N.avi
where N=1 means live, others mean spoof.

Output layout:
    dst_path/{train,test}/{live,spoof}/<filename>.avi
"""
import argparse
import os
import shutil


def split(src_root, dst_root):
    for phase in ['train', 'test']:
        sub_dir_path = os.path.join(src_root, 'Train_files' if phase == 'train' else 'Test_files')
        videos = [f for f in os.listdir(sub_dir_path) if f.endswith('.avi')]

        for video in videos:
            file_number = int(video.split('_')[-1].split('.')[0])
            label = 'live' if file_number == 1 else 'spoof'

            video_path = os.path.join(sub_dir_path, video)
            dest_dir = os.path.join(dst_root, phase, label)
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy(video_path, os.path.join(dest_dir, video))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--src', required=True, help='Path to OULU-NPU raw root with Train_files/ and Test_files/.')
    parser.add_argument('--dst', required=True, help='Output path; will create {train,test}/{live,spoof}/.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    split(args.src, args.dst)
