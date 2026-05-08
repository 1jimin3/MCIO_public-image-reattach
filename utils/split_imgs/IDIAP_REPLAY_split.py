"""Split IDIAP-REPLAY videos into {train,test}/{live,spoof}/.

Expected raw layout:
    src_path/{train,test}/
      real/*.{avi,mov,mp4}
      attack/
        fixed/*.{avi,mov,mp4}
        hand/*.{avi,mov,mp4}

Output layout:
    dst_path/{train,test}/live/<filename>
    dst_path/{train,test}/spoof/{fixed_,hand_}<filename>
"""
import argparse
import os
import shutil


def classify(src_root, dst_root):
    for phase in ['train', 'test']:
        phase_path = os.path.join(src_root, phase)

        for label in ['live', 'spoof']:
            dst_path = os.path.join(dst_root, phase, label)
            os.makedirs(dst_path, exist_ok=True)

            if label == 'live':
                src_path = os.path.join(phase_path, 'real')
                for file in os.listdir(src_path):
                    shutil.copy(os.path.join(src_path, file), os.path.join(dst_path, file))
            else:
                fixed_path = os.path.join(phase_path, 'attack', 'fixed')
                hand_path = os.path.join(phase_path, 'attack', 'hand')

                for file in os.listdir(fixed_path):
                    shutil.copy(os.path.join(fixed_path, file), os.path.join(dst_path, f'fixed_{file}'))

                for file in os.listdir(hand_path):
                    shutil.copy(os.path.join(hand_path, file), os.path.join(dst_path, f'hand_{file}'))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--src', required=True, help='Path to IDIAP-REPLAY raw root with train/test subdirs.')
    parser.add_argument('--dst', required=True, help='Output path; will create {train,test}/{live,spoof}/.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    classify(args.src, args.dst)
