"""Split CASIA-FASD videos into {train,test}/{live,spoof}/.

Expected raw layout:
    src_path/{train,test}/{user_id}/{HR_N|N}_*.avi
where HR_1 / 1 / 2 prefixes denote live, others denote spoof.

Output layout:
    dst_path/{train,test}/{live,spoof}/{user_id}{prefix}{file}
"""
import argparse
import os
import shutil


def split(src_root, dst_root):
    for phase in ['train', 'test']:
        live_path = os.path.join(dst_root, phase, 'live')
        spoof_path = os.path.join(dst_root, phase, 'spoof')
        os.makedirs(live_path, exist_ok=True)
        os.makedirs(spoof_path, exist_ok=True)

        phase_src = os.path.join(src_root, phase)
        for folder_id in os.listdir(phase_src):
            if 'live' in folder_id or 'spoof' in folder_id:
                continue
            folder_id_path = os.path.join(phase_src, folder_id)
            if not os.path.isdir(folder_id_path):
                continue

            for file in os.listdir(folder_id_path):
                src_file_path = os.path.join(folder_id_path, file)
                file_prefix = '_' if 'HR' in file else '_NM_'

                if file.startswith('HR_1') or file.startswith('1') or file.startswith('2'):
                    dst_file_path = os.path.join(live_path, f"{folder_id}{file_prefix}{file}")
                else:
                    dst_file_path = os.path.join(spoof_path, f"{folder_id}{file_prefix}{file}")

                shutil.copy(src_file_path, dst_file_path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--src', required=True, help='Path to CASIA-FASD raw root with train/ and test/ subdirs.')
    parser.add_argument('--dst', required=True, help='Output path; will create {train,test}/{live,spoof}/.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    split(args.src, args.dst)
