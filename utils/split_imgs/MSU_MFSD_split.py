"""Split MSU-MFSD videos into {train,test}/{live,spoof}/.

Expected raw layout:
    src_path/
      information/
        train_sub_list.txt   # one client ID per line
        test_sub_list.txt
      real/                  # client0{ID}_*.{avi,mov,mp4}
      attack/                # client0{ID}_*.{avi,mov,mp4}

Output layout:
    dst_path/{train,test}/{live,spoof}/<original filename>
"""
import argparse
import os
import shutil


def read_id_list(path):
    with open(path, 'r') as f:
        return [line for line in f.read().splitlines() if line]


def classify(sub_list, src_folder, dst_folder):
    os.makedirs(dst_folder, exist_ok=True)
    for client_id in sub_list:
        for file in os.listdir(src_folder):
            if f"client0{client_id}_" in file:
                shutil.copy2(os.path.join(src_folder, file), os.path.join(dst_folder, file))


def split(src_root, dst_root):
    train_ids = read_id_list(os.path.join(src_root, 'information', 'train_sub_list.txt'))
    test_ids = read_id_list(os.path.join(src_root, 'information', 'test_sub_list.txt'))

    for phase, ids in [('train', train_ids), ('test', test_ids)]:
        print(f'Copy processing for {phase.upper()} data')
        classify(ids, os.path.join(src_root, 'real'), os.path.join(dst_root, phase, 'live'))
        classify(ids, os.path.join(src_root, 'attack'), os.path.join(dst_root, phase, 'spoof'))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--src', required=True, help='Path to MSU-MFSD raw root with real/, attack/, information/.')
    parser.add_argument('--dst', required=True, help='Output path; will create {train,test}/{live,spoof}/.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    split(args.src, args.dst)
