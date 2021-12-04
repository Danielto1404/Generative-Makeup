import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--tra', help='path to the json COCO annotations file')
parser.add_argument('--out', help='path to the output file')
parser.add_argument('--log ', default=True, type=bool, help='show the progress bar')

if __name__ == '__main__':
    args = parser.parse_args()
    coco_path = Path(args.coco_path)
    out_path = Path(args.out)

