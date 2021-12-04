import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--coco_path', help='path to the json COCO annotations')

if __name__ == '__main__':
    args = parser.parse_args()
