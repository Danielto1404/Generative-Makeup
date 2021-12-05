import argparse

from train import run_from_args

parser = argparse.ArgumentParser()
parser.add_argument('--annotation_path', help='path to the json COCO annotations file')
parser.add_argument('--images_root', help='path to images folder')
parser.add_argument('--model_path', help='path to which the model will be saved')
parser.add_argument('--num_classes', type=int, help='num classes for segmentation')
parser.add_argument('--num_epochs', type=int, help='amount of training epochs')
parser.add_argument('--batch_size', type=int, help='batch size in train loader')
parser.add_argument('--lr', type=float, help='learning rate in Adam optimizer')
parser.add_argument('--l2', type=float, help='weight_decay in Adam optimizer')
parser.add_argument('--device', help='torch device to use')

if __name__ == '__main__':
    args = parser.parse_args()
    run_from_args(
        annotation_path=args.annotation_path,
        images_root=args.images_root,
        model_path=args.model_path,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        weight_decay=args.l2,
        device=args.device
    )
