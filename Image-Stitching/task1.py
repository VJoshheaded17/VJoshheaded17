"""Run the two-image mosaic example."""
import argparse
import torch
from stitching import stitch_background
from utils import read_images, write_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input_path', default='examples/task1')
    parser.add_argument('--output_path', '--output', default='results/runs/task1.png')
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    write_image(stitch_background(read_images(args.input_path)), args.output_path)
    print(f'Saved {args.output_path}')


if __name__ == '__main__':
    main()
