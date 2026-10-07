"""Run graph-based multi-image stitching and save a binary overlap matrix."""
import argparse
import json
from pathlib import Path
import torch
from stitching import panorama
from utils import read_images, write_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input_path', default='examples/task2')
    parser.add_argument('--output_path', '--output', default='results/runs/task2.png')
    parser.add_argument('--json', default='results/runs/task2_overlap.json')
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    images = read_images(args.input_path)
    image, overlap = panorama(images)
    write_image(image, args.output_path)
    path = Path(args.json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(overlap.tolist(), indent=2) + '\n')
    print(f'Saved {args.output_path}; overlap rows/columns: {sorted(images)}')


if __name__ == '__main__':
    main()
