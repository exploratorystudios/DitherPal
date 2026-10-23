"""Prefetch the model or export a neural depth preview without launching the GUI."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from .model import DEFAULT_MODEL, download_model, image_and_depth


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download-model', action='store_true')
    parser.add_argument('--image', type=Path)
    parser.add_argument('--output', type=Path, help='New 16-bit PNG depth preview, white = near')
    args = parser.parse_args()
    if not args.download_model and not args.image:
        parser.error('Use --download-model or --image PATH --output PATH.png')
    if args.image and (not args.output or args.output.suffix.lower() != '.png'):
        parser.error('Depth export requires --output with a .png extension')
    if args.output and args.output.exists():
        parser.error('Output already exists; choose a new filename')
    if args.download_model:
        print(download_model(DEFAULT_MODEL, print))
    if args.image:
        _, near = image_and_depth(args.image, print)
        with args.output.open('xb') as output:
            Image.fromarray(np.uint16(np.round(near*65535))).save(output, 'PNG')
        print(args.output)


if __name__ == '__main__':
    main()
