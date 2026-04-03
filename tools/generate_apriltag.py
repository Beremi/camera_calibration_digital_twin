"""Generate canonical AprilTag 36h11 marker images with OpenCV."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag-id", type=int, required=True, help="Marker id within the 36h11 dictionary.")
    parser.add_argument("--side-px", type=int, default=1024, help="Output image size in pixels.")
    parser.add_argument("--out", type=Path, required=True, help="Output PNG path.")
    args = parser.parse_args()

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    marker = cv2.aruco.generateImageMarker(dictionary, args.tag_id, args.side_px)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), marker)


if __name__ == "__main__":
    main()
