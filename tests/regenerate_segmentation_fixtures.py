"""Saves current segmentation output as the expected files for test_segmentation."""
import os

from test_segmentation import EXPECTED, LABELS, segment

os.makedirs(EXPECTED, exist_ok=True)
for label in LABELS:
    for name, output in segment(label).items():
        with open(os.path.join(EXPECTED, name), "w", newline="\n") as handle:
            handle.write(output)
        print("wrote", name)
