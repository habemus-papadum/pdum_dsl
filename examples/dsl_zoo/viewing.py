"""Zero-dependency viewing: ASCII in the terminal, PGM/PPM files on disk."""

import numpy as np

RAMP = " .:-=+*#%@"


def show(img, title=""):
    a = np.clip(img.to_numpy(), 0.0, 1.0)
    if title:
        print(title)
    for row in a:
        print("".join(RAMP[min(int(v * 9.99), 9)] for v in row))


def pgm(img, path):
    a = (np.clip(img.to_numpy(), 0.0, 1.0) * 255).astype(np.uint8)
    with open(path, "wb") as f:
        f.write(f"P5 {a.shape[1]} {a.shape[0]} 255\n".encode())
        f.write(a.tobytes())
    return path
