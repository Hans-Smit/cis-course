from pathlib import Path
import numpy as np

from dataclasses import dataclass


@dataclass
class Header:
    """Image header data structure."""

    sz: int
    res1: int
    res2: int
    rows: int
    cols: int
    bits: int

    def __post_init__(self):
        assert self.sz == 6

    @property
    def image_size(self) -> int:
        """Return the image (data) size in bytes."""
        return self.rows * self.cols * (self.bits // 8)

    @property
    def dtype(self) -> str:
        """Return the numpy data-type."""
        # TODO: how to determine unsigned int or float types?
        if self.bits == 16:
            return "int16"
        if self.bits == 32:
            return "int32"
        if self.bits == 64:
            return "int64"
        raise ValueError(f"Cannot determine data-type. {self.bits=}")


def parse_header(byte_seq: bytes, index: int = 0) -> Header:
    """Extract the (rows, cols) from the header."""
    # typical byte_seq:
    # b'\x06\x00\x00\x00\x00\x00*\x01,\x01\x10\x00'
    #   | size  | ???   |???    |rows|cols|bits   |
    #  [6,      0,      0,      298, 300, 16]
    # i:0       1       2       3    4    5
    nums = np.frombuffer(byte_seq, dtype=np.uint16, count=6, offset=index)
    nums_int = [int(v) for v in nums]  # convert from `numpy.uint16` list to `int` list
    header = Header(*nums_int)
    return header


def load_images(image_file: Path | str) -> list[np.ndarray]:
    """Parse the binary image file and return a list of image arrays."""
    # TODO: implement this as generator (with yield)
    result = []

    pth = Path(image_file).expanduser()
    with open(pth, mode="rb") as f:
        header_bytes = f.read(12)

        # Loop over all data bytes (header + image) until the end is reached.
        while header_bytes:
            header = parse_header(header_bytes, 0)
            image_bytes = f.read(header.image_size)
            image = np.frombuffer(image_bytes, dtype=header.dtype)
            image = image.reshape((header.rows, header.cols))
            result.append(image)
            header_bytes = f.read(12)  # get next header

    return result
