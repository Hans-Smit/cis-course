from pathlib import Path
import numpy as np


def bytes_to_uint16(byte_seq: bytes, index: int) -> int:
    return int.from_bytes(byte_seq[index:index + 2], byteorder="little")

    
def extract_images(image_file: Path | str) -> list[np.ndarray]:
    """Parse the binary image file and return a list of image arrays."""
    result = []
    pth = Path(image_file).expanduser()
    with open(pth, mode="rb") as f:
        img_all_bytes = f.read()

    header_size = bytes_to_uint16(img_all_bytes, 0)
    assert header_size == 6
    header = [bytes_to_uint16(img_all_bytes, i) for i in range(0, header_size * 2, 2)]
    
    rows, cols = header[3:5]
    hdr_byte_sz = header_size * 2
    img_byte_sz = (rows * cols) * 2  # 16 bit numbers
    tot_byte_sz = hdr_byte_sz + img_byte_sz
    i = 0
    while True:
        data = img_all_bytes[tot_byte_sz * i: tot_byte_sz * (i + 1)]
        if not data:
            break
        img_bytes = data[12:]
        assert data[0] == 0x06
        img = np.frombuffer(img_bytes, dtype="int16")
        img = img.reshape((rows, cols))
        result.append(img)
        i += 1
    return result
