from pathlib import Path
from cis_course import utils


def test_extract_image():
    img_pth = Path(__file__).parent.joinpath("data/task7_noise_ea.img")
    result = utils.extract_images(img_pth)
    print(result)
    assert len(result) == 2
    assert result[0].shape == (298, 300)
    assert result[0].sum() == -33331141
    assert result[1].sum() == -33324132
