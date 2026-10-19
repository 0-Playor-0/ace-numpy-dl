import numpy as np

from experiments.hidden_eval import (detect, idx_read, idx_write, load_any, png_read, png_write,
                                     resize_matrix, to_unit_gray28)


def test_png_roundtrip_gray_rgb(tmp_path):
    rng = np.random.default_rng(0)
    for c in (1, 3, 4):
        img = rng.integers(0, 256, (13, 17, c), dtype=np.uint8)
        png_write(tmp_path / f"a{c}.png", img if c > 1 else img[..., 0])
        assert np.array_equal(png_read(tmp_path / f"a{c}.png"), img)


def test_png_matches_matplotlib_writer(tmp_path):
    """matplotlib (Pillow) writes with adaptive row filters: exercises Sub/Up/Average/Paeth."""
    import matplotlib.image as mpimg
    rng = np.random.default_rng(1)
    img = np.cumsum(rng.integers(0, 4, (30, 40, 3)), axis=1).astype(np.float64)
    img = np.clip(img / img.max(), 0, 1)
    mpimg.imsave(tmp_path / "m.png", img)
    ref = np.round(mpimg.imread(tmp_path / "m.png") * 255).astype(np.uint8)
    assert np.array_equal(png_read(tmp_path / "m.png"), ref)


def test_idx_roundtrip_and_detect(tmp_path):
    X = np.arange(2 * 28 * 28, dtype=np.uint8).reshape(2, 28, 28)
    idx_write(tmp_path / "t-images-idx3-ubyte.gz", X)
    idx_write(tmp_path / "t-labels-idx1-ubyte.gz", np.array([3, 7], np.uint8))
    assert np.array_equal(idx_read(tmp_path / "t-images-idx3-ubyte.gz"), X)
    assert detect(str(tmp_path)) == "idx"
    fmt, X01, y = load_any(str(tmp_path))
    assert fmt == "idx" and X01.shape == (2, 28, 28) and list(y) == [3, 7]


def test_resize_and_inversion():
    for n_in in (14, 28, 56, 100):
        R = resize_matrix(n_in, 28)
        assert np.allclose(R.sum(1), 1)
    img = np.zeros((56, 56, 3), np.uint8)
    img[20:36, 20:36] = 255
    X = to_unit_gray28([255 - img], invert="auto")      # dark object on white -> inverted
    assert X.shape == (1, 28, 28) and X[0, 0, 0] < 0.01 and X[0, 14, 14] > 0.99
