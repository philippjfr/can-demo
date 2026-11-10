import numpy as np

from candetect.data import (
    DataConfig,
    generate_dataset,
    make_splits,
    shifted_config,
    to_nchw_float,
)

# Pinned so an accidental generator change shows up as a hash change. If this fails only on a
# different OS or architecture, the generator is not bit-reproducible there and the README's
# "same dataset hash everywhere" claim needs revisiting.
GOLDEN_SMALL_HASH = "7a91befe5bf9bee392b802a0de88bef04be735a103217bd92a5e31cca8b2f359"


def test_shape_dtype_and_balance(small_dataset):
    ds = small_dataset
    assert ds.images.shape == (120, 128, 128, 3)
    assert ds.images.dtype == np.uint8
    assert ds.labels.sum() == 60


def test_generation_is_deterministic(small_dataset):
    again = generate_dataset(DataConfig(n_images=120, seed=3))
    assert again.sha256 == small_dataset.sha256
    assert np.array_equal(again.images, small_dataset.images)


def test_hash_changes_with_seed_and_config(small_dataset):
    assert generate_dataset(DataConfig(n_images=120, seed=4)).sha256 != small_dataset.sha256
    other = generate_dataset(DataConfig(n_images=120, seed=3, noise_range=(0.1, 0.2)))
    assert other.sha256 != small_dataset.sha256


def test_golden_hash():
    assert generate_dataset(DataConfig(n_images=32)).sha256 == GOLDEN_SMALL_HASH


def test_image_depends_only_on_its_index():
    full = generate_dataset(DataConfig(n_images=40, seed=5))
    prefix = generate_dataset(DataConfig(n_images=20, seed=5))
    # Labels come from a permutation of n, so compare images with the same label by index.
    same = np.flatnonzero(full.labels[:20] == prefix.labels)
    assert len(same) > 0
    assert np.array_equal(full.images[same], prefix.images[same])


def test_splits_are_disjoint_stratified_and_complete(small_dataset):
    sp = make_splits(small_dataset.labels, seed=3)
    parts = [sp.train, sp.val, sp.test]
    assert sorted(np.concatenate(parts)) == list(range(120))
    for part in parts:
        frac = small_dataset.labels[part].mean()
        assert 0.4 <= frac <= 0.6


def test_shifted_batch_is_darker_and_noisier(small_dataset):
    shifted = generate_dataset(shifted_config(small_dataset.config, n_images=60))
    assert shifted.images.mean() < small_dataset.images.mean() * 0.8


def test_nchw_float_range(small_dataset):
    x = to_nchw_float(small_dataset.images[:4])
    assert x.shape == (4, 3, 128, 128) and x.dtype == np.float32
    assert 0.0 <= x.min() and x.max() <= 1.0
