"""Offline contracts for the EmbeddingGemma 2 helpers (no model download)."""
from __future__ import annotations

import numpy as np
from PIL import Image

from _gemma_encoder import (
    MAX_SOFT_TOKENS,
    PATCH_SIZE,
    POOLING_KERNEL,
    aspect_ratio_size,
    image_patches,
    model_settings,
    token_batches,
)


def test_token_batches_cover_every_item_once_within_budget():
    lengths = [5, 300, 40, 40, 1200, 7, 90, 3]
    batches = list(token_batches(lengths, 400))
    assert sorted(i for b in batches for i in b) == list(range(len(lengths)))
    for batch in batches:
        assert len(batch) == 1 or max(lengths[i] for i in batch) * len(batch) <= 400
    assert batches[-1] == [4]  # longer than the budget: alone, never dropped
    flat = [lengths[i] for b in batches for i in b]
    assert flat == sorted(flat)  # shortest first keeps padding small


def test_resize_target_matches_the_reference_processor():
    # Values from transformers' Gemma4ImageProcessor on IWAC photographs.
    assert aspect_ratio_size(3096, 4128) == (672, 912)
    assert aspect_ratio_size(4128, 3096) == (912, 672)
    side = POOLING_KERNEL * PATCH_SIZE
    for height, width in [(480, 640), (1000, 1000), (60, 4000), (4000, 60), (17, 23)]:
        h, w = aspect_ratio_size(height, width)
        assert h % side == 0 and w % side == 0
        assert (h // PATCH_SIZE) * (w // PATCH_SIZE) <= MAX_SOFT_TOKENS * POOLING_KERNEL ** 2


def test_patches_follow_the_reference_layout_and_padding():
    rng = np.random.default_rng(0)
    pixels = rng.integers(0, 256, (96, 144, 3), dtype=np.uint8)  # already a valid target size
    assert aspect_ratio_size(96, 144) != (96, 144)  # the encoder upsamples small images
    patches, positions, soft_tokens = image_patches(Image.fromarray(pixels))
    h, w = aspect_ratio_size(96, 144)
    rows, cols = h // PATCH_SIZE, w // PATCH_SIZE
    assert soft_tokens == rows * cols // POOLING_KERNEL ** 2
    budget = MAX_SOFT_TOKENS * POOLING_KERNEL ** 2
    assert patches.shape == (budget, PATCH_SIZE * PATCH_SIZE * 3) and positions.shape == (budget, 2)
    assert (positions[rows * cols:] == -1).all() and not patches[rows * cols:].any()
    # (x, y) positions in row-major patch order, as torch.meshgrid(..., indexing="xy").
    assert positions[1].tolist() == [1, 0] and positions[cols].tolist() == [0, 1]
    # Patch flattening equals the reference permute(1, 3, 2, 4, 0) of a CHW tensor.
    resized = np.asarray(Image.fromarray(pixels).resize((w, h), Image.Resampling.BICUBIC), np.float32) / 255
    chw = resized.transpose(2, 0, 1).reshape(3, rows, PATCH_SIZE, cols, PATCH_SIZE)
    reference = chw.transpose(1, 3, 2, 4, 0).reshape(rows * cols, -1)
    assert np.array_equal(patches[:rows * cols], reference)


def test_model_identity_is_pinned_and_free_of_library_versions():
    settings = model_settings()
    assert len(settings["model_revision"]) == 40
    assert settings["onnx_export"].split("@")[1] and len(settings["onnx_export"].split("@")[1]) == 40
    assert settings["precision"] == "fp32"
    assert not {"onnxruntime", "tokenizers"} & set(settings)
