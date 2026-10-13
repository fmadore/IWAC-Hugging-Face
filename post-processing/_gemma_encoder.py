"""Local EmbeddingGemma 2 encoder shared by the two semantic-embedding scripts.

Runs the published ONNX export of ``google/embeddinggemma-2`` in fp32 with
onnxruntime and the model's own ``tokenizer.json``: no API key, no torch, and
the same revisions below let anyone recompute the published vectors offline.
The 2026-10 pilot (``analyses/output/embeddinggemma2_pilot.md``) checked this
path against the transformers reference implementation (text vectors equal to
cosine 1.000000, identical token ids on 403 corpus chunks).

Text and images land in one 768-dim space, so ``embedding_image`` stays
comparable with ``embedding_OCR`` / ``embedding_tableOfContents``. Every vector
returned here is L2-normalized; pooled multi-chunk documents are not
(``_embedding_utils.average_embeddings``).

The model card forbids float16 (activations overflow it), so only the fp32
graph is used for corpus vectors. Query-side consumers may use the q8 export,
which the pilot measured at cosine ≥ 0.99988 to fp32.
"""
from __future__ import annotations

import math
from importlib.metadata import version
from typing import Iterator, List, Sequence

import numpy as np

MODEL_ID = "google/embeddinggemma-2"
MODEL_REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
ONNX_ID = "onnx-community/embeddinggemma-2-ONNX"
ONNX_REVISION = "daa72c51243991dfcaf9f9137d2c573d8f7790c0"
DIMENSION = 768
MAX_TOKENS = 8192
QUERY_PREFIX = "task: search result | query: "
# Gemma 4 image processor settings (processor_config.json at ONNX_REVISION).
PATCH_SIZE = 16
POOLING_KERNEL = 3
MAX_SOFT_TOKENS = 280
IMAGE_PROCESSING = "gemma4-aspect-preserving-pil-bicubic"


def document_text(text: str, title: str = "none") -> str:
    """The model card's document prompt; also what the Gemini column used."""
    return f"title: {title or 'none'} | text: {text}"


def query_text(query: str) -> str:
    return QUERY_PREFIX + query


def model_settings() -> dict:
    """Identity of the vectors, for provenance hashes.

    Library versions (onnxruntime, tokenizers) are deliberately absent: they
    do not change fp32 outputs beyond float noise, and hashing them would
    invalidate the whole corpus on every patch upgrade. Hashing the model and
    export revisions is what keeps vector spaces from mixing.
    """
    return {"model": MODEL_ID, "model_revision": MODEL_REVISION, "runtime": "onnxruntime",
            "onnx_export": f"{ONNX_ID}@{ONNX_REVISION}", "precision": "fp32", "dimension": DIMENSION,
            "max_tokens": MAX_TOKENS, "token_pooling": "mean", "query_format": QUERY_PREFIX + "{query}"}


def runtime_versions() -> dict:
    return {"onnxruntime": version("onnxruntime"), "tokenizers": version("tokenizers")}


def validate_vectors(vectors: np.ndarray, expected_count: int) -> np.ndarray:
    """Reject a whole batch whose shape or values could be silently wrong."""
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.shape != (expected_count, DIMENSION):
        raise ValueError(f"Encoder returned shape {vectors.shape}, expected ({expected_count}, {DIMENSION})")
    if not np.isfinite(vectors).all():
        raise ValueError("Encoder returned a non-finite value")
    if not np.abs(vectors).sum(axis=1).all():
        raise ValueError("Encoder returned an all-zero vector")
    return vectors


def token_batches(lengths: Sequence[int], budget: int) -> Iterator[List[int]]:
    """Indices grouped shortest-first so a batch's padded size stays within ``budget``.

    Sorting by length keeps padding (wasted CPU) small; a single sequence longer
    than the budget still gets a batch of its own.
    """
    batch: List[int] = []
    longest = 0
    for index in sorted(range(len(lengths)), key=lambda i: (lengths[i], i)):
        size = lengths[index]
        if batch and max(longest, size) * (len(batch) + 1) > budget:
            yield batch
            batch, longest = [], 0
        batch.append(index)
        longest = max(longest, size)
    if batch:
        yield batch


def aspect_ratio_size(height: int, width: int) -> tuple[int, int]:
    """Gemma 4's resize target: the largest size within the patch budget whose
    sides are multiples of ``POOLING_KERNEL * PATCH_SIZE``."""
    max_patches = MAX_SOFT_TOKENS * POOLING_KERNEL ** 2
    target_px = max_patches * PATCH_SIZE ** 2
    factor = math.sqrt(target_px / (height * width))
    side = POOLING_KERNEL * PATCH_SIZE
    target_h = int(math.floor(factor * height / side)) * side
    target_w = int(math.floor(factor * width / side)) * side
    if target_h == 0 and target_w == 0:
        raise ValueError(f"Cannot resize a {height}x{width} image within the patch budget")
    max_side = (max_patches // POOLING_KERNEL ** 2) * side
    if target_h == 0:
        target_h, target_w = side, min(int(math.floor(width / height)) * side, max_side)
    elif target_w == 0:
        target_w, target_h = side, min(int(math.floor(height / width)) * side, max_side)
    return target_h, target_w


def image_patches(image) -> tuple[np.ndarray, np.ndarray, int]:
    """Pixel patches, (x, y) patch positions and soft-token count for one image,
    padded to the patch budget exactly as ``Gemma4ImageProcessor`` does."""
    from PIL import Image

    image = image.convert("RGB")
    target_h, target_w = aspect_ratio_size(image.height, image.width)
    if (target_h, target_w) != (image.height, image.width):
        image = image.resize((target_w, target_h), Image.Resampling.BICUBIC)
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    rows, cols = target_h // PATCH_SIZE, target_w // PATCH_SIZE
    patches = (pixels.reshape(rows, PATCH_SIZE, cols, PATCH_SIZE, 3)
               .transpose(0, 2, 1, 3, 4).reshape(rows * cols, -1))
    xs, ys = np.meshgrid(np.arange(cols), np.arange(rows), indexing="xy")
    positions = np.stack([xs, ys], axis=-1).reshape(rows * cols, 2).astype(np.int64)
    budget = MAX_SOFT_TOKENS * POOLING_KERNEL ** 2
    padded = np.zeros((budget, patches.shape[1]), np.float32)
    padded[:len(patches)] = patches
    padded_positions = np.full((budget, 2), -1, np.int64)
    padded_positions[:len(positions)] = positions
    return padded, padded_positions, len(patches) // POOLING_KERNEL ** 2


class GemmaEncoder:
    """fp32 ONNX sessions for text (and optionally images) at the pinned revisions."""

    def __init__(self, *, images: bool = False, providers: Sequence[str] = ("CPUExecutionProvider",),
                 threads: int | None = None, token: str | None = None):
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        def fetch(name: str) -> str:
            return hf_hub_download(ONNX_ID, name, revision=ONNX_REVISION, token=token)

        self.tokenizer = Tokenizer.from_file(fetch("tokenizer.json"))
        # Same as transformers' truncation=True: specials (BOS/EOS) are kept.
        self.tokenizer.enable_truncation(MAX_TOKENS)
        self.pad_id = self.tokenizer.token_to_id("<pad>")
        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = threads

        def session(stem: str):
            path = fetch(f"onnx/{stem}.onnx")
            fetch(f"onnx/{stem}.onnx_data")
            return ort.InferenceSession(path, options, providers=list(providers))

        self.text = session("model")
        feature_dim = {i.name: i.shape for i in self.text.get_inputs()}["image_features"][-1]
        self.no_features = np.zeros((0, feature_dim), np.float32)
        self.vision = session("vision_encoder") if images else None
        self.providers = self.text.get_providers()

    def token_lengths(self, texts: Sequence[str]) -> List[int]:
        return [len(encoding.ids) for encoding in self.tokenizer.encode_batch(list(texts))]

    def _run(self, input_ids: np.ndarray, attention_mask: np.ndarray, image_features=None) -> np.ndarray:
        feed = {"input_ids": input_ids, "attention_mask": attention_mask,
                "image_features": self.no_features if image_features is None else image_features,
                "video_features": self.no_features, "audio_features": self.no_features}
        return self.text.run(["sentence_embedding"], feed)[0]

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Unit vectors for already-formatted texts (see ``document_text``)."""
        encodings = self.tokenizer.encode_batch(list(texts))
        width = max(len(e.ids) for e in encodings)
        input_ids = np.full((len(encodings), width), self.pad_id, np.int64)
        attention_mask = np.zeros((len(encodings), width), np.int64)
        for row, encoding in enumerate(encodings):  # right padding, as the tokenizer config says
            input_ids[row, :len(encoding.ids)] = encoding.ids
            attention_mask[row, :len(encoding.ids)] = 1
        return validate_vectors(self._run(input_ids, attention_mask), len(texts))

    def encode_image(self, image) -> np.ndarray:
        """Unit vector for one PIL image: ``<|image>`` + soft tokens + ``<image|>``."""
        if self.vision is None:
            raise RuntimeError("GemmaEncoder was created without images=True")
        pixels, positions, soft_tokens = image_patches(image)
        features = self.vision.run(["image_features"], {"pixel_values": pixels[None],
                                                         "pixel_position_ids": positions[None]})[0]
        if len(features) != soft_tokens:
            raise ValueError(f"Vision encoder returned {len(features)} tokens, expected {soft_tokens}")
        ids = self.tokenizer.encode("<|image>" + "<|image|>" * soft_tokens + "<image|>").ids
        input_ids = np.asarray([ids], np.int64)
        vector = self._run(input_ids, np.ones_like(input_ids), features.astype(np.float32))
        return validate_vectors(vector, 1)[0]


__all__ = [
    "MODEL_ID", "MODEL_REVISION", "ONNX_ID", "ONNX_REVISION", "DIMENSION", "MAX_TOKENS",
    "QUERY_PREFIX", "IMAGE_PROCESSING", "document_text", "query_text", "model_settings",
    "runtime_versions", "validate_vectors", "token_batches", "aspect_ratio_size",
    "image_patches", "GemmaEncoder",
]
