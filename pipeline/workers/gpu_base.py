"""
pipeline/workers/gpu_base.py
Base class for GPU inference workers: fp16, mem-fraction capping, and recursive
OOM-halving batch retry — the pattern proven in scripts/12_language_id.py:141-181
and scripts/11_audio_tag.py, generalised so every label/ASR/speaker node reuses it
instead of re-implementing its own OOM backoff.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

import torch

log = logging.getLogger(__name__)


class GPUWorkerBase(ABC):
    """Subclasses implement load_model() and forward_batch(); this base class
    supplies device setup (fp16, mem-fraction cap) and OOM-safe batched inference.
    """

    def __init__(
        self,
        device: str = "cuda:0",
        *,
        mem_fraction: float | None = None,
        fp16: bool = True,
    ) -> None:
        if device.startswith("cuda") and not torch.cuda.is_available():
            log.warning("CUDA not available — falling back to cpu")
            device = "cpu"
        self.device = device
        self.use_fp16 = fp16 and device.startswith("cuda")

        if device.startswith("cuda") and mem_fraction:
            torch.cuda.set_per_process_memory_fraction(mem_fraction, device=device)
            log.info(f"GPU mem fraction capped at {mem_fraction} on {device}")

        self.model = self.load_model()

    @abstractmethod
    def load_model(self) -> Any:
        """Load and return the model, already .to(self.device).eval() (and
        .half() if self.use_fp16 — subclass decides, since some models keep
        parts in fp32)."""

    @abstractmethod
    def forward_batch(self, items: list) -> list:
        """Run inference on *items* (no OOM handling) and return one result
        per item, same order. May raise torch.cuda.OutOfMemoryError."""

    def infer_with_oom_halving(self, items: list) -> list:
        """Call forward_batch(items); on CUDA OOM, clear the cache and retry on
        halves (down to single items) so a transient co-running-training memory
        spike backs off instead of crashing the whole run. A true single-item
        OOM is left to propagate — there's nothing smaller to split into.
        """
        try:
            return self.forward_batch(items)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(items) == 1:
                log.warning("OOM on single clip — retrying after cache clear")
                return self.forward_batch(items)
            mid = len(items) // 2
            log.warning(f"OOM on batch {len(items)} — splitting (training spike?)")
            return (
                self.infer_with_oom_halving(items[:mid])
                + self.infer_with_oom_halving(items[mid:])
            )
