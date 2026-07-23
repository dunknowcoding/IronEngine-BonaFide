"""Re-export of the completion prior trainer for users."""
from __future__ import annotations

from ironengine_bonafide.backends.cuda.completion import (
    CompletionPrior,
    train_completion_prior,
)

__all__ = ["CompletionPrior", "train_completion_prior"]
