# services/ingestion-worker/models/uap_net.py
#
# UAPNet — a multi-modal, multi-task PyTorch model for UAP classification.
# Visual branch + telemetry branch → fused embedding → two heads:
#   * Head A: morphological shape  (5 classes, matches Shape enum)
#   * Head B: kinematic profile    (4 classes, matches KINEMATIC_PROFILES)
"""Custom UAPNet architecture and ONNX export helper."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_SHAPE_CLASSES: int = 5      # Tic-Tac, Sphere, Disc, Triangle, Unknown
NUM_KINEMATIC_CLASSES: int = 4  # Subsonic, Supersonic, Hypersonic, Trans-Medium
TELEMETRY_DIM: int = 5          # [v, a, h, rcs, g-force]
VISUAL_EMBED_DIM: int = 256
TELEMETRY_EMBED_DIM: int = 256
FUSED_DIM: int = VISUAL_EMBED_DIM + TELEMETRY_EMBED_DIM  # 512


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _conv_block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class UAPNet(nn.Module):
    """Multi-modal multi-task model: image (3×224×224) + telemetry (5-d)."""

    def __init__(
        self,
        num_shape_classes: int = NUM_SHAPE_CLASSES,
        num_kinematic_classes: int = NUM_KINEMATIC_CLASSES,
        telemetry_dim: int = TELEMETRY_DIM,
    ) -> None:
        super().__init__()

        # ---- Visual branch ------------------------------------------------
        self.visual_branch = nn.Sequential(
            _conv_block(3, 32),     # 224 → 112
            _conv_block(32, 64),    # 112 →  56
            _conv_block(64, VISUAL_EMBED_DIM),  # 56 → 28
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        # ---- Telemetry branch --------------------------------------------
        self.telemetry_branch = nn.Sequential(
            nn.Linear(telemetry_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, TELEMETRY_EMBED_DIM),
            nn.ReLU(inplace=True),
        )

        # Learned "missing telemetry" embedding. ``zeros`` initialisation
        # means an absent telemetry stream contributes nothing on the first
        # gradient step, but the parameter is still trainable.
        self.telemetry_missing = nn.Parameter(torch.zeros(TELEMETRY_EMBED_DIM))

        # ---- Heads --------------------------------------------------------
        self.head_shape     = nn.Linear(FUSED_DIM, num_shape_classes)
        self.head_kinematic = nn.Linear(FUSED_DIM, num_kinematic_classes)

        # Loss-balancing coefficients (tunable).
        self.alpha = 1.0
        self.beta  = 1.0

    # ---- Forward ---------------------------------------------------------

    def forward(
        self,
        visual_input: torch.Tensor,
        telemetry_input: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(shape_logits, kinematic_logits)``.

        When ``telemetry_input`` is ``None`` the learned
        ``telemetry_missing`` embedding is broadcast over the batch and
        gradients are detached for that path so the missing-modality token
        evolves independently of the visual loss.
        """
        b = visual_input.shape[0]
        v_emb = self.visual_branch(visual_input)            # (B, 256)

        if telemetry_input is None:
            t_emb = self.telemetry_missing.detach().unsqueeze(0).expand(b, -1)
        else:
            t_emb = self.telemetry_branch(telemetry_input)  # (B, 256)

        fused = torch.cat([v_emb, t_emb], dim=1)            # (B, 512)
        return self.head_shape(fused), self.head_kinematic(fused)

    # ---- Loss ------------------------------------------------------------

    def compute_loss(
        self,
        shape_logits:     torch.Tensor,
        kinematic_logits: torch.Tensor,
        shape_targets:    torch.Tensor,
        kinematic_targets: torch.Tensor,
    ) -> torch.Tensor:
        """``total = α · CE(shape) + β · CE(kinematic)``."""
        ce = nn.functional.cross_entropy
        return (
            self.alpha * ce(shape_logits,     shape_targets)
            + self.beta  * ce(kinematic_logits, kinematic_targets)
        )


# ---------------------------------------------------------------------------
# ONNX export helper
# ---------------------------------------------------------------------------

def save_onnx(
    model: UAPNet,
    dummy_visual: torch.Tensor,
    dummy_telemetry: torch.Tensor,
    path: str,
    opset: int = 17,
) -> None:
    """Export ``model`` to ONNX for C++/Rust deployment.

    Both inputs and both outputs are declared dynamic on the batch axis.
    """
    model.eval()
    torch.onnx.export(
        model,
        (dummy_visual, dummy_telemetry),
        path,
        input_names  = ["visual_input", "telemetry_input"],
        output_names = ["shape_logits", "kinematic_logits"],
        dynamic_axes = {
            "visual_input":     {0: "batch"},
            "telemetry_input":  {0: "batch"},
            "shape_logits":     {0: "batch"},
            "kinematic_logits": {0: "batch"},
        },
        opset_version = opset,
    )


__all__ = [
    "UAPNet",
    "save_onnx",
    "NUM_SHAPE_CLASSES",
    "NUM_KINEMATIC_CLASSES",
    "TELEMETRY_DIM",
]
