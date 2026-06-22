"""
model.py — PCViT Architecture
Pyramid Convolutional Vision Transformer for 24-class coal-gangue spectral classification.

Shape flow:
  Input:        (Batch, 3, 1500)   — 3-channel physics tensor
  After stem:   (Batch, 128, 375)  — CNN compresses 1500 → 375 tokens (O(N²) solved)
  After permute:(Batch, 375, 128)  — reshape for Transformer (channels become features)
  + CLS token:  (Batch, 376, 128)  — prepend learnable summary vector
  After TF:     (Batch, 376, 128)  — self-attention across all 376 positions
  x[:, 0, :]:  (Batch, 128)       — CLS token has absorbed global context
  After Linear: (Batch, 24)        — raw logits → Softmax in main.py
"""
import torch
import torch.nn as nn


class PCViT(nn.Module):
    def __init__(self, num_classes: int = 24):
        super().__init__()

        # ── CNN Stem: 1500 → 750 → 750 → 375 tokens ──────────────────────────
        # Stride-2 convolutions solve the O(N²) complexity problem.
        # Feeding 1500 bands directly into Transformer = memory explosion.
        # The stem builds a compressed, dense representation first.
        # CRITICAL: attribute keys must be exactly {"stem", "cls", "pos", "tf", "head"}
        # to match the saved state_dict. Renaming any key = load crash.
        self.stem = nn.Sequential(
            nn.Conv1d(3,   64,  kernel_size=3, stride=2, padding=1),  # 1500 → 750
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64,  128, kernel_size=1, stride=1, padding=0),  # bottleneck
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1),  # 750 → 375
            nn.BatchNorm1d(128),
            nn.GELU(),
        )

        # ── CLS Token + Positional Embeddings ────────────────────────────────
        # CLS token: a blank learnable vector prepended to the sequence.
        # During self-attention, it "absorbs" global geological context from all
        # 375 other tokens. Only this token is sent to the classification head.
        # Analogy: like a class president who listens to all 375 students and
        # then reports a single verdict.
        self.cls = nn.Parameter(torch.randn(1, 1, 128))
        self.pos = nn.Parameter(torch.randn(1, 376, 128))  # 375 bands + 1 CLS

        # ── Transformer Body: 4 layers, 4 attention heads ────────────────────
        # 4 heads can specialize: head 1 → water band at 1400nm,
        # head 2 → water band at 1900nm, head 3 → Al-OH trap at 2200nm, etc.
        # batch_first=True: input is (Batch, Seq, Features) not (Seq, Batch, Features).
        self.tf = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=128,
                nhead=4,
                dim_feedforward=512,
                dropout=0.1,
                activation='gelu',
                batch_first=True,      # REQUIRED — must match training config
            ),
            num_layers=4,
        )

        # ── Classification Head ───────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.LayerNorm(128),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, 3, 1500)
        x = self.stem(x)                          # (Batch, 128, 375)
        x = x.permute(0, 2, 1)                    # (Batch, 375, 128)

        # Prepend CLS token to every sample in the batch
        cls = self.cls.expand(x.shape[0], -1, -1) # (Batch, 1, 128)
        x = torch.cat([cls, x], dim=1)             # (Batch, 376, 128)

        # Add positional embeddings (slice handles any minor shape mismatches)
        x = x + self.pos[:, :x.shape[1], :]

        x = self.tf(x)                             # (Batch, 376, 128)
        return self.head(x[:, 0, :])               # CLS token only → (Batch, 24)
