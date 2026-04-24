"""
InfoNCE loss for ft_03 hard-negative fine-tuning.

Handles two modes depending on how many sentence columns the dataset has:
  2 columns (anchor, positive)           → types 1 + 2
  3 columns (anchor, positive, negative) → types 1 + 2 + 3

Negative types:
  Type 1 — moral vs. other fables in the batch (in-batch, free)
  Type 2 — moral vs. other morals in the batch (prevents embedding collapse)
  Type 3 — moral vs. mined hard negative fable (explicit, requires mining)

Multi-positive masking:
  27 morals in the dataset appear in 2–4 fables. When two such items land in
  the same batch, one fable is a false negative for the other. Pass the moral
  group ID as the dataset label and the loss will exclude these from the
  denominator automatically.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    InfoNCE loss compatible with SentenceTransformerTrainer.

    Args:
        model        the SentenceTransformer being trained
        temperature  softmax temperature τ (lower = sharper, harder training signal)
    """

    def __init__(self, model, temperature: float = 0.05):
        super().__init__()
        self.model = model
        self.temperature = temperature

    def forward(self, sentence_features: list, labels=None) -> torch.Tensor:
        # Encode and L2-normalise all inputs
        moral_emb = F.normalize(self.model(sentence_features[0])["sentence_embedding"], dim=-1)
        pos_emb   = F.normalize(self.model(sentence_features[1])["sentence_embedding"], dim=-1)

        has_hard_neg = len(sentence_features) == 3
        if has_hard_neg:
            # Stop-gradient on the hard negative: we still push the moral embedding
            # away from it, but skip backprop through its forward pass. This saves
            # ~33% activation memory and avoids OOM on 24 GB cards.
            with torch.no_grad():
                neg_emb = F.normalize(self.model(sentence_features[2])["sentence_embedding"], dim=-1).detach()

        B      = moral_emb.size(0)
        device = moral_emb.device
        τ      = self.temperature

        # ── Type 1: moral_i vs all in-batch fables ────────────────────────────
        # Diagonal = positive pair; off-diagonal = in-batch negatives
        fable_sim = torch.mm(moral_emb, pos_emb.T) / τ  # (B, B)

        # ── Multi-positive masking ────────────────────────────────────────────
        # Two items with the same moral group ID share a moral text → the other
        # item's positive fable is also a correct answer, not a negative.
        if labels is not None:
            same_moral = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B, B)
            same_moral.fill_diagonal_(False)  # keep the actual positive (diagonal)
            fable_sim = fable_sim.masked_fill(same_moral, float("-inf"))

        # ── Type 3: explicit hard negative ────────────────────────────────────
        if has_hard_neg:
            hard_neg_sim = (moral_emb * neg_emb).sum(dim=-1, keepdim=True) / τ  # (B, 1)

        # ── Type 2: moral_i vs other morals in the batch ─────────────────────
        moral_sim = torch.mm(moral_emb, moral_emb.T) / τ  # (B, B)
        # Mask diagonal — moral vs itself carries no information
        moral_sim = moral_sim.masked_fill(torch.eye(B, dtype=torch.bool, device=device), float("-inf"))

        # ── Combine all negatives ─────────────────────────────────────────────
        # Column layout: [fable_sim (B) | hard_neg (1, if present) | moral_sim (B)]
        # The positive for item i sits at column i in fable_sim.
        parts = [fable_sim]
        if has_hard_neg:
            parts.append(hard_neg_sim)
        parts.append(moral_sim)
        all_logits = torch.cat(parts, dim=1)  # (B, 2B) or (B, 2B+1)

        targets = torch.arange(B, device=device)
        return F.cross_entropy(all_logits, targets)
