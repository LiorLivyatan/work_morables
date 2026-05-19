"""
Model loading + encoding. Single chokepoint for all transformers internals.

Public surface:
    load_model(cfg) -> EncoderHandle
    encode(handle, texts, batch_size=None) -> np.ndarray
    extract_hidden_states(handle, texts, layers, batch_size=None) -> dict[layer, ndarray]
    encode_with_intervention(handle, texts, layer, direction, alpha, ...) -> (ndarray, ndarray)

Pooling and layer indexing are detected at load time and logged so they are
verified, not assumed.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


@dataclass
class EncoderHandle:
    st_model: SentenceTransformer
    transformer: torch.nn.Module
    pooling_kind: str
    device: torch.device
    dtype: torch.dtype
    n_layers: int
    hidden_dim: int


_DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def load_model(cfg: dict) -> EncoderHandle:
    mc = cfg["model"]
    dtype = _DTYPE_MAP[mc["dtype"]]
    device = torch.device(mc["device"])
    st_model = SentenceTransformer(mc["hf_id"], device=str(device))
    st_model = st_model.to(dtype=dtype)

    pooling_kind = _detect_pooling(st_model)
    transformer = st_model._modules["0"].auto_model
    n_layers = transformer.config.num_hidden_layers
    hidden_dim = transformer.config.hidden_size

    print(f"[model] loaded {mc['hf_id']} on {device} dtype={dtype}")
    print(f"[model] pooling={pooling_kind} n_layers={n_layers} hidden_dim={hidden_dim}")
    # Surface ST prompt config so any silent prefix that encode() applies is
    # visible — extract_hidden_states mirrors this via _resolve_prompt() below.
    prompts = getattr(st_model, "prompts", {}) or {}
    default_prompt_name = getattr(st_model, "default_prompt_name", None)
    print(f"[model] st prompts={prompts} default_prompt_name={default_prompt_name}")
    return EncoderHandle(st_model, transformer, pooling_kind, device, dtype, n_layers, hidden_dim)


def _resolve_prompt(st_model: SentenceTransformer) -> str:
    """Return the prompt string that st_model.encode(texts) would prepend with
    no explicit prompt_name. Mirrors SentenceTransformer.encode's behaviour so
    extract_hidden_states tokenises the same text encode() would."""
    default = getattr(st_model, "default_prompt_name", None)
    prompts = getattr(st_model, "prompts", {}) or {}
    if default and default in prompts and prompts[default]:
        return prompts[default]
    return ""


def _detect_pooling(st_model: SentenceTransformer) -> str:
    pooling_module = st_model._modules.get("1")
    if pooling_module is None:
        raise RuntimeError("Could not find pooling module — unexpected SentenceTransformer architecture")
    # Try common attribute names used by SentenceTransformers Pooling module
    for attr, kind in (
        ("pooling_mode_lasttoken",       "last_token"),
        ("pooling_mode_mean_tokens",     "mean"),
        ("pooling_mode_cls_token",       "cls"),
    ):
        if getattr(pooling_module, attr, False):
            return kind
    name = type(pooling_module).__name__.lower()
    if "lasttoken" in name:
        return "last_token"
    if "mean" in name:
        return "mean"
    if "cls" in name:
        return "cls"
    raise RuntimeError(
        f"Unexpected pooling module: {type(pooling_module).__name__}. "
        f"Inspect ST modules.json and extend _detect_pooling."
    )


def encode(handle: EncoderHandle, texts: Sequence[str], batch_size: int = 8) -> np.ndarray:
    """Plain encoder. Returns L2-normalised float32 embeddings."""
    embs = handle.st_model.encode(
        list(texts),
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32)
    return embs


def _pool(hs: torch.Tensor, attn_mask: torch.Tensor, kind: str) -> torch.Tensor:
    """hs: (B, T, H), attn_mask: (B, T). Returns (B, H)."""
    if kind == "last_token":
        seq_lengths = attn_mask.sum(dim=1) - 1
        idx = seq_lengths.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, hs.size(-1))
        return hs.gather(dim=1, index=idx).squeeze(1)
    if kind == "mean":
        mask = attn_mask.unsqueeze(-1).type_as(hs)
        return (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    if kind == "cls":
        return hs[:, 0, :]
    raise ValueError(f"unknown pooling kind: {kind}")


def extract_hidden_states(
    handle: EncoderHandle,
    texts: Sequence[str],
    layers: Sequence[int],
    batch_size: int = 4,
) -> dict[int, np.ndarray]:
    """For each requested BLOCK INDEX, return (n_texts, hidden_dim) array of POOLED
    hidden states at the OUTPUT of that transformer block.

    Convention (consistent with encode_with_intervention and model.layers[]):
        block index b in [0, n_layers - 1]; b == -1 resolves to n_layers - 1.
        The output of block b is res.hidden_states[b + 1] (index 0 is embeddings).

    Pooling matches handle.pooling_kind.
    """
    requested_blocks = [((handle.n_layers - 1) if l == -1 else l) for l in layers]
    out: dict[int, list[np.ndarray]] = {b: [] for b in requested_blocks}

    tok = handle.st_model.tokenizer
    # Match whatever prefix encode() would prepend; if ST has no default prompt
    # configured this is a no-op and inputs are identical to before.
    prompt = _resolve_prompt(handle.st_model)
    prefixed = [prompt + t for t in texts] if prompt else list(texts)
    handle.transformer.eval()
    with torch.no_grad():
        for batch_start in range(0, len(prefixed), batch_size):
            batch = prefixed[batch_start: batch_start + batch_size]
            enc = tok(batch, padding=True, truncation=True, return_tensors="pt",
                       max_length=tok.model_max_length)
            enc = {k: v.to(handle.device) for k, v in enc.items()}
            res = handle.transformer(**enc, output_hidden_states=True)
            attn_mask = enc["attention_mask"]
            for block in requested_blocks:
                hs = res.hidden_states[block + 1]   # output of block `block`
                pooled = _pool(hs, attn_mask, handle.pooling_kind)
                out[block].append(pooled.float().cpu().numpy())

    return {b: np.concatenate(out[b], axis=0) for b in requested_blocks}


def _get_layer_module(handle: EncoderHandle, layer_idx: int) -> torch.nn.Module:
    """Return the transformer block at layer_idx. Mistral and most decoder
    LMs expose blocks at `model.layers[i]`. Some HF models expose `encoder.layer[i]`.
    """
    tr = handle.transformer
    # Mistral / Llama style
    if hasattr(tr, "model") and hasattr(tr.model, "layers"):
        return tr.model.layers[layer_idx]
    # Encoder style
    if hasattr(tr, "encoder") and hasattr(tr.encoder, "layer"):
        return tr.encoder.layer[layer_idx]
    if hasattr(tr, "layers"):
        return tr.layers[layer_idx]
    raise RuntimeError(
        "Could not locate transformer block list — extend _get_layer_module for this architecture"
    )


def encode_with_intervention(
    handle: EncoderHandle,
    texts: Sequence[str],
    *,
    layer_idx: int,
    direction: np.ndarray | None,
    alpha: float = 0.0,
    batch_size: int = 4,
    renormalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Forward with hook: at the OUTPUT of transformer block `layer_idx`, subtract
    `alpha * direction` from the residual stream at every token position.

    Returns (embeddings, pooled_cosine_pre_post).
        embeddings: (n_texts, hidden_dim) float32 (L2-normalised when renormalize)
        pooled_cosine_pre_post: per-text cosine between baseline and intervened pooled.
    """
    no_intervention = direction is None or alpha == 0.0
    layer_idx_resolved = handle.n_layers + layer_idx if layer_idx < 0 else layer_idx

    if no_intervention:
        embs = encode(handle, texts, batch_size=batch_size)
        return embs, np.ones(len(texts), dtype=np.float32)

    base_embs = encode(handle, texts, batch_size=batch_size)
    # Unit-normalise the direction so α has the same physical meaning across
    # layers (raw ‖v_caa‖ grows with depth — without this, the layer axis of
    # the sweep mixes "depth" and "intervention magnitude").
    dir_np = np.asarray(direction, dtype=np.float32)
    dir_norm = float(np.linalg.norm(dir_np))
    if dir_norm < 1e-12:
        return base_embs, np.ones(len(texts), dtype=np.float32)
    direction_t = torch.as_tensor(dir_np / dir_norm, device=handle.device, dtype=handle.dtype)

    def hook_fn(module, inputs, output):
        # Sign convention (locked in spec §1): v = mean(h_pos − h_neg) from
        # build_caa_vector. The intervention subtracts α·v̂; α > 0 and α < 0 are
        # both meaningful — direction of effect is read from the data, not
        # asserted as "enhance" or "suppress" a-priori.
        if isinstance(output, tuple):
            hs = output[0]
            modified = hs - alpha * direction_t
            return (modified,) + output[1:]
        return output - alpha * direction_t

    if not renormalize:
        raise NotImplementedError(
            "renormalize=False would require bypassing SentenceTransformer.encode; "
            "see follow-up plan if needed."
        )

    target_block = _get_layer_module(handle, layer_idx_resolved)
    h = target_block.register_forward_hook(hook_fn)
    try:
        intervened = encode(handle, texts, batch_size=batch_size)
    finally:
        h.remove()

    pooled_cosine = (intervened * base_embs).sum(axis=1)
    return intervened, pooled_cosine.astype(np.float32)
