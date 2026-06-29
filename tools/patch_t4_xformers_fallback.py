import json
from pathlib import Path


def patch_notebook(path: Path) -> None:
    nb = json.loads(path.read_text(encoding="utf-8"))

    src = "".join(nb["cells"][2]["source"])
    if "XFORMERS_DISABLED" not in src:
        src = src.replace(
            'os.environ["COMPUTE_TIER"] = "T4"\n',
            'os.environ["COMPUTE_TIER"] = "T4"\n'
            "# T4 fallback: avoid xformers kernels that lack backward support for Qwen GQA.\n"
            'os.environ.setdefault("XFORMERS_DISABLED", "1")\n'
            'os.environ.setdefault("USE_XFORMERS", "0")\n',
        )
    nb["cells"][2]["source"] = src.splitlines(keepends=True)

    src = "".join(nb["cells"][3]["source"])
    if "pip uninstall -y -q xformers" not in src:
        src += (
            "\n"
            "# xformers memory_efficient_attention backward can fail on T4 for Qwen GQA.\n"
            "# Removing it lets Transformers/Unsloth fall back to PyTorch SDPA.\n"
            "!pip uninstall -y -q xformers\n"
        )
    nb["cells"][3]["source"] = src.splitlines(keepends=True)

    fallback = '''import os
import sys
import gc
import torch

assert torch.cuda.is_available(), "DPO needs a CUDA GPU. See HARDWARE-GUIDE.md."

# If a previous DPO attempt failed, clear the old model/trainer before reloading.
for _name in ("trainer", "model", "tokenizer"):
    if _name in globals():
        del globals()[_name]
gc.collect()
torch.cuda.empty_cache()

# T4 repair: xformers can select a forward kernel whose backward pass does not
# support Qwen's grouped-query attention layout on compute capability 7.5.
# Patch xformers to use PyTorch SDPA before the DPO model is loaded.
os.environ.setdefault("XFORMERS_DISABLED", "1")
os.environ.setdefault("USE_XFORMERS", "0")

try:
    import xformers.ops as xops
    import torch.nn.functional as F

    def _sdpa_attention_fallback(query, key, value, attn_bias=None, p=0.0, scale=None, *args, **kwargs):
        dropout_p = p if query.requires_grad else 0.0
        is_causal = attn_bias is not None

        # xformers shape for GQA: [batch, seq, groups, heads, dim].
        if query.ndim == 5:
            bsz, q_len, groups, heads, dim = query.shape
            k_len = key.shape[1]
            v_dim = value.shape[-1]
            q = query.permute(0, 2, 3, 1, 4).reshape(bsz, groups * heads, q_len, dim)
            k = key.permute(0, 2, 3, 1, 4).reshape(bsz, groups * heads, k_len, dim)
            v = value.permute(0, 2, 3, 1, 4).reshape(bsz, groups * heads, k_len, v_dim)
            out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=dropout_p, is_causal=is_causal, scale=scale
            )
            return out.reshape(bsz, groups, heads, q_len, v_dim).permute(0, 3, 1, 2, 4)

        # xformers shape for MHA: [batch, seq, heads, dim].
        if query.ndim == 4:
            q = query.transpose(1, 2)
            k = key.transpose(1, 2)
            v = value.transpose(1, 2)
            out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=dropout_p, is_causal=is_causal, scale=scale
            )
            return out.transpose(1, 2)

        return F.scaled_dot_product_attention(
            query, key, value, dropout_p=dropout_p, is_causal=is_causal, scale=scale
        )

    xops.memory_efficient_attention = _sdpa_attention_fallback
    if hasattr(xops, "fmha"):
        xops.fmha.memory_efficient_attention = _sdpa_attention_fallback
    print("Patched xformers memory_efficient_attention -> torch SDPA fallback for T4 DPO.")
except Exception as exc:
    print(f"xformers fallback patch skipped ({exc}).")
'''
    nb["cells"][48]["source"] = fallback.splitlines(keepends=True)

    # Remove the stale failed output while keeping earlier successful outputs as evidence.
    for idx in range(59, 66):
        if idx < len(nb["cells"]) and nb["cells"][idx].get("cell_type") == "code":
            nb["cells"][idx]["outputs"] = []
            nb["cells"][idx]["execution_count"] = None

    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")


patch_notebook(Path("colab/Lab22_DPO_T4.ipynb"))
patch_notebook(Path("kaggle/Lab22_DPO_T4_Kaggle.ipynb"))
print("patched T4 xformers fallback in Colab and Kaggle notebooks")
