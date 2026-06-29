import json
from pathlib import Path


NOTEBOOKS = [
    Path("colab/Lab22_DPO_T4.ipynb"),
    Path("kaggle/Lab22_DPO_T4_Kaggle.ipynb"),
]

QWEN_TEMPLATE_CODE = '''QWEN_CHAT_TEMPLATE = """{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}<|im_start|>system
You are a helpful assistant.<|im_end|>
{% endif %}<|im_start|>{{ message['role'] }}
{{ message['content'] }}<|im_end|>
{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
{% endif %}"""


def ensure_chat_template(tokenizer):
    if not getattr(tokenizer, "chat_template", None):
        tokenizer.chat_template = QWEN_CHAT_TEMPLATE
    return tokenizer
'''

CELL_48 = '''import os
import gc
import torch

assert torch.cuda.is_available(), "DPO needs a CUDA GPU. See HARDWARE-GUIDE.md."

QWEN_CHAT_TEMPLATE = """{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}<|im_start|>system
You are a helpful assistant.<|im_end|>
{% endif %}<|im_start|>{{ message['role'] }}
{{ message['content'] }}<|im_end|>
{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
{% endif %}"""


def ensure_chat_template(tokenizer):
    if not getattr(tokenizer, "chat_template", None):
        tokenizer.chat_template = QWEN_CHAT_TEMPLATE
    return tokenizer

# If a previous DPO attempt failed, clear the old model/trainer before reloading.
for _name in ("trainer", "model", "tokenizer"):
    if _name in globals():
        del globals()[_name]
gc.collect()
torch.cuda.empty_cache()

os.environ["XFORMERS_DISABLED"] = "1"
os.environ["USE_XFORMERS"] = "0"
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)


def _torch_sdpa_xformers_fallback(query, key, value, attn_bias=None, p=0.0, scale=None, *args, **kwargs):
    """Drop-in replacement for xformers attention on T4.

    Unsloth's attention_dispatch keeps its own xformers_attention function
    reference, so patching xformers.ops alone is not enough.
    """
    import torch.nn.functional as F

    dropout_p = p if query.requires_grad else 0.0
    is_causal = attn_bias is not None

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


def patch_t4_attention():
    patched = []
    try:
        import xformers.ops as xops
        xops.memory_efficient_attention = _torch_sdpa_xformers_fallback
        patched.append("xformers.ops")
        if hasattr(xops, "fmha"):
            xops.fmha.memory_efficient_attention = _torch_sdpa_xformers_fallback
            patched.append("xformers.ops.fmha")
    except Exception as exc:
        print(f"xformers.ops patch skipped ({exc}).")

    try:
        import unsloth.utils.attention_dispatch as attention_dispatch
        attention_dispatch.xformers_attention = _torch_sdpa_xformers_fallback
        patched.append("unsloth.utils.attention_dispatch")
    except Exception as exc:
        print(f"Unsloth attention_dispatch patch skipped ({exc}).")

    print("T4 attention fallback active:", ", ".join(patched) if patched else "no xformers bindings found")


patch_t4_attention()
'''

CELL_50 = '''from unsloth import FastLanguageModel
from peft import PeftModel

# Importing Unsloth can bind xformers_attention inside attention_dispatch.
# Patch again after import, before loading the model.
patch_t4_attention()

# Policy — gets new DPO LoRA adapter on top of SFT LoRA
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=MAX_LEN,
    dtype=None,
    load_in_4bit=True,
    attn_implementation="eager",
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
ensure_chat_template(tokenizer)

# Load SFT adapter on top of base
model = PeftModel.from_pretrained(model, str(SFT_PATH), is_trainable=True)
print(f"Policy: {model.__class__.__name__} with SFT adapter loaded")
'''


def patch_generation_cell(source: str) -> str:
    if "QWEN_CHAT_TEMPLATE" not in source:
        source = QWEN_TEMPLATE_CODE + "\n" + source
    source = source.replace(
        "    if tokenizer.pad_token is None:\n        tokenizer.pad_token = tokenizer.eos_token\n\n    model = PeftModel.from_pretrained",
        "    if tokenizer.pad_token is None:\n        tokenizer.pad_token = tokenizer.eos_token\n    ensure_chat_template(tokenizer)\n\n    model = PeftModel.from_pretrained",
    )
    source = source.replace(
        "    if tokenizer.pad_token is None:\n        tokenizer.pad_token = tokenizer.eos_token\n    model = PeftModel.from_pretrained",
        "    if tokenizer.pad_token is None:\n        tokenizer.pad_token = tokenizer.eos_token\n    ensure_chat_template(tokenizer)\n    model = PeftModel.from_pretrained",
    )
    return source


def patch(path: Path) -> None:
    nb = json.loads(path.read_text(encoding="utf-8"))

    # Install cell: remove xformers after dependency install on fresh runs.
    src = "".join(nb["cells"][3]["source"])
    if "pip uninstall -y -q xformers" not in src:
        src += (
            "\n"
            "# xformers backward can fail on T4 for Qwen grouped-query attention.\n"
            "!pip uninstall -y -q xformers\n"
        )
    nb["cells"][3]["source"] = src.splitlines(keepends=True)

    nb["cells"][48]["source"] = CELL_48.splitlines(keepends=True)
    nb["cells"][50]["source"] = CELL_50.splitlines(keepends=True)

    for idx in (73, 127):
        src = "".join(nb["cells"][idx]["source"])
        nb["cells"][idx]["source"] = patch_generation_cell(src).splitlines(keepends=True)

    # Cell 94 also loads a tokenizer for merge/deploy; keep template there too.
    src = "".join(nb["cells"][94]["source"])
    if "ensure_chat_template(tokenizer)" not in src:
        src = src.replace(
            "if tokenizer.pad_token is None:\n    tokenizer.pad_token = tokenizer.eos_token\n\n# Stack",
            "if tokenizer.pad_token is None:\n    tokenizer.pad_token = tokenizer.eos_token\nensure_chat_template(tokenizer)\n\n# Stack",
        )
    nb["cells"][94]["source"] = src.splitlines(keepends=True)

    # Clear stale failed outputs and downstream outputs that depended on them.
    for idx in list(range(48, 66)) + list(range(73, 78)):
        if idx < len(nb["cells"]) and nb["cells"][idx].get("cell_type") == "code":
            nb["cells"][idx]["outputs"] = []
            nb["cells"][idx]["execution_count"] = None

    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")


for notebook in NOTEBOOKS:
    if notebook.exists():
        patch(notebook)

print("repaired T4 notebooks")
