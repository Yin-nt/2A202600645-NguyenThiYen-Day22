import json
from pathlib import Path


NOTEBOOKS = [
    Path("colab/Lab22_DPO_T4.ipynb"),
    Path("kaggle/Lab22_DPO_T4_Kaggle.ipynb"),
]

CELL_95 = """> **Note:** The aligned model is produced by applying adapters in order:
> base → SFT-mini → DPO. The first merge path tries Unsloth's saver for speed.
> If that hits a `transformers.core_model_loading.reverse_op` issue, the next
> cell falls back to standard `transformers + PEFT merge_and_unload`.
"""

CELL_97 = '''# Merge SFT + DPO into base weights.
# Primary path: Unsloth's saver. Fallback path: standard Transformers + PEFT.
import gc
import torch


def _base_model_for_full_precision(base_model_name: str) -> str:
    """Map Unsloth 4-bit repos to the regular HF base used for merging."""
    mapping = {
        "unsloth/Qwen2.5-3B-bnb-4bit": "Qwen/Qwen2.5-3B-Instruct",
        "unsloth/Qwen2.5-7B-bnb-4bit": "Qwen/Qwen2.5-7B-Instruct",
    }
    return mapping.get(
        base_model_name,
        base_model_name.replace("unsloth/", "Qwen/").replace("-bnb-4bit", "-Instruct"),
    )


def _fallback_merge_with_peft():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    merge_base = _base_model_for_full_precision(BASE_MODEL)
    print(f"Fallback merge: loading full-precision base {merge_base}")

    fallback_tokenizer = AutoTokenizer.from_pretrained(merge_base, trust_remote_code=True)
    if fallback_tokenizer.pad_token is None:
        fallback_tokenizer.pad_token = fallback_tokenizer.eos_token
    ensure_chat_template(fallback_tokenizer)

    base = AutoModelForCausalLM.from_pretrained(
        merge_base,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    print(f"Fallback merge: applying SFT adapter from {SFT_PATH}")
    merged = PeftModel.from_pretrained(base, str(SFT_PATH))
    merged = merged.merge_and_unload()

    print(f"Fallback merge: applying DPO adapter from {DPO_PATH}")
    merged = PeftModel.from_pretrained(merged, str(DPO_PATH))
    merged = merged.merge_and_unload()

    merged.save_pretrained(str(MERGED_PATH), safe_serialization=True, max_shard_size="2GB")
    fallback_tokenizer.save_pretrained(str(MERGED_PATH))
    print(f"Saved merged FP16 via PEFT fallback to {MERGED_PATH}")

    del merged
    gc.collect()
    torch.cuda.empty_cache()


try:
    print("Trying Unsloth save_pretrained_merged...")
    model.save_pretrained_merged(
        str(MERGED_PATH),
        tokenizer,
        save_method="merged_16bit",
    )
    tokenizer.save_pretrained(str(MERGED_PATH))
    print(f"Saved merged FP16 via Unsloth to {MERGED_PATH}")
except NotImplementedError as exc:
    print(f"Unsloth merge hit NotImplementedError ({exc}); using PEFT fallback.")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    _fallback_merge_with_peft()

# Free GPU memory before GGUF conversion (which spawns a subprocess that needs RAM)
if "model" in globals():
    del model
gc.collect()
torch.cuda.empty_cache()
'''


def patch(path: Path) -> None:
    nb = json.loads(path.read_text(encoding="utf-8"))
    nb["cells"][95]["source"] = CELL_95.splitlines(keepends=True)
    nb["cells"][97]["source"] = CELL_97.splitlines(keepends=True)
    for idx in range(97, 104):
        if idx < len(nb["cells"]) and nb["cells"][idx].get("cell_type") == "code":
            nb["cells"][idx]["outputs"] = []
            nb["cells"][idx]["execution_count"] = None
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")


for notebook in NOTEBOOKS:
    if notebook.exists():
        patch(notebook)

print("patched merge fallback in T4 notebooks")
