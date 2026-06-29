import json
from pathlib import Path


NOTEBOOKS = [
    Path("colab/Lab22_DPO_T4.ipynb"),
    Path("kaggle/Lab22_DPO_T4_Kaggle.ipynb"),
]

CELL_48 = '''import os
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

# T4 repair: xformers backward does not support Qwen grouped-query attention
# layout on compute capability 7.5. Block xformers before loading the DPO model
# so Transformers/Unsloth use eager/SDPA attention instead.
os.environ["XFORMERS_DISABLED"] = "1"
os.environ["USE_XFORMERS"] = "0"

for _module_name in list(sys.modules):
    if _module_name == "xformers" or _module_name.startswith("xformers."):
        del sys.modules[_module_name]
sys.modules["xformers"] = None

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

print("T4 DPO fallback active: xformers blocked, PyTorch math/SDPA attention enabled.")
'''


def patch(path: Path) -> None:
    nb = json.loads(path.read_text(encoding="utf-8"))

    # Early env setup.
    src = "".join(nb["cells"][2]["source"])
    if "XFORMERS_DISABLED" not in src:
        src = src.replace(
            'os.environ["COMPUTE_TIER"] = "T4"\n',
            'os.environ["COMPUTE_TIER"] = "T4"\n'
            "# T4 fallback: avoid xformers kernels that lack backward support for Qwen GQA.\n"
            'os.environ["XFORMERS_DISABLED"] = "1"\n'
            'os.environ["USE_XFORMERS"] = "0"\n',
        )
    else:
        src = src.replace('os.environ.setdefault("XFORMERS_DISABLED", "1")', 'os.environ["XFORMERS_DISABLED"] = "1"')
        src = src.replace('os.environ.setdefault("USE_XFORMERS", "0")', 'os.environ["USE_XFORMERS"] = "0"')
    nb["cells"][2]["source"] = src.splitlines(keepends=True)

    # Install cell: uninstall xformers after dependency install.
    src = "".join(nb["cells"][3]["source"])
    if "pip uninstall -y -q xformers" not in src:
        src += (
            "\n"
            "# xformers memory_efficient_attention backward can fail on T4 for Qwen GQA.\n"
            "# Removing it lets Transformers/Unsloth fall back to PyTorch SDPA/eager attention.\n"
            "!pip uninstall -y -q xformers\n"
        )
    nb["cells"][3]["source"] = src.splitlines(keepends=True)

    nb["cells"][48]["source"] = CELL_48.splitlines(keepends=True)

    # DPO model load: pass attention implementation into the model loader.
    src = "".join(nb["cells"][50]["source"])
    if "attn_implementation=" not in src:
        src = src.replace(
            "    load_in_4bit=True,\n",
            "    load_in_4bit=True,\n"
            '    attn_implementation="eager",\n',
        )
    nb["cells"][50]["source"] = src.splitlines(keepends=True)

    # Remove the ineffective standalone assignment if present.
    src = "".join(nb["cells"][51]["source"])
    src = src.replace('attn_implementation="eager"\n', "")
    nb["cells"][51]["source"] = src.splitlines(keepends=True)

    # Clear stale failed DPO outputs; keep earlier successful SFT/data outputs.
    for idx in range(48, 66):
        if nb["cells"][idx].get("cell_type") == "code":
            nb["cells"][idx]["outputs"] = []
            nb["cells"][idx]["execution_count"] = None

    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")


for notebook in NOTEBOOKS:
    if notebook.exists():
        patch(notebook)

print("patched T4 notebooks to disable xformers for DPO")
