import json
from pathlib import Path


src = Path("colab/Lab22_DPO_T4.ipynb")
dst = Path("kaggle/Lab22_DPO_T4_Kaggle.ipynb")
dst.parent.mkdir(exist_ok=True)

nb = json.loads(src.read_text(encoding="utf-8"))

nb["cells"][0]["source"] = [
    "# Lab 22 — DPO/ORPO Alignment (Kaggle T4 tier)\n",
    "\n",
    "**Track 3 · Day 22 · VinUni AICB program**\n",
    "\n",
    "This is a single-file Kaggle notebook stitching all 6 stages of the lab:\n",
    "1. SFT-mini build (replaces Lab 21)\n",
    "2. Preference data prep\n",
    "3. DPO training (the main event)\n",
    "4. Side-by-side comparison + eval\n",
    "5. Merge → GGUF → llama.cpp smoke test\n",
    "6. LLM benchmark (IFEval / GSM8K / MMLU / AlpacaEval-lite)\n",
    "\n",
    "**Tier:** `T4` — Qwen2.5-3B + 2k UltraFeedback\n",
    "\n",
    "> **Before running on Kaggle:** Settings → Accelerator → GPU T4 x2, and turn **Internet on**.\n",
    "> This notebook intentionally uses only GPU 0 so it behaves like the Colab T4 path.\n",
    "\n",
    "> **Reference:** `README.md`, `HARDWARE-GUIDE.md`, and the deck source\n",
    "> `day22/day07-dpo-orpo-alignment-tu-sft-en-preference-learning.tex`.\n",
]

nb["cells"][1]["source"] = [
    "## A. Kaggle setup — install deps + set tier\n",
    "Run these cells first in a Kaggle Notebook. Internet must be enabled.\n",
]

nb["cells"][2]["source"] = [
    "# Set tier early — every downstream cell reads this.\n",
    "import os\n",
    "os.environ[\"COMPUTE_TIER\"] = \"T4\"\n",
    "# Kaggle often exposes two T4s. This lab is written for one GPU; use GPU 0.\n",
    "os.environ.setdefault(\"CUDA_VISIBLE_DEVICES\", \"0\")\n",
    "os.environ.setdefault(\"TOKENIZERS_PARALLELISM\", \"false\")\n",
    "os.environ.setdefault(\"HF_HOME\", \"/kaggle/working/.cache/huggingface\")\n",
    "os.environ.setdefault(\"TRANSFORMERS_CACHE\", \"/kaggle/working/.cache/huggingface/transformers\")\n",
    "print(f\"COMPUTE_TIER set to {os.environ['COMPUTE_TIER']}\")\n",
    "print(f\"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}\")\n",
]

nb["cells"][3]["source"] = [
    "# Install the tested stack. On Kaggle this needs Notebook Internet = On.\n",
    "!pip install -q \\\n",
    "  \"unsloth>=2025.10,<2026.5\" \"trl>=0.12,<0.20\" \"peft>=0.13,<1.0\" \\\n",
    "  \"bitsandbytes>=0.44,<1.0\" \"datasets>=3.1,<4.0\" \"accelerate>=1.1,<2.0\" \\\n",
    "  \"llama-cpp-python>=0.3,<1.0\" \"lm-eval[ifeval,math]>=0.4.5,<1.0\" \\\n",
    "  \"matplotlib>=3.9,<4.0\" \"pandas>=2.2,<3.0\" \"pyarrow>=17,<22\" \\\n",
    "  \"openai>=1.55,<2.0\" \"anthropic>=0.40,<1.0\"\n",
    "\n",
    "# Kaggle/Colab images can include a torchcodec wheel that is incompatible with\n",
    "# the active PyTorch + FFmpeg stack. This lab does not use audio/video codecs,\n",
    "# and removing it prevents sentence_transformers from breaking Unsloth import.\n",
    "!pip uninstall -y -q torchcodec\n",
]

nb["cells"][4]["source"] = [
    "# Probe GPU\n",
    "import torch\n",
    "assert torch.cuda.is_available(), \"Enable Kaggle GPU: Settings → Accelerator → GPU T4 x2\"\n",
    "gpu = torch.cuda.get_device_properties(0)\n",
    "print(f\"Visible CUDA devices: {torch.cuda.device_count()}\")\n",
    "print(f\"GPU: {gpu.name}  ({gpu.total_memory / 1e9:.1f} GB)\")\n",
]

nb["cells"][5]["source"] = [
    "# Set up working directory matching the repo layout — Kaggle writes to /kaggle/working\n",
    "from pathlib import Path\n",
    "WORK = Path(\"/kaggle/working/lab22\")\n",
    "WORK.mkdir(exist_ok=True)\n",
    "(WORK / \"notebooks\").mkdir(exist_ok=True)\n",
    "(WORK / \"data\" / \"pref\").mkdir(parents=True, exist_ok=True)\n",
    "(WORK / \"data\" / \"eval\").mkdir(parents=True, exist_ok=True)\n",
    "(WORK / \"adapters\" / \"sft-mini\").mkdir(parents=True, exist_ok=True)\n",
    "(WORK / \"adapters\" / \"dpo\").mkdir(parents=True, exist_ok=True)\n",
    "(WORK / \"adapters\" / \"merged-fp16\").mkdir(parents=True, exist_ok=True)\n",
    "(WORK / \"gguf\").mkdir(exist_ok=True)\n",
    "(WORK / \"submission\" / \"screenshots\").mkdir(parents=True, exist_ok=True)\n",
    "import os\n",
    "os.chdir(WORK / \"notebooks\")\n",
    "print(f\"Working dir: {Path.cwd()}\")\n",
]

for cell in nb["cells"]:
    if cell.get("cell_type") == "markdown":
        text = "".join(cell.get("source", []))
        text = text.replace("Colab setup", "Kaggle setup")
        text = text.replace("Colab CPU runtime", "Kaggle CPU runtime")
        text = text.replace("Colab T4", "Kaggle T4")
        text = text.replace("free Colab", "Kaggle")
        text = text.replace("Free Colab", "Kaggle")
        cell["source"] = text.splitlines(keepends=True)
    elif cell.get("cell_type") == "code":
        cell["execution_count"] = None
        cell["outputs"] = []

nb["metadata"].pop("colab", None)
nb["metadata"]["kaggle"] = {"accelerator": "gpu"}
nb["metadata"]["accelerator"] = "GPU"

dst.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(dst)
