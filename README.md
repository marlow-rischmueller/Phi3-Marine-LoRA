# LoRA Fine-Tuning of Microsoft Phi-3 for Marine Engineering Applications

> Master's Thesis · M.Sc. Management Information Systems · University of Bremen · 2025

Fine-tuning an open-source large language model for a specialised scientific domain while remaining deployable on consumer-grade hardware (< 8 GB VRAM).

---

## Motivation

General-purpose LLMs perform well on broad benchmarks but often fall short in highly specialised domains such as marine engineering and oceanography, where precise terminology and domain-specific reasoning matter. At the same time, deploying large models in resource-constrained environments - research labs, offshore vessels, embedded systems - remains a practical challenge.

This project investigates whether **Low-Rank Adaptation (LoRA)** can bridge both gaps: specialising Microsoft Phi-3-mini for marine engineering knowledge while keeping inference feasible on consumer hardware (< 8 GB VRAM).

A second research strand explores adapting the model to **xVal continuous numerical tokenisation** ([Golkar et al., 2024](https://arxiv.org/abs/2310.02989)) to improve the handling of numerical data - relevant for engineering calculations and oceanographic measurements.

---

## Methodology

### 1 · Domain Adaptation via LoRA Fine-Tuning

| Step | Detail                                                                                                                                 |
|------|----------------------------------------------------------------------------------------------------------------------------------------|
| Base model | `microsoft/Phi-3-mini-128k-instruct`                                                                                                   |
| Fine-tuning method | LoRA (Low-Rank Adaptation) via PEFT                                                                                                    |
| Training objective | Supervised Fine-Tuning (SFT)                                                                                                           |
| Training dataset | [OceanInstruct](https://huggingface.co/datasets/zjunlp/OceanInstruct-v0.1) - 10,000 entries covering marine engineering & oceanography |
| Evaluation dataset | [OceanBench](https://huggingface.co/datasets/zjunlp/OceanBench) - domain-specific benchmark                                            |
| General benchmark | MMLU (5-shot) - to monitor catastrophic forgetting                                                                                     |
| Hardware (training) | NVIDIA RTX 3090 Ti (24 GB VRAM)                                                                                                        |
| Hardware (inference) | Simulated consumer hardware target: < 8 GB VRAM                                                                                        |

LoRA hyperparameters were initialised following the original Phi-3 repository and tuned iteratively. The training and evaluation pipelines are fully contained in the provided notebooks.

### 2 · xVal Numerical Tokenisation (Experimental)

The standard tokeniser treats numbers as sequences of digit tokens, losing numerical magnitude information. The xVal approach encodes numbers as a single scalar value alongside a special token, enabling the model to reason more directly over continuous quantities. This strand adapts the Phi-3 architecture and tokeniser to support xVal encoding and evaluates the approach on mathematical reasoning benchmarks (GSM8K, custom math dataset).

---

## Results

### Domain Specialisation

| Metric | Baseline Phi-3 | Fine-tuned Phi-3 | Δ       |
|--------|----------------|-----------------|---------|
| METEOR (OceanBench) | 0.300          | **0.348**       | + 0.048 |
| BLEU (OceanBench) | 0.134          | **0.176**       | + 0.042 |
| ROUGE (OceanBench) | 0.304          | **0.347**       | + 0.043 |

**Key finding:** The fine-tuned model gains domain knowledge without displacing general capabilities - no catastrophic forgetting on MMLU. Reproducibility of evaluation results was confirmed via a 10-run repeatability test with fixed seeds.


---

## Repository Structure

```
.
├── Phi3_Training.ipynb               # Main fine-tuning pipeline
├── Phi3_Evaluation.ipynb             # Evaluation on OceanBench & MMLU
├── Custom_Phi3_xVal_Training.ipynb   # xVal-adapted model training
├── Custom_Phi3_xVal_Evaluation.ipynb # xVal-adapted model evaluation
├── phi_chat.py                       # Interactive CLI chat with any checkpoint
├── custom_utils.py                   # Custom model classes, data collators, train/eval loops
├── custom_tokenizer/                 # Manually modified tokenizer files (xVal extension)
├── xval/                             # xVal implementation (Golkar et al., 2024)
│                                     # Original: https://github.com/PolymathicAI/xVal
├── results/                          # Saved evaluation results
└── reqs.txt                          # Dependencies
```

---

## Technologies

| Category | Libraries / Tools |
|----------|---------------|
| Deep Learning | PyTorch, CUDA |
| LLM & Fine-Tuning | HuggingFace Transformers, PEFT, TRL |
| Evaluation | `evaluate`, METEOR / BLEU / ROUGE / Accuracy |
| Data | HuggingFace `datasets`, pandas |
| Utilities | NumPy, tqdm, accelerate, flash-attn |

---

## Setup & Usage

### 1 · Installation

```bash
git clone https://github.com/marlow-rischmueller/Phi3-Marine-LoRA.git
cd Phi3-Marine-LoRA

pip install -r reqs.txt
```

> **Note:** `flash-attn` requires a CUDA-capable GPU. For CPU-only testing, remove it from `reqs.txt` and set `attn_implementation="eager"` in the notebooks.

### 2 · Fine-Tuning (`Phi3_Training.ipynb`)

Open the notebook and set:
- `model_id` - base model path or HuggingFace model ID (e.g. `microsoft/Phi-3-mini-128k-instruct`)
- `new_model` - output name for the fine-tuned checkpoint
- `output_dir` - local path for saving checkpoints

Adjust `per_device_train_batch_size` and `gradient_accumulation_steps` to match your available VRAM.

### 3 · Evaluation (`Phi3_Evaluation.ipynb`)

Runs evaluation on OceanBench (METEOR, BLEU, ROUGE) and MMLU (5-shot accuracy) for both baseline and fine-tuned model. Batch sizes are pre-configured for an RTX 3090 Ti - adjust as needed.

### 4 · Interactive Chat (`phi_chat.py`)

```bash
python phi_chat.py
```

You will be prompted for:
1. A model checkpoint path (local) or HuggingFace model ID (e.g. `microsoft/Phi-3-mini-128k-instruct`)
2. An optional system message

Type `exit` to quit the chat loop. Adjust `max_new_tokens` in the script to control response length.

### 5 · xVal Experiments

Use `Custom_Phi3_xVal_Training.ipynb` and `Custom_Phi3_xVal_Evaluation.ipynb` in sequence. The modified tokenizer files in `custom_tokenizer/` are required and loaded automatically.

---

## Citation & Acknowledgements

**xVal numerical tokenisation:**
```
Golkar et al. (2024). xVal: A Continuous Number Encoding for Large Language Models.
https://arxiv.org/abs/2310.02989 | https://github.com/PolymathicAI/xVal
```

**Datasets:**
- OceanInstruct / OceanBench: [zjunlp on HuggingFace](https://huggingface.co/zjunlp)

---

## License

This project is licensed under the [MIT License](LICENSE).  
Note: Code in `xval/` is derived from [PolymathicAI/xVal](https://github.com/PolymathicAI/xVal) and is subject to its original License.
