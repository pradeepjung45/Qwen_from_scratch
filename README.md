# 🔥 Qwen_from_scratch: Building a Modern LLM from the Ground Up

> 🧠 A fully implemented small-scale language model inspired by **Qwen3**, built **from scratch** using PyTorch.  
> Every component — from **GQA** to **RoPE**, **SwiGLU**, **Muon optimizer**, and **weight tying** — is explained and implemented in pure code.

🎯 **Purpose**: Learn how real large language models work — not just use them.

🚀 Trained on 500K tokens | 📊 Final Perplexity: **3.05** | 💾 Only **32M parameters**

🔗 Inspired by Qwen3 architecture | ✅ Educational focus | 🚀 Production-ready training loop

---

## 🌟 Features & Modern Techniques

This repo implements **state-of-the-art components** used in today’s top LLMs:

| Feature | Description |
|--------|-------------|
| ✅ **Grouped-Query Attention (GQA)** | Shared KV heads → faster inference, less memory |
| ✅ **Rotary Position Embeddings (RoPE)** | Relative position encoding, works on longer sequences |
| ✅ **SwiGLU Feed-Forward** | More expressive than ReLU, used in LLaMA-3, Qwen |
| ✅ **RMSNorm** | Faster, simpler alternative to LayerNorm |
| ✅ **Muon Optimizer** | Momentum + orthogonalization via Newton-Schulz iteration |
| ✅ **Hybrid Optimization** | Muon for 2D weights, AdamW for embeddings/norms |
| ✅ **Automatic Mixed Precision (AMP)** | Train with `bfloat16` → 2x speed |
| ✅ **Gradient Accumulation** | Simulate larger batch sizes |
| ✅ **Weight Tying** | Share input/output embeddings → better generalization |
| ✅ **Text Generation** | With temperature, top-k, top-p sampling |

Perfect for learning, research, or prototyping.

---

## 📦 Model Architecture Summary

| Parameter | Value |
|---------|-------|
| **Model Size (`d_model`)** | 384 |
| **Number of Layers (`n_layers`)** | 6 |
| **Attention Heads (`n_heads`)** | 8 |
| **KV Heads (GQA)** | 4 |
| **FFN Hidden Dim (`d_ff`)** | 1536 |
| **Max Sequence Length** | 512 |
| **Vocab Size** | ~50k (SmolLM tokenizer) |
| **Total Parameters** | 32.15M |
| **Trained On** | 500K tokens from SmolLM corpus |
| **Training Steps** | 2000 |

Despite its tiny size, it learns meaningful language patterns.

---

## 📊 Training Results

After **2000 steps** (~12.6 minutes on Tesla T4), the model achieved excellent performance:

| Metric | Final Value |
|-------|-------------|
| **Validation Loss** | 1.1146 |
| **Validation Accuracy** | 75.12% |
| **Validation Perplexity** | **3.05** ✅ |

> 🔍 Best model saved at step 1500 with PPL = 6.67

These are **outstanding results** for a 32M-parameter model trained on limited data.

---

## 🚀 How to Run

### 1. Clone the Repo
```bash
git clone https://github.com/pradeepjung45/Qwen_from_scratch.git
cd Qwen_from_scratch
