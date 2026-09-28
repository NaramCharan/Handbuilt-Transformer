# 🧠 Handbuilt Transformer

**`Deep Learning · Transformers · PyTorch · NLP · From Scratch`**

A decoder-only Transformer built from scratch in PyTorch and trained on Shakespeare's works to generate Shakespeare-style text.

The goal of this project was to understand how GPT-style Transformers work by implementing the core architecture manually instead of using a pre-built Transformer.

---

## ✨ Highlights

- 🧠 Decoder-only Transformer built from scratch
- 🔍 Causal self-attention
- 🧩 Multi-head attention
- 🔄 Residual connections and LayerNorm
- 📝 Character-level language modeling
- 🎲 Autoregressive text generation
- 🔥 Implemented using PyTorch

---

## 🏗️ Architecture

```text
Character Tokens
       ↓
Token + Positional Embeddings
       ↓
Transformer Blocks
       ├── Multi-Head Self-Attention
       ├── Feed-Forward Network
       ├── LayerNorm
       └── Residual Connections
       ↓
Linear Layer
       ↓
Next Character Prediction
```

---

## 📜 Dataset

The model was trained on Shakespeare's works using character-level tokenization. It learns to predict the next character based on the previous context.

```text
Context → Transformer → Next Character
```

---

## 📊 Results

The model can generate text that loosely resembles Shakespeare's writing style, learning patterns in vocabulary, words, and dialogue.

It is not intended to reproduce Shakespeare perfectly — the project focuses on understanding the Transformer architecture.

---

## 💡 Key Learnings

- How Query, Key, and Value work in self-attention
- How causal masking prevents future-token access
- How multiple attention heads work together
- How Transformer blocks use residual connections
- How autoregressive generation works
- How GPT-style models predict the next token

---

## ⚠️ Limitations

- Character-level tokenization
- Small model compared with modern LLMs
- Generated text only loosely resembles Shakespeare

---

## 🔮 Future Roadmap

- [ ] Add temperature and top-k sampling
- [ ] Experiment with larger models
- [ ] Train on larger datasets

---

## 🙏 Acknowledgements

All thanks to **Andrej Karpathy**. This project was inspired by his work and his "Let's build GPT: from scratch, in code, spelled out" lecture, which made the inner workings of Transformers approachable.
