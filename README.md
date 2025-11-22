# **README.md** for Transformer

```markdown
# Transformer from Scratch (PyTorch)

This is a **small, clean implementation of the original Transformer** in PyTorch, built step-by-step for learning purposes.  
It includes a toy dataset, full model, and training script. Perfect for understanding **attention, encoder-decoder layers, and multi-head attention**.

---

## 🧱 Project Structure

```

Transformer/
│
├── model/
│   ├── **init**.py       # Expose Transformer class
│   ├── attention.py      # Scaled dot-product and multi-head attention
│   ├── layers.py         # Encoder/Decoder layers + FeedForward
│   └── transformer.py    # Full Transformer model
│
├── dataset.py            # Toy dataset for sequence copying
├── train.py              # Training script
└── utils.py              # Save/load model helpers

````

---

## ⚡ Features

- Original Transformer architecture
- Multi-head self-attention
- Encoder and Decoder layers
- Toy dataset for sequence copying
- Easy to extend for real tasks (translation, GPT, etc.)

---

## 🛠 Installation

```bash
# Clone the repo
git clone https://github.com/akashpoptani/Transformer.git
cd Transformer

# Load Python + PyTorch (example HPC)
ml python/3.10.4
ml pytorch/2.0.1
````

No other dependencies are needed for the toy dataset example.

---

## 🎓 How it works

* **dataset.py** → Creates random sequences to teach the Transformer to copy sequences.
* **model/** → Contains all layers:

  * `attention.py` → Scaled dot-product and multi-head attention
  * `layers.py` → Encoder/Decoder layers + FeedForward network
  * `transformer.py` → Combines all layers into a full Transformer
* **train.py** → Trains the Transformer on the toy dataset.

---

## 🚀 Training

```bash
python train.py
```

This will train the model on the toy dataset.
Key parameters you can adjust:

```python
VOCAB_SIZE = 100       # number of unique tokens
SEQ_LENGTH = 10        # length of each sequence
BATCH_SIZE = 32        # batch size
```

---

## 📝 Usage

Once trained, you can save the model:

```python
from utils import save_model, load_model

save_model(model, "transformer.pt")
load_model(model, "transformer.pt")
```

---

## 📌 Next Steps / Extensions

* Add sinusoidal positional encoding
* Implement autoregressive decoding for inference
* Train on a real dataset (e.g., EN→DE translation)
* Extend to GPT-style model

---

## 📚 References

* [Attention is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
* PyTorch documentation: [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)

```