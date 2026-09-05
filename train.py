"""
Tiny decoder-only LM (~1M params) — CPU-trainable from scratch.
Architecture matches your diagram: RMSNorm -> GQA(+RoPE) -> residual ->
RMSNorm -> FFN(GELU) -> residual, stacked N times, final RMSNorm, tied head.

Why the sizes are what they are:
  A 768-dim/10-layer model (your diagram) is ~100M+ params. For a genuine
  ~1M-param model, the embedding table has to be tiny too, so this script
  uses BYTE-LEVEL tokenization (vocab=256) instead of GPT-2 BPE (vocab~50k).
  A 50k-vocab embedding alone would cost 50k*dim params -- bigger than your
  whole 1M budget. Byte-level means: no tokenizer to train, works on any
  text out of the box, slightly less efficient than BPE, fully fine for a
  1M/20M-token model.

Hardware notes for your machine (Ryzen 5000 octa-core, 16GB RAM, no CUDA GPU):
  - Train on CPU. PyTorch will use all 8 cores automatically for matmuls.
  - 4GB Radeon iGPU is NOT usable by PyTorch (no ROCm on Windows laptops
    like this) -- don't bother trying to send tensors to a GPU device.
  - 20M tokens x ~1 epoch at this model size should take on the order of
    1-3 hours on CPU. Reduce MAX_STEPS to test quickly first.
"""

import math, os, time, urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

# ----------------------------- CONFIG ---------------------------------
DIM          = 128     # embedding / residual stream width
N_LAYERS     = 4
N_Q_HEADS    = 4
N_KV_HEADS   = 2        # GQA: KV heads shared across groups of Q heads
HEAD_DIM     = 32
FFN_HIDDEN   = 512      # ~4x DIM, GELU MLP
VOCAB_SIZE   = 256      # byte-level tokenizer
SEQ_LEN      = 256
BATCH_SIZE   = 32
LR           = 3e-4
MAX_STEPS    = 6000     # ~6000*32*256 ≈ 49M token-slots seen (with reuse); adjust to your data size
WARMUP_STEPS = 200
GRAD_CLIP    = 1.0
DEVICE       = "cpu"
CORPUS_PATH  = "corpus.txt"
CKPT_PATH    = "model.pt"

# ----------------------- DATA (byte-level) ------------------------------
def ensure_corpus():
    """If you don't have your own 20M-token text file yet, grab a small
    public-domain sample so the script runs end-to-end. REPLACE THIS with
    your own corpus.txt for real training -- point CORPUS_PATH at it."""
    if os.path.exists(CORPUS_PATH):
        return
    print("No corpus.txt found -- downloading a small sample (tiny shakespeare) "
          "so you can test the pipeline. Replace corpus.txt with your real ~20M-token dataset.")
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    urllib.request.urlretrieve(url, CORPUS_PATH)

def load_data():
    ensure_corpus()
    with open(CORPUS_PATH, "rb") as f:
        data = f.read()
    ids = torch.tensor(list(data), dtype=torch.long)
    n = len(ids)
    split = int(n * 0.95)
    print(f"Corpus: {n:,} bytes/tokens  (train {split:,} / val {n - split:,})")
    return ids[:split], ids[split:]

def get_batch(data, batch_size=BATCH_SIZE, seq_len=SEQ_LEN):
    ix = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[i:i + seq_len] for i in ix])
    y = torch.stack([data[i + 1:i + seq_len + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

# ----------------------------- MODEL ------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


def rope_freqs(head_dim, seq_len, device):
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)                      # (seq_len, head_dim/2)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin):
    # x: (B, H, T, head_dim)
    x1, x2 = x[..., ::2], x[..., 1::2]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    rot_x1 = x1 * cos - x2 * sin
    rot_x2 = x1 * sin + x2 * cos
    out = torch.stack([rot_x1, rot_x2], dim=-1).flatten(-2)
    return out


class GQAAttention(nn.Module):
    def __init__(self, dim, n_q_heads, n_kv_heads, head_dim):
        super().__init__()
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.group_size = n_q_heads // n_kv_heads

        self.q_proj = nn.Linear(dim, n_q_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_q_heads * head_dim, dim, bias=False)

    def forward(self, x, cos, sin, mask):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_q_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # repeat KV heads to match Q heads (grouped-query attention)
        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        out = att @ v                                       # (B, H, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = RMSNorm(DIM)
        self.attn = GQAAttention(DIM, N_Q_HEADS, N_KV_HEADS, HEAD_DIM)
        self.norm2 = RMSNorm(DIM)
        self.ffn = FeedForward(DIM, FFN_HIDDEN)

    def forward(self, x, cos, sin, mask):
        x = x + self.attn(self.norm1(x), cos, sin, mask)
        x = x + self.ffn(self.norm2(x))
        return x


class TinyDecoderLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, DIM)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(N_LAYERS)])
        self.final_norm = RMSNorm(DIM)
        self.head = nn.Linear(DIM, VOCAB_SIZE, bias=False)
        self.head.weight = self.tok_emb.weight  # tied weights

        cos, sin = rope_freqs(HEAD_DIM, SEQ_LEN, DEVICE)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        mask = torch.triu(torch.ones(SEQ_LEN, SEQ_LEN, dtype=torch.bool), diagonal=1)
        self.register_buffer("mask", mask, persistent=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        # PyTorch's default init is too large here (esp. with tied embedding/
        # head weights), which makes logits explode and loss blow up to 100+.
        # Small, GPT-style init keeps the model numerically sane from step 1.
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx)
        cos, sin, mask = self.cos[:T], self.sin[:T], self.mask[:T, :T]
        for layer in self.layers:
            x = layer(x, cos, sin, mask)
        x = self.final_norm(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=200, temperature=0.8):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -SEQ_LEN:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx

# ------------------------------ TRAIN -----------------------------------
def lr_at(step):
    if step < WARMUP_STEPS:
        return LR * step / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(1, MAX_STEPS - WARMUP_STEPS)
    return 0.5 * LR * (1 + math.cos(math.pi * progress))


def main():
    train_data, val_data = load_data()
    model = TinyDecoderLM().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    t0 = time.time()
    for step in range(1, MAX_STEPS + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)

        x, y = get_batch(train_data)
        logits, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()

        if step % 100 == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                vx, vy = get_batch(val_data)
                _, vloss = model(vx, vy)
            model.train()
            elapsed = time.time() - t0
            print(f"step {step:5d} | train loss {loss.item():.3f} | val loss {vloss.item():.3f} "
                  f"| lr {opt.param_groups[0]['lr']:.2e} | {elapsed:.0f}s")

        if step % 1000 == 0:
            torch.save(model.state_dict(), CKPT_PATH)

    torch.save(model.state_dict(), CKPT_PATH)
    print(f"Saved checkpoint to {CKPT_PATH}")

    # quick sample
    model.eval()
    start = torch.zeros((1, 1), dtype=torch.long)  # byte 0 as seed
    out = model.generate(start, max_new_tokens=300)
    text = bytes(out[0].tolist()).decode("utf-8", errors="replace")
    print("\n--- sample generation ---\n" + text)


if __name__ == "__main__":
    main()
