import streamlit as st
import torch
from train import TinyDecoderLM

st.set_page_config(page_title="TinyStories LM", page_icon="🧸")
st.title("🧸 TinyStories LM (0.75M params)")

@st.cache_resource
def load_model():
    model = TinyDecoderLM()
    state = torch.load("model.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model

model = load_model()

def encode(text):
    return torch.tensor([list(text.encode("utf-8"))], dtype=torch.long)

def decode(ids):
    return bytes(ids).decode("utf-8", errors="replace")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

def stream_generate(model, idx, max_new_tokens, temperature):
    """Yields text incrementally as tokens are generated, for st.write_stream.
    Buffers raw bytes so multi-byte UTF-8 characters don't get split mid-stream."""
    generated_bytes = bytearray()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -256:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_id], dim=1)

            generated_bytes.append(next_id.item())
            try:
                # only yield once we have a clean, fully-decodable chunk
                text = generated_bytes.decode("utf-8")
                yield text
                generated_bytes.clear()
            except UnicodeDecodeError:
                continue  # wait for more bytes to complete the character

with st.sidebar:
    max_new_tokens = st.slider("Max new tokens", 50, 1000, 400, step=50)
    temperature = st.slider("Temperature", 0.1, 1.5, 1.0, step=0.1)

prompt = st.chat_input("Say something...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    idx = encode(prompt)
    with st.chat_message("assistant"):
        reply = st.write_stream(stream_generate(model, idx, max_new_tokens, temperature))

    st.session_state.messages.append({"role": "assistant", "content": reply})
