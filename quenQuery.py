# ultra_simple_chat.py
import streamlit as st
from llama_cpp import Llama
import time

# Hardcoded model path
MODEL_PATH = r"C:\Models\qwen-2.5.1-coder-it\Qwen2.5.1-Coder-7B-Instruct-Q4_K_M.gguf"


# Page config
st.set_page_config(page_title="Qwen Chat", page_icon="💬", layout="centered")

# Title
st.title("💬 Qwen2.5 Coder Chat")
st.write("Chat with your local Qwen model")

# Initialize
if 'model' not in st.session_state:
    with st.spinner("Loading Qwen model (30-60 seconds)..."):
        st.session_state.model = Llama(
            model_path=MODEL_PATH,
            n_ctx=2048,
            n_threads=4,
            verbose=False
        )
    st.success("✅ Model loaded!")

if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
if prompt := st.chat_input("Type your message..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Show user message
    with st.chat_message("user"):
        st.write(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Create prompt from history
            history = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])
            full_prompt = f"{history}\nassistant:"
            
            # Generate
            response = st.session_state.model(
                full_prompt,
                max_tokens=512,
                temperature=0.7,
                stop=["</s>", "user:", "assistant:"]
            )
            
            answer = response['choices'][0]['text'].strip()
            st.write(answer)
            
            # Add to history
            st.session_state.messages.append({"role": "assistant", "content": answer})