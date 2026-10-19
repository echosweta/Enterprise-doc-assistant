"""Chat UI. Run: streamlit run ui/streamlit_app.py  (API must be running on :8000)"""
import os

import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")
st.set_page_config(page_title="Company Assistant", page_icon="📄")
st.title("Enterprise Document Assistant")
role = st.sidebar.selectbox("Simulated role", ["employee", "hr", "it"])
st.sidebar.caption("In production the role comes from SSO, not a dropdown.")

if "history" not in st.session_state:
    st.session_state.history = []

for i, m in enumerate(st.session_state.history):
    with st.chat_message(m["role"]):
        st.write(m["content"])
        if m["role"] == "assistant" and m.get("meta"):
            meta = m["meta"]
            st.caption(f"confidence {meta['confidence']} · {meta['latency_ms']} ms")
            for c in meta.get("citations", []):
                with st.expander(f"[{c['n']}] {c['title']} › {c['section']} (v{c['version']})"):
                    st.write(c["snippet"] + "…")
            a, b, _ = st.columns([1, 1, 6])
            for label, val, col in (("👍", True, a), ("👎", False, b)):
                if col.button(label, key=f"fb{i}{label}"):
                    requests.post(f"{API}/feedback", json={"question": m["question"], "answer": m["content"], "helpful": val})
                    st.toast("Thanks for the feedback")

if q := st.chat_input("Ask about HR, IT, compliance or SOPs…"):
    st.session_state.history.append({"role": "user", "content": q})
    r = requests.post(f"{API}/ask", json={"question": q, "role": role}, timeout=60).json()
    st.session_state.history.append({"role": "assistant", "content": r["answer"], "meta": r, "question": q})
    st.rerun()
