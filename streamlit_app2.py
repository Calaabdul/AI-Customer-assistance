import requests
import streamlit as st


st.title("Simple AI Assistant")

question = st.text_input("Ask a question")

if st.button("Ask"):

    response = requests.post(
        "http://127.0.0.1:8000/chat",
        json={"question": question},
    )

    answer = response.json()["answer"]

    st.write(answer)