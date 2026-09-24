import requests
import streamlit as st


st.title("Simple AI Assistant")

question = st.text_input("Ask a question")

if st.button("Ask"):

    response = requests.post(
        "https://ai-customer-assistance-3.onrender.com/chat",
        json={"question": question},
    )

    answer = response.json()["answer"]

    st.write(answer)