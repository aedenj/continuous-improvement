''' EE P 596 Mini Project Part 2 Streamlit App Chatbot by the GloVetrotters'''

import streamlit as st
from openai import OpenAI

st.title("GloVetrotters Mini Project 2: Streamlit Chatbot")

openai_key = 'key'
client = OpenAI(api_key=openai_key)

# Define a function to get the conversation history (Not required for Part-2, will be useful in Part-3)
def get_conversation() -> str:
    # return: A formatted string representation of the conversation.
    conversation = ""
    for message in st.session_state.messages:
        conversation += "Role: " + str(message["role"]) + "Content: " + str(message["content"] + ". ")

    return conversation

# Check for existing session state variables
if "openai_model" not in st.session_state:
    st.session_state.openai_model = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Wait for user input
if prompt := st.chat_input("What would you like to chat about?"):

    # ... (append user message to messages)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate AI response
    with st.chat_message("assistant"):
        # ... (send request to OpenAI API)
        message = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
        stream = client.chat.completions.create(model=st.session_state.openai_model, messages=message, stream=True)

        # ... (get AI response and display it)
        response = st.write_stream(stream)

    # ... (append AI response to messages)
    st.session_state.messages.append({"role": "assistant", "content": response})
