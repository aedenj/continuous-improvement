''' EE P 596 Mini Project Part 2 Streamlit App Chatbot by the GloVetrotters'''

import os
import streamlit as st
from openai import OpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings


# Define a function to get the conversation history (Not required for Part-2, will be useful in Part-3)
def get_conversation() -> str:
    # return: A formatted string representation of the conversation.
    conversation = ""
    for message in st.session_state.messages:
        conversation += "Role: " + str(message["role"]) + "Content: " + str(message["content"] + ". ")

    return conversation



class Head_Agent:
    def __init__(self, openai_key, pinecone_key, pinecone_index_name) -> None:
        self._openai_key = openai_key
        self._pinecone_key = pinecone_key
        self._pinecone_index_name = pinecone_index_name
        self._client = OpenAI(api_key=openai_key)

        os.environ['PINECONE_API_KEY'] = pinecone_key
        os.environ['OPENAI_API_KEY'] = openai_key

        # Check for existing session state variables
        if "openai_model" not in st.session_state:
            st.session_state.openai_model = 'gpt-3.5-turbo-0125'

        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display existing chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    def setup_sub_agents(self):
        # TODO: Setup the sub-agents
        self._query_agent = Query_Agent(self._pinecone_index_name, self._client, OpenAIEmbeddings(model="text-embedding-3-small"))
        self._answering_agent = Answering_Agent(self._client, st.session_state.openai_model)


    def main_loop(self):
        # TODO: Run the main loop for the chatbot
        if prompt := st.chat_input("What would you like to chat about?"):

            # ... (append user message to messages)
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("user"):
                st.markdown(prompt)

            relevant_docs = self._query_agent.query_vector_store(prompt)

            # Generate AI response
            with st.chat_message("assistant"):
                # ... (send request to OpenAI API)
                message = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
                #stream = self._client.chat.completions.create(model=st.session_state.openai_model, messages=message, stream=True)

                answer_stream = self._answering_agent.generate_response(prompt, relevant_docs, message)

                # ... (get AI response and display it)
                response = st.write_stream(answer_stream)

            # ... (append AI response to messages)
            st.session_state.messages.append({"role": "assistant", "content": response})



class Query_Agent:
    def __init__(self, pinecone_index, openai_client, embeddings) -> None:
        # TODO: Initialize the Query_Agent agent
        self._client = openai_client
        self._vectorstore = PineconeVectorStore(index_name=pinecone_index, embedding=embeddings)

    def query_vector_store(self, query, k=5):
        # TODO: Query the Pinecone vector store
        return self._vectorstore.similarity_search(query=query, k=k, namespace='ns768x80')

    def set_prompt(self, prompt):
        # TODO: Set the prompt for the Query_Agent agent
        pass

    def extract_action(self, response, query = None):
        # TODO: Extract the action from the response
        pass


class Answering_Agent:
    def __init__(self, openai_client, model) -> None:
        # TODO: Initialize the Answering_Agent
        self._client = openai_client
        self._model = model

    def generate_response(self, query, docs, conv_history, k=5):
        # TODO: Generate a response to the user's query
        conv_history[-1]['content'] = self._create_prompt(query, docs)

        return self._client.chat.completions.create(
          model=self._model,
          messages=conv_history,
          stream=True,
        )


    def _create_prompt(self, query, docs) -> str:
        paragraphs = [f"{i+1}. {d.page_content.replace(chr(10), ' ')}" for i, d in enumerate(docs)]

        prompt = f"""
        Consider the following {len(docs)} paragraphs numbered paragraphs:

        {f"{chr(10)}{chr(10)}".join(paragraphs)}

        Use only the five paragraphs above to answer the following question: {query}

        DO NOT ANSWER irrelevant questions with respect to the {len(docs)} numbered paragraphs above. IF AND ONLY IF the question is irrelevant, respond with the text,
        This question is not relevant to the context of this book. I would be happy to answer the question based on the books context.<|endofprompt|>

        The answer to your question is,
        """

        return prompt



if __name__ == "__main__":
    st.title("GloVetrotters Mini Project 2: Streamlit Chatbot")

    head_agent = Head_Agent(openai_key, pinecone_key, 'uw-glovetrotters')
    head_agent.setup_sub_agents()
    head_agent.main_loop()
