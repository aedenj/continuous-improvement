
## Mini Project 1 - Part 1: Getting Familiar with Word Embeddings.
# This assignment introduces students to text similarity measures using cosine similarity and sentence embeddings.
# Students will implement and compare different methods for computing and analyzing text similarity using GloVe and Sentence Transformers.

#Learning Objectives
#By the end of this assignment, students will:
#Understand how cosine similarity is used to measure text similarity.
#Learn to encode sentences using GloVe embeddings and Sentence Transformers.
#Compare the performance of different embedding techniques.
#Create a Web interface for your model

# Context: In this part, you are going to play around with some commonly used pretrained text embeddings for text search. For example, GloVe is an unsupervised learning algorithm for obtaining vector representations for words. Pretrained on
# 2 billion tweets with vocabulary size of 1.2 million. Download from [Stanford NLP](http://nlp.stanford.edu/data/glove.twitter.27B.zip).
# Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. *GloVe: Global Vectors for Word Representation*.

### Import necessary libraries: here you will use streamlit library to run a text search demo, please make sure to install it.
import re
import streamlit as st
import numpy as np
import numpy.linalg as la
import pickle
import os
import gdown
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import math



### Some predefined utility functions for you to load the text embeddings

# Function to Load Glove Embeddings
def load_glove_embeddings(glove_path="Data/embeddings.pkl"):
    with open(glove_path, "rb") as f:
        embeddings_dict = pickle.load(f, encoding="latin1")

    return embeddings_dict


def get_model_id_gdrive(model_type):
    if model_type == "25d":
        word_index_id = "13qMXs3-oB9C6kfSRMwbAtzda9xuAUtt8"
        embeddings_id = "1-RXcfBvWyE-Av3ZHLcyJVsps0RYRRr_2"
    elif model_type == "50d":
        embeddings_id = "1DBaVpJsitQ1qxtUvV1Kz7ThDc3az16kZ"
        word_index_id = "1rB4ksHyHZ9skes-fJHMa2Z8J1Qa7awQ9"
    elif model_type == "100d":
        word_index_id = "1-oWV0LqG3fmrozRZ7WB1jzeTJHRUI3mq"
        embeddings_id = "1SRHfX130_6Znz7zbdfqboKosz-PfNvNp"

    return word_index_id, embeddings_id


def download_glove_embeddings_gdrive(model_type):
    # Get glove embeddings from google drive
    word_index_id, embeddings_id = get_model_id_gdrive(model_type)

    # Use gdown to get files from google drive
    embeddings_temp = "embeddings_" + str(model_type) + "_temp.npy"
    word_index_temp = "word_index_dict_" + str(model_type) + "_temp.pkl"

    # Download word_index pickle file
    print("Downloading word index dictionary....\n")
    gdown.download(id=word_index_id, output=word_index_temp, quiet=False)

    # Download embeddings numpy file
    print("Donwloading embedings...\n\n")
    gdown.download(id=embeddings_id, output=embeddings_temp, quiet=False)


# @st.cache_data()
def load_glove_embeddings_gdrive(model_type):
    word_index_temp = "word_index_dict_" + str(model_type) + "_temp.pkl"
    embeddings_temp = "embeddings_" + str(model_type) + "_temp.npy"

    # Load word index dictionary
    word_index_dict = pickle.load(open(word_index_temp, "rb"), encoding="latin")

    # Load embeddings numpy
    embeddings = np.load(embeddings_temp)

    return word_index_dict, embeddings


@st.cache_resource()
def load_sentence_transformer_model(model_name):
    sentenceTransformer = SentenceTransformer(model_name)
    return sentenceTransformer


def get_sentence_transformer_embeddings(sentence, model_name="all-MiniLM-L6-v2"):
    """
    Get sentence transformer embeddings for a sentence
    """
    # 384 dimensional embedding
    # Default model: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

    sentenceTransformer = load_sentence_transformer_model(model_name)

    try:
        return sentenceTransformer.encode(sentence)
    except:
        if model_name == "all-MiniLM-L6-v2":
            return np.zeros(384)
        else:
            return np.zeros(512)


def get_glove_embeddings(word, word_index_dict, embeddings, model_type):
    """
    Get glove embedding for a single word
    """
    if word.lower() in word_index_dict:
        return embeddings[word_index_dict[word.lower()]]
    else:
        return np.zeros(int(model_type.split("d")[0]))


def get_category_embeddings(embeddings_metadata):
    """
    Get embeddings for each category
    1. Split categories into words
    2. Get embeddings for each word
    """
    model_name = embeddings_metadata["model_name"]
    st.session_state["cat_embed_" + model_name] = {}
    for category in st.session_state.categories.strip().split(" "):
        if model_name:
            if not category in st.session_state["cat_embed_" + model_name]:
                st.session_state["cat_embed_" + model_name][category] = get_sentence_transformer_embeddings(category, model_name=model_name)
        else:
            if not category in st.session_state["cat_embed_" + model_name]:
                st.session_state["cat_embed_" + model_name][category] = get_sentence_transformer_embeddings(category)


def update_category_embeddings(embeddings_metadata):
    """
    Update embeddings for each category
    """
    get_category_embeddings(embeddings_metadata)




### Plotting utility functions

def plot_piechart(sorted_cosine_scores_items):
    sorted_cosine_scores = np.array([
            sorted_cosine_scores_items[index][1]
            for index in range(len(sorted_cosine_scores_items))
        ]
    )
    categories = st.session_state.categories.split(" ")
    categories_sorted = [
        categories[sorted_cosine_scores_items[index][0]]
        for index in range(len(sorted_cosine_scores_items))
    ]
    fig, ax = plt.subplots()
    ax.pie(sorted_cosine_scores, labels=categories_sorted, autopct="%1.1f%%")
    st.pyplot(fig)  # Figure


def plot_piechart_helper(sorted_cosine_scores_items):
    sorted_cosine_scores = np.array([s[1] for s in sorted_cosine_scores_items])
    categories_sorted = [s[0] for s in sorted_cosine_scores_items]

    fig, ax = plt.subplots(figsize=(3, 3))
    my_explode = np.zeros(len(categories_sorted))
    my_explode[0] = 0.2
    if len(categories_sorted) == 3:
        my_explode[1] = 0.1  # explode this by 0.2
    elif len(categories_sorted) > 3:
        my_explode[2] = 0.05
    ax.pie(
        sorted_cosine_scores,
        labels=categories_sorted,
        autopct="%1.1f%%",
        explode=my_explode,
    )

    return fig


def plot_piecharts(sorted_cosine_scores_models):
    scores_list = []
    categories = st.session_state.categories.split(" ")
    index = 0
    for model in sorted_cosine_scores_models:
        scores_list.append(sorted_cosine_scores_models[model])
        # scores_list[index] = np.array([scores_list[index][ind2][1] for ind2 in range(len(scores_list[index]))])
        index += 1

    if len(sorted_cosine_scores_models) == 2:
        fig, (ax1, ax2) = plt.subplots(2)

        categories_sorted = [
            categories[scores_list[0][index][0]] for index in range(len(scores_list[0]))
        ]
        sorted_scores = np.array(
            [scores_list[0][index][1] for index in range(len(scores_list[0]))]
        )
        ax1.pie(sorted_scores, labels=categories_sorted, autopct="%1.1f%%")

        categories_sorted = [
            categories[scores_list[1][index][0]] for index in range(len(scores_list[1]))
        ]
        sorted_scores = np.array(
            [scores_list[1][index][1] for index in range(len(scores_list[1]))]
        )
        ax2.pie(sorted_scores, labels=categories_sorted, autopct="%1.1f%%")

    st.pyplot(fig)


def plot_alatirchart(sorted_cosine_scores_models):
    models = list(sorted_cosine_scores_models.keys())
    tabs = st.tabs(models)
    figs = {}
    for model in models:
        figs[model] = plot_piechart_helper(sorted_cosine_scores_models[model])

    for index in range(len(tabs)):
        with tabs[index]:
            st.pyplot(figs[models[index]])


### Your Part To Complete: Follow the instructions in each function below to complete the similarity calculation between text embeddings

# Task I: Compute Cosine Similarity
def cosine_similarity(x, y):
    """
    Exponentiated cosine similarity
    1. Compute cosine similarity
    2. Exponentiate cosine similarity
    3. Return exponentiated cosine similarity
    (20 pts)
    """

    cs = np.dot(x, y) / (la.norm(x) * la.norm(y))

    return np.exp(cs)

# Task II: Average Glove Embedding Calculation
def averaged_glove_embeddings_gdrive(sentence, word_index_dict, embeddings, model_type=50):
    """
    Get averaged glove embeddings for a sentence
    1. Split sentence into words
    2. Get embeddings for each word
    3. Add embeddings for each word
    4. Divide by number of words
    5. Return averaged embeddings
    (30 pts)
    """

    words = re.findall(r"\w+", sentence)

    embedding = np.zeros(int(model_type.split("d")[0]))
    for w in words:
        word_embedding = get_glove_embeddings(w, word_index_dict, embeddings, model_type)
        embedding = embedding + word_embedding

    return embedding / len(words)


# Task III: Sort the cosine similarity
def get_sorted_cosine_similarity(search_text, embeddings_metadata):
    """
    Get sorted cosine similarity between input sentence and categories
    Steps:
    1. Get embeddings for input sentence
    2. Get embeddings for categories (if not found, update category embeddings)
    3. Compute cosine similarity between input sentence and categories
    4. Sort cosine similarity
    5. Return sorted cosine similarity
    (50 pts)
    """

    categories = st.session_state.categories.strip().split(" ")
    cosine_sim = {}
    if embeddings_metadata["embedding_model"] == "glove":
        word_index_dict = embeddings_metadata["word_index_dict"]
        embeddings = embeddings_metadata["embeddings"]
        model_type = embeddings_metadata["model_type"]

        input_embedding = averaged_glove_embeddings_gdrive(st.session_state.text_search,
                                                            word_index_dict,
                                                            embeddings, model_type)


        for c in categories:
            cat_embed = get_glove_embeddings(c, word_index_dict, embeddings, model_type)
            cosine_sim[c] = cosine_similarity(input_embedding, cat_embed)

    else:
        model_name = embeddings_metadata["model_name"]
        if not "cat_embed_" + model_name in st.session_state:
            get_category_embeddings(embeddings_metadata)

        category_embeddings = st.session_state["cat_embed_" + model_name]

        print("text_search = ", st.session_state.text_search)
        if model_name:
            input_embedding = get_sentence_transformer_embeddings(st.session_state.text_search, model_name=model_name)
        else:
            input_embedding = get_sentence_transformer_embeddings(st.session_state.text_search)

        for c in categories:
            if c not in category_embeddings:
                update_category_embeddings(embeddings_metadata)
                category_embeddings = st.session_state["cat_embed_" + model_name]
                print(f"UPDATED CATEGORIES: {category_embeddings}")

            cosine_sim[c] = cosine_similarity(input_embedding, category_embeddings[c])


    return sorted(cosine_sim.items(), key=lambda x: x[1])


### Below is the main function, creating the app demo for text search engine using the text embeddings.

if __name__ == "__main__":
    ### Text Search ###
    ### There will be Bonus marks of 10% for the teams that submit a URL for your deployed web app.
    ### Bonus: You can also submit a publicly accessible link to the deployed web app.

    st.sidebar.title("GloVe Twitter")
    st.sidebar.markdown(
        """
    GloVe is an unsupervised learning algorithm for obtaining vector representations for words. Pretrained on
    2 billion tweets with vocabulary size of 1.2 million. Download from [Stanford NLP](http://nlp.stanford.edu/data/glove.twitter.27B.zip).

    Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. *GloVe: Global Vectors for Word Representation*.
    """
    )

    model_type = st.sidebar.selectbox("Choose the model", ("25d", "50d", "100d"), index=1)


    st.title("Search Based Retrieval Demo")
    st.subheader(
        "Pass in space separated categories you want this search demo to be about."
    )
    # st.selectbox(label="Pick the categories you want this search demo to be about...",
    # options=("Flowers Colors Cars Weather Food", "Chocolate Milk", "Anger Joy Sad Frustration Worry Happiness", "Positive Negative"),
    # key="categories"
    # )
    st.text_input(
        label="Categories", key="categories", value="Flowers Colors Cars Weather Food"
    )
    print(st.session_state["categories"])
    print(type(st.session_state["categories"]))
    # print("Categories = ", categories)
    # st.session_state.categories = categories

    st.subheader("Pass in an input word or even a sentence")
    text_search = st.text_input(
        label="Input your sentence",
        key="text_search",
        value="Roses are red, trucks are blue, and Seattle is grey right now",
    )
    # st.session_state.text_search = text_search

    # Download glove embeddings if it doesn't exist
    embeddings_path = "embeddings_" + str(model_type) + "_temp.npy"
    word_index_dict_path = "word_index_dict_" + str(model_type) + "_temp.pkl"
    if not os.path.isfile(embeddings_path) or not os.path.isfile(word_index_dict_path):
        print("Model type = ", model_type)
        glove_path = "Data/glove_" + str(model_type) + ".pkl"
        print("glove_path = ", glove_path)

        # Download embeddings from google drive
        with st.spinner("Downloading glove embeddings..."):
            download_glove_embeddings_gdrive(model_type)


    # Load glove embeddings
    word_index_dict, embeddings = load_glove_embeddings_gdrive(model_type)


    # Find closest word to an input word
    if st.session_state.text_search:
        # Glove embeddings
        print("Glove Embedding")
        embeddings_metadata = {
            "embedding_model": "glove",
            "word_index_dict": word_index_dict,
            "embeddings": embeddings,
            "model_type": model_type,
        }
        with st.spinner("Obtaining Cosine similarity for Glove..."):
            sorted_cosine_sim_glove = get_sorted_cosine_similarity(
                st.session_state.text_search, embeddings_metadata
            )

        # Sentence transformer embeddings
        print("Sentence Transformer Embedding")
        embeddings_metadata = {"embedding_model": "transformers", "model_name": ""}
        with st.spinner("Obtaining Cosine similarity for 384d sentence transformer..."):
            sorted_cosine_sim_transformer = get_sorted_cosine_similarity(
                st.session_state.text_search, embeddings_metadata
            )

        # Results and Plot Pie Chart for Glove
        print("Categories are: ", st.session_state.categories)
        st.subheader(
            "Closest word I have between: "
            + st.session_state.categories
            + " as per different Embeddings"
        )

        print(sorted_cosine_sim_glove)
        print(sorted_cosine_sim_transformer)
        # print(sorted_distilbert)
        # Altair Chart for all models
        plot_alatirchart(
            {
                "glove_" + str(model_type): sorted_cosine_sim_glove,
                "sentence_transformer_384": sorted_cosine_sim_transformer,
            }
        )
        # "distilbert_512": sorted_distilbert})

        st.write("")
        st.write(
            "Demo developed by [Your Name](https://www.linkedin.com/in/your_id/ - Optional)"
        )
