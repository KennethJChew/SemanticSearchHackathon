import streamlit as st

import pandas as pd
from transformers import pipeline

from paperparser import PaperParser


def get_patent_desc(text):
    generator = pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B")
    res = generator(text, max_length=150, do_sample=True, temperature=0.9)
    return res[0]["generated_text"]


parser = PaperParser()
parser.create_corpus("corpus.tsv", file_type="tsv")
parser.get_corpus_embeddings()

st.title("Patent search")
input_patent = st.text_input(
    "Enter a patent abstract:",
    key="input_text",
    value="A garden tool is shown that has the compactness and general shape of a conventional straight shaft weeder. However, certain angles are formed in the shaft to facilitate easier use, particular by those with limited flexibility.",
)
if input_patent:
    res, tfidf = parser.get_cos_sim(input_patent)
    st.write(f"Top {len(res)} closest match patents")
    st.dataframe(
        pd.DataFrame(
            res,
            columns=["Patent abstract", "Cosine similarity"],
        )
    )
    st.write("New patent description")
    # st.write(get_patent_desc(input_patent))
