import streamlit as st
from typing import List, Tuple

import cohere
import pandas as pd
from numpy import dot
from numpy.linalg import norm
from transformers import pipeline


class PaperParser:
    def __init__(self, corpus_text=None, model="small") -> None:
        """
        Model choices are:-
        1. large - length of embeddings per token is 4096
        2. small - length of embeddings per token is 1024
        3. multilingual-22-12
        """
        self.model = model
        self.models = ["small", "large", "multilingual-22-12"]
        self.client = cohere.Client("7lUDtMMSa1bVCEVIKEOms0jPImRnselfUQucOH5v")
        self.corpus = None
        # self.corpus_embeddings = co.embed(texts=corpus_text,model=model).embeddings

    def create_corpus(self, datafile, file_type="csv"):
        if file_type != "csv":
            data = pd.read_csv(datafile, sep="\t")
        else:
            data = pd.read_csv(datafile)
        corpus = {}
        corpus_records = data.to_dict(orient="records")

        for idx, record in enumerate(corpus_records):
            corpus[idx] = record
        self.corpus = corpus

    def get_corpus_embeddings(self):
        # indices are as follows:
        # 0 : dictionary index
        # 1 : Title
        # 2 : Language
        # 3 : Abstract
        # 4 : URL
        corpus_texts = []
        for idx in self.corpus:
            corpus_texts.append(self.corpus[idx]["ABSTRACT"])
        if len(corpus_texts) > 16:
            pass
        corpus_embeddings = self.client.embed(
            texts=corpus_texts, model=self.model
        ).embeddings
        for idx, embedding in enumerate(corpus_embeddings):
            self.corpus[idx]["EMBEDDING"] = embedding

    def get_cos_sim(self, text: str) -> List[Tuple[str, float]]:
        """Return cosine similarity scores (sorted in descending order) of corpus documents with input text.

        Args:
            text (str): Input text to be compared against corpus.

        Returns:
            List[Tuple[str, float]]: Corpus documents and cosine similarity scores, sorted in descending order.
        """
        # Get embedding
        input_text = [text]
        text_embeddings = self.client.embed(
            texts=input_text, model=self.model
        ).embeddings
        # Get cosine similarities
        res = []
        for record in self.corpus.items():
            cos_sim = dot(text_embeddings, record[1]["EMBEDDING"]) / (
                norm(text_embeddings) * norm(record[1]["EMBEDDING"])
            )
            res.append((record[1]["TITLE"], float(cos_sim)))
        res.sort(key=lambda a: a[1], reverse=True)

        return res


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
    res = parser.get_cos_sim(input_patent)
    st.write(f"Top {len(res)} closest match patents")
    st.dataframe(
        pd.DataFrame(
            res,
            columns=["Patent abstract", "Cosine similarity"],
        )
    )
    st.write("New patent description")
    # st.write(get_patent_desc(input_patent))
