import streamlit as st

import pandas as pd
# import torch
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
    value="",
)
if input_patent:
    res = parser.get_cos_sim(input_patent)
    st.write(f"Top {len(res)} closest match patents")
    # st.write(tfidf)
    st.dataframe(
        pd.DataFrame(
            res,
            columns=["Patent abstract", "Cosine similarity", "Important Words"],
        )
    )
    # st.write("New patent description")
    if st.button("Generate sample patent application"):
        st.markdown("""Title: Compact Garden Tool with Angled Shaft for Improved Usability
Abstract: A garden tool is disclosed having a compact, straight shaft with angled portions to facilitate easier use, particularly by individuals with limited flexibility. The tool is shaped like a conventional straight shaft weeder, but includes angled portions in the shaft to allow for a more ergonomic grip and use of the tool.

Detailed Description: The present invention relates to a garden tool that is compact and easy to use, particularly for individuals with limited flexibility. The garden tool is shaped like a conventional straight shaft weeder, but includes angled portions in the shaft to allow for a more ergonomic grip and use of the tool.

The angled portions of the shaft are designed to allow for a more comfortable and efficient grip on the tool, reducing strain on the user's wrist and hand. The compact size of the tool also allows for easy storage and transport.

In one embodiment, the garden tool includes a handle at the top of the shaft, and the angled portions are located near the handle. In another embodiment, the angled portions may be located further down the shaft, closer to the tool head.

The tool head may be any suitable type of garden tool, such as a weeder, hoe, rake, or shovel. The tool head is attached to the shaft via a suitable fastening mechanism, such as a screw or bolt.

The garden tool of the present invention is particularly useful for individuals with limited flexibility or mobility, as the angled portions of the shaft allow for a more comfortable and efficient grip on the tool. The compact size of the tool also makes it easy to store and transport.""")
        # st.write(get_patent_desc(input_patent))
    else:
        pass
    # st.write(get_patent_desc(input_patent))
