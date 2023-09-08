import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="FDQS",
    page_icon="💷",
)

headerimage = Image.open("newspaper_graphic.jpg")
bloomyu = Image.open("bloomyu.png")

# Centering the header image
col1, col2, col3 = st.columns([1,6,1])
with col1:
  st.write("")
with col2:
  st.image(headerimage, use_column_width="auto")

with col3:
  st.write("")

# Including Landing Page text
st.markdown("## **Financial Document Querying System (FDQS)**")

st.sidebar.title("")

st.markdown(
    """
    FDQS is a framework designed to help simplify the process of financial document analysis. Using the power of vector embeddings and large language models (LLMs),
    you can now upload any type of financial document and then query relevant information from the report by asking questions in a Q&A style format. For example,
    if you want to learn about Apple's most recent annual report without having to read through all 50+ pages, you can use FDQS to query only the information that
    is relevant to you. Simply ask "How many iPhone 12s were sold in 2022?" or "Is Apple planning to release any new products in the coming year?", and FDQS will
    use information from the report to give you an accurate and relevant response.

    As of Summer 2023, FDQS supports major SEC filings (such as 10-K and 10-Q reports) as well as news reports from most major publications.
    We are currently working to extend our infrastructure to handle more report sources and formats.

    **👈 Select a frameowrk version from the sidebar** to choose the FDQS version which supports your desired document type.
"""
)

st.write(" ")
st.write(" ")

st.markdown(
  """
  ## **About the Project**
  FDQS was developed by [Sasha Agapiev](https://www.linkedin.com/in/sasha-agapiev/) and [Tanmaay Kankaria](https://www.linkedin.com/in/tanmaay-kankaria/) in collaboration with Bloomberg L.P and the NYU Department of Financial Engineering.
  Our project methodology is outlined in this [pre-print](https://arxiv.org/).

  If you have any feedback or would like to get in touch, feel free to contact us:
  - aba439@nyu.edu
  - tk2976@nyu.edu
"""
)

st.write(" ")
st.write(" ")
st.write(" ")
st.write(" ")
st.write(" ")
st.write(" ")


col1bottom, col2bottom, col3bottom = st.columns([1,6,1])
with col1bottom:
  st.write("")
with col2bottom:
  st.image(bloomyu)
with col3bottom:
  st.write("")
