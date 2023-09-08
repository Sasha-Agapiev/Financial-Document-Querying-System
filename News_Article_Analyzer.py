import streamlit as st
import pandas as pd
import numpy as np
import os
import datetime
import html
from numpy.linalg import norm
from langchain import HuggingFaceHub, HuggingFacePipeline, PromptTemplate,  LLMChain
from langchain.text_splitter import CharacterTextSplitter
from transformers import AutoTokenizer, AutoModel
from torch import Tensor
from sec_api import QueryApi
from sec_api import ExtractorApi
from newsapi import NewsApiClient
from newspaper import Article


##########################################################################################################
# Done importing packages, start defining and caching helper functions
##########################################################################################################

newsapi = NewsApiClient(api_key='7f27acd08d6a4c30b878ebc59f1e0cae')

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

@st.cache_resource
def get_llm(show_spinner=False):
  os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
  llm = HuggingFaceHub(repo_id="MBZUAI/LaMini-Flan-T5-783M", model_kwargs={"temperature":0.05, "max_length":512})
  return llm

@st.cache_resource(show_spinner=False)
def get_E5():
  E5_tokenizer_small = AutoTokenizer.from_pretrained('intfloat/e5-small-v2')
  E5_model_small = AutoModel.from_pretrained('intfloat/e5-small-v2')
  return E5_tokenizer_small, E5_model_small

@st.cache_data(show_spinner=False)
def LangChainSplitter(input_string, overlap, cs=400):
  text_splitter = CharacterTextSplitter(chunk_size=cs, chunk_overlap=overlap)
  docs = text_splitter.create_documents([input_string])
  langchain_chunks = []
  for chunk in docs:
    chunk_content = chunk.page_content
    langchain_chunks.append(chunk_content)
  return langchain_chunks

@st.cache_data(show_spinner=False)
def CosineSimilarity(embeddingA, embeddingB):
  A = np.array(embeddingA)
  B = np.array(embeddingB)
  A = A.flatten()
  B = B.flatten()
  cos = np.dot(A, B) / (norm(A)*norm(B))
  return cos

@st.cache_data(show_spinner=False)
def getArticles(topic, start_date, end_date, page_size=5):
  all_articles = newsapi.get_everything(topic,
                                        from_param=start_date,
                                        to=end_date,
                                        language='en',
                                        sort_by='relevancy',
                                        page_size=page_size)
  sources = [article['source']['name'] for article in all_articles['articles']]
  authors = [article['author'] for article in all_articles['articles']]
  titles = [article['title'] for article in all_articles['articles']]
  publish_date = [article['publishedAt'] for article in all_articles['articles']]
  urls = [article['url'] for article in all_articles['articles']]

  response_dict = {'Source': sources, 'Author': authors, 'Title': titles, 'Date Published': publish_date, 'Link': urls}
  response_df = pd.DataFrame(response_dict)
  return response_df

@st.cache_data(show_spinner=False)
def getArticleText(response_df):
  links = response_df['Link'].values
  count = 0
  progress_bar = st.progress(0, text="Retrieving news article text. Please wait...")
  percent_complete = 0
  percent_update = 1 / len(links)
  slist = []
  for link in links:
    count += 1
    linkarticle = Article(link)
    linkarticle.download()
    linkarticle.parse()
    article_text = linkarticle.text
    article_text_processed = article_text.replace('Advertisement', '')
    slist.append(article_text_processed)
    if count != len(links):
      progress_bar.progress(percent_complete + percent_update, text="Retrieving news article text. Please wait...")
    else:
      progress_bar.progress(100, text="Done retrieving news article text.")
    percent_complete += percent_update
  joined_string = "\n\n".join(slist)
  return joined_string

@st.cache_data(show_spinner=False)
def GetEmbeddings(input_sentences, embedding_model="E5 Small", verbosity=True):
  E5_tokenizer_small, E5_model_small = get_E5()
  embeddings = []
  count = 0
  progress_bar = st.progress(0, text="Embedding report text. Please Wait...")
  percent_complete = 0
  len_input_sentences = len(input_sentences)
  percent_update = 1 / len_input_sentences
  for sent in input_sentences:
    count += 1
    batch_dict = E5_tokenizer_small([sent], max_length=512, padding=True, truncation=True, return_tensors='pt')
    outputs = E5_model_small(**batch_dict)
    E5embedding = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    E5embedding_np = E5embedding.detach().numpy()
    embedding = E5embedding_np.flatten()
    embeddings.append(embedding)
    if count != len_input_sentences:
      progress_bar.progress(percent_complete + percent_update, text="Embedding report text. Please Wait...")
    else:
      # progress_bar.progress(percent_complete + percent_update, text="Done creating embeddings.")
      progress_bar.progress(100, text="Done creating embeddings.")
    percent_complete += percent_update
    # if verbosity and count % 100 == 0:
      # progress_report = str("⌛ {} embeddings out of {} generated").format(count, len_input_sentences)
      # st.write(progress_report)
  report_success = str("✅ Done creating all {} embeddings! Ready to accept queries.").format(count)
  st.write(report_success)

  # Creating the embedding DataFrame
  i = 0
  cols = []
  while i < len(embeddings[0]):
    colstring = 'Coord' + str(i)
    cols.append(colstring)
    i += 1
  embedding_df = pd.DataFrame(index=input_sentences, columns=cols, data=embeddings)
  # embedding_df = pd.DataFrame(index=input_sentences, columns=['Coord'+str(coord) for coord in range(len(embeddings[0]))], data=embeddings)
  return embeddings, embedding_df

@st.cache_data(show_spinner=False)
def ChunkAndEmbedText(joined_string):
  langchain_chunks = LangChainSplitter(joined_string, 20)
  embedding_list, edf = GetEmbeddings(langchain_chunks, 'E5 Small')
  return edf

@st.cache_data(show_spinner=False)
def EmbeddingRetriever(input_sentence, edf, k=5):
  E5_tokenizer_small, E5_model_small = get_E5()
  batch_dict = E5_tokenizer_small([input_sentence], max_length=512, padding=True, truncation=True, return_tensors='pt')
  outputs = E5_model_small(**batch_dict)
  E5embedding = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
  E5embedding_np = E5embedding.detach().numpy()
  input_embedding = E5embedding_np.flatten()
  cosine_similarities = []
  for index, row in edf.iterrows():
    cos = CosineSimilarity(input_embedding, row.values)
    cosine_similarities.append(cos)
  cosine_df = pd.DataFrame(index=edf.index.values, columns=["Cosine Similarity"], data=cosine_similarities)
  # Sorting the cosine similarity df by top k
  sorted_df = cosine_df.sort_values(by=["Cosine Similarity"], ascending=False)
  topK_sorted = sorted_df[:k]
  topK_sorted_list = topK_sorted.index.values.tolist()
  return topK_sorted_list


@st.cache_data
def process_query(query, embedding_dataframe, prompt_template, num_results_to_query=3):
  llm_model = get_llm()
  similar_embedding_values = EmbeddingRetriever(query, embedding_dataframe, num_results_to_query)
  context_list = [str(text_chunk) for text_chunk in similar_embedding_values]
  QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=prompt_template,)
  chain = LLMChain(llm=llm_model, prompt=QA_CHAIN_PROMPT)
  result = chain.run({
      'context': context_list,
      'question': query
      })
  return result

lamini_llm = get_llm()

template = """Pretend you are a financial analyst.
Use the following context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
{context}
Question: {question}
Answer using complete sentences:"""

##########################################################################################################
# Done defining helper functions, now writing actual Streamlit App
##########################################################################################################

st.markdown("## 📰 News Article Querying Tool")

col1top, col2top = st.columns(2)
with col1top:
    company = st.text_input("Please enter the company name or ticker:")
    st.session_state['company'] = company

with col2top:
    keywords = st.text_input("Enter additional keywords for news articles:")
    st.session_state['keywords'] = keywords

col1mid, col2mid = st.columns(2)
with col1mid:
  range = st.slider("How many days back do you want to search?", 1, 30)
  st.session_state['range'] = range

with col2mid:
  number_of_articles = st.slider("What is the max. number of articles you want to include?", 1, 15)
  st.session_state['num_articles'] = number_of_articles

writestring = "Currently searching for news articles pertaining to: " + st.session_state['company'] + " " + st.session_state['keywords']
writestring2 = "Retrieving articles posted within the past " + str(st.session_state['range']) + " days and including a maximum of " + str(st.session_state['num_articles']) + " articles."

st.write(writestring)
st.write(writestring2)

if st.button("Retrieve Articles"):
  if 'range' in st.session_state and 'num_articles' in st.session_state and 'company' in st.session_state and 'keywords' in st.session_state:
    today = datetime.date.today()
    Days_ago = today - datetime.timedelta(days=st.session_state['range'])
    topic = st.session_state['company'] + " " + st.session_state['keywords']
    response_df = getArticles(topic, today, Days_ago, st.session_state['num_articles'])
    urls = response_df['Link'].values
    st.session_state['response_df'] = response_df
    st.session_state['urls'] = urls
    st.write("Retrieved the following article data:")
    st.write(st.session_state['response_df'])
  else:
    st.write("Please fill out all fields before proceeding.")

if st.button('Click to Process Report Data for Querying'):
  if 'urls' not in st.session_state:
    st.write("Please retrieve news report data before proceeding.")
  else:
    urls = st.session_state['urls']
    if len(urls) == 0:
      st.write("No reports to analyze.")
    else:
      joined_string = getArticleText(st.session_state['response_df'])
      embedding_df = ChunkAndEmbedText(joined_string)
      st.write("debugging...")
      st.session_state['embedding_df'] = embedding_df
      st.write("still debugging...")

user_query_prompt = st.text_input("Enter Your Query")
if st.button('Run'):
  if user_query_prompt:
    if 'embedding_df' not in st.session_state:
      st.write("Please process report text first before proceeding.")
    else:
      response = process_query(user_query_prompt, st.session_state['embedding_df'], template, 3)
      st.write(response)
