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


##########################################################################################################
# Done importing packages, start defining and caching helper functions
##########################################################################################################

# Establish a session state to store variables
# st.session_state

API_KEY = 'eca9b8caf9d9d1cc2d15c60df163953d47fe07f684e92d1ac3c1583f6415d382'
extractorApi = ExtractorApi(API_KEY)

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

@st.cache_resource
def get_llm():
  os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
  llm = HuggingFaceHub(repo_id="MBZUAI/LaMini-Flan-T5-783M", model_kwargs={"temperature":0.05, "max_length":512})
  return llm

@st.cache_resource()
def get_E5():
  E5_tokenizer_small = AutoTokenizer.from_pretrained('intfloat/e5-small-v2')
  E5_model_small = AutoModel.from_pretrained('intfloat/e5-small-v2')
  return E5_tokenizer_small, E5_model_small

@st.cache_data(show_spinner=False)
def GetStatementSections(statement_type):
  if statement_type == "10-K":
    tenK_sections = ["1", "1A", "1B", "2", "3", "4", "5", "6", "7", "7A", "8", "9", "9A", "9B", "10", "11", "12", "13", "14", "15"]
    return tenK_sections
  elif statement_type == "10-Q":
    tenQ_sections = ["part1item1", "part1item2", "part1item3", "part1item4", "part2item1", "part2item1a", "part2item2", "part2item3", "part2item4", "part2item5", "part2item6"]
    return tenQ_sections
  elif statement_type == "8-K":
    eightK_sections = ["1-1", "1-2", "1-3", "1-4", "2-1", "2-2", "2-3", "2-4", "2-5", "2-6", "3-1", "3-2", "3-3", "4-1", "4-2", "5-1", "5-2", "5-3", "5-4", "5-5", "5-6", "5-7", "5-8", "6-1", "6-2", "6-3", "6-4", "6-5", "6-6", "6-10", "7-1", "8-1", "9-1", "signature"]
    return eightK_sections
  else:
    return []

@st.cache_data
def retrieve_filing_URLS(ticker, formtype, start_date, end_date):
  # Filter invalid year range
  if start_date >= end_date:
    return 0, 0

  queryApi = QueryApi(api_key='eca9b8caf9d9d1cc2d15c60df163953d47fe07f684e92d1ac3c1583f6415d382')
  ticker_query = 'ticker:({})'.format(ticker)
  query_string = '{ticker_query} AND filedAt:[{start_date} TO {end_date}] AND formType:\"{formtype}\" AND NOT formType:"10-K/A" AND NOT formType:NT'.format(ticker_query=ticker_query, formtype=formtype, start_date=start_date, end_date=end_date)
  query = {
        "query": { "query_string": {
            "query": query_string,
            "time_zone": "America/New_York"
        } },
        "from": "0",
        "size": "250",
        "sort": [{ "filedAt": { "order": "desc" } }]
  }

  response = queryApi.get_filings(query)
  filings = response['filings']

  metadata = list(map(lambda f: {'Ticker': f['ticker'], 'Statement Type': f['formType'], 'Filing Date': f['filedAt'], 'Filing URL': f['linkToFilingDetails']}, filings))
  df = pd.DataFrame.from_records(metadata)
  df_trimmed = df[:5]
  return 1, df_trimmed

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
def RetrieveText(urls, formtype, sections):
  slist = []
  progress_bar = st.progress(0, text="Retrieving report text. Please Wait...")
  count = 0
  percent_complete = 0
  total_count = len(urls) * len(sections)
  percent_update = 1 / total_count
  for url in urls:
    for section in sections:
      count += 1
      sectiontext = extractorApi.get_section(url, section, "text")
      sectiontext_processed = html.unescape(sectiontext)
      if sectiontext_processed != "processing":
        slist.append(sectiontext_processed)
      if count != total_count:
        progress_bar.progress(percent_complete + percent_update, "Retrieving report text. Please Wait...")
      else:
        progress_bar.progress(100, text="Done retrieving text.")
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
  embedding_df = pd.DataFrame(index=input_sentences, columns=['Coord'+str(coord) for coord in range(len(embeddings[0]))], data=embeddings)
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


st.markdown("## 📈 SEC Report Querying Tool")

col1top, col2top = st.columns(2)
with col1top:
    ticker = st.text_input("Please enter the company ticker:")
    st.session_state['ticker'] = ticker

with col2top:
    statement_type = st.selectbox("Select financial statement type:", ["10-K", "10-Q", "8-K"])
    st.session_state['statement_type'] = statement_type

if 'statement_type' not in st.session_state:
  section_list = []
else:
  statement_type = st.session_state['statement_type']
  sections = GetStatementSections(statement_type)
  chosen_sections = st.multiselect('Specify report sections:', sections, default=sections)
  st.session_state['chosen_sections'] = chosen_sections

col1bottom, col2bottom = st.columns(2)
with col1bottom:
    start_date = st.date_input("Start Date (YY-MM-DD):")
    st.session_state['start_date'] = start_date

with col2bottom:
    end_date = st.date_input("End Date (YY-MM-DD):")
    st.session_state['end_date'] = end_date


if st.button('Retrieve SEC reports'):
  if ticker and statement_type and start_date and end_date:
    err, results = retrieve_filing_URLS(ticker, statement_type, start_date, end_date)
    if err == 0:
      st.write("ERROR: Start date must be before end date.")
    else:
      # st.write("Retrieved the Following Report Data from SEC EDGAR:")
      # st.write(results)
      st.session_state['results'] = results
  else:
    st.write("Please fill out all required fields.")


if 'results' in st.session_state:
  st.write("Retrieved the Following Report Data from SEC EDGAR:")
  st.write(st.session_state['results'])

if st.button('Click to Process Report Data for Querying'):
  results = st.session_state['results']
  urls = results["Filing URL"].values
  sections = st.session_state['chosen_sections']
  joined_string = RetrieveText(urls, st.session_state['statement_type'], sections)
  embedding_df = ChunkAndEmbedText(joined_string)
  st.session_state['embedding_df'] = embedding_df

user_query_prompt = st.text_input("Enter your query")
if st.button('Run'):
  if user_query_prompt:
    response = process_query(user_query_prompt, st.session_state['embedding_df'], template, 3)
    st.write(response)
