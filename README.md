# Financial Document Querying System (FDQS)
This is a Streamlit-based web application which allows users to query information from SEC reports (10-K, 8-K, 10-Q, etc...) as well as business news articles. Users can fetch relevent financial reports or news articles by filling out the fields on the web app, then our FDQS framework will process the report / article text by generating text chunk embeddings. Using this embedding context along with prompt constraints, we augment a [Lamini GPT LLM](https://huggingface.co/MBZUAI/LaMini-GPT-1.5B) to answer user queries using only information contained in the report / article. Our goal is to minimize hallucinations and non-explainability when using LLMs for financial document analysis. FDQS offers several advantages over leading standard LLMs such as ChatGPT and Bard for financial analysis purposes, because these standard models are not trained on recent financial data and they do not possess much knowledge of low level company-specific information. 

## About the Project
This project was developped by Sasha Agapiev and Tanmaay Kankaria as part of an NYU research project done in conjunction with the Bloomberg Data Department. This is still a work in progress, so if there are any bugs or other areas for improvement please let us know.

## How to run
Make sure to first run the following installs: 
* !pip install langchain transformers huggingface_hub tiktoken 
* !pip install -q sec-api
* !pip install streamlit newsapi-python newspaper3k

Ensure that "SEC_Report_Analyzer.py" and "News_Article_Analyzer.py" are in the "pages" folder before running! Folder must be named "pages" in all lowercase for the Streamlit app to work. 

**To run the app on Jupyter Notebook / Colab**, use the following lines to create a local tunnel:
!npm install localtunnel
!streamlit run /content/Landing_Page.py &>/content/logs.txt &
!npx localtunnel --port 8501


This will create a "logs.txt" file (shown in the image below). In "logs.txt", copy the External URL IP address (excluding :8501). Then, paste this IP on the local tunnel page and this will direct you to the web app landing page. 
![image](https://github.com/Sasha-Agapiev/Financial-Document-Querying-System/assets/57875787/d5d8d965-a063-4e46-aa57-78002359c2ae)

**To run from a standard IDE**, just run the following line:
streamlit run Landing_Page.py 

If this does not work, refer to the instructions from the [Streamlit website](https://docs.streamlit.io/knowledge-base/using-streamlit/how-do-i-run-my-streamlit-script). 
