# Financial Document Querying System (FDQS)
A Streamlit-based web application which allows users to query information from SEC reports (10-K, 8-K, 10-Q, etc...) as well as business news articles. 

# About the Project
This project ...

# How to run
Make sure to first run the following installs: 
* !pip install langchain transformers huggingface_hub tiktoken 
* !pip install -q sec-api
* !pip install streamlit newsapi-python newspaper3k

Ensure that "SEC_Report_Analyzer.py" and "News_Article_Analyzer.py" are in the pages folder before running! Folder must be named "pages" in all lowercase for the Streamlit app to work. 

To run the app on Jupyter Notebook / Colab, use the following lines to create a local tunnel. This will create a "logs.txt" file. In "logs.txt", copy 
