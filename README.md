# module-Agentic-RAG

How to get started? 

1. Clone the repo to your local

2. Update pip & uv -
        
        pip install -U pip uv

3. Create a venv & install requirements -
        
        uv venv
        uv pip install -r requirements.txt

4. Create a local .env file to configure all the required parameters from example.env
        NOTE - Some of these keys like Azure, NGC, etc needs to be updated.

5. Make sure the qdrant database it setup for the module module.
    Use crawl_web_docs.py to crawl a sitemap url and create a qdrant collection for specified module.
    Make sure to edit the file before executing the commands. Replace "module" with the module you want to create.

6. Once you have updated the .env file, and the SQL db has been setup, run this command to fetch & embed all the web documentation urls for the module Module -
        
        uv run crawl_web_docs.py

    (Use the read_module_files.py for local data files)

7. Run the streamlit server & chatbot -


        uv run streamlit run streamlit_ui.py
