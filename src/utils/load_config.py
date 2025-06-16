
import os
from dotenv import load_dotenv
import yaml
from pyprojroot import here
import shutil
import chromadb

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # if needed here
from langchain_core.prompts import ChatPromptTemplate

print("Environment variables are loaded:", load_dotenv())


class LoadConfig:
    def __init__(self) -> None:
        with open(here("configs/app_config.yaml")) as cfg:
            app_config = yaml.load(cfg, Loader=yaml.FullLoader)

        self.load_directories(app_config=app_config)
        self.load_llm_configs(app_config=app_config)
        self.load_gemini_models()
        self.load_chroma_client()
        self.load_rag_config(app_config=app_config)

        # Un comment the code below if you want to clean up the upload csv SQL DB on every fresh run of the chatbot. (if it exists)
        # self.remove_directory(self.uploaded_files_sqldb_directory)

    def load_directories(self, app_config):
        self.stored_csv_xlsx_directory = here(
            app_config["directories"]["stored_csv_xlsx_directory"])
        self.sqldb_directory = str(here(
            app_config["directories"]["sqldb_directory"]))
        self.uploaded_files_sqldb_directory = str(here(
            app_config["directories"]["uploaded_files_sqldb_directory"]))
        self.stored_csv_xlsx_sqldb_directory = str(here(
            app_config["directories"]["stored_csv_xlsx_sqldb_directory"]))
        self.persist_directory = app_config["directories"]["persist_directory"]

    def load_llm_configs(self, app_config):
        # Read system roles
        self.agent_llm_system_role = app_config["llm_config"]["agent_llm_system_role"]
        self.rag_llm_system_role = app_config["llm_config"]["rag_llm_system_role"]
        # Read engine name for Gemini
        self.engine = app_config["llm_config"].get("engine")
        if not self.engine:
            raise ValueError("llm_config.engine must be set in app_config.yaml")
        # Read temperature
        self.temperature = app_config["llm_config"].get("temperature", 0.0)
        # Embeddings config will be read separately
        self.embeddings_config = app_config.get("embeddings_config", {})

    def load_gemini_models(self):
    # Instantiate Gemini chat LLM
        self.langchain_llm = ChatGoogleGenerativeAI(
            model=self.engine,
            temperature=self.temperature,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )
        # Instantiate embeddings
        emb_model = self.embeddings_config.get("model", "models/gemini-embedding-exp-03-07")
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=emb_model,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )
        print(f"Initializing Gemini LLM with model: {self.engine}, temperature: {self.temperature}")


    def load_chroma_client(self):
        self.chroma_client = chromadb.PersistentClient(
            path=str(here(self.persist_directory)))

    def load_rag_config(self, app_config):
        self.collection_name = app_config["rag_config"]["collection_name"]
        self.top_k = app_config["rag_config"]["top_k"]
        # Reassign embeddings model name if stored in rag_config
        self.embeddings_config = app_config.get("embeddings_config", {})

    def remove_directory(self, directory_path: str):
        """
        Removes the specified directory.

        Parameters:
            directory_path (str): The path of the directory to be removed.

        Raises:
            OSError: If an error occurs during the directory removal process.

        Returns:
            None
        """
        if os.path.exists(directory_path):
            try:
                shutil.rmtree(directory_path)
                print(
                    f"The directory '{directory_path}' has been successfully removed.")
            except OSError as e:
                print(f"Error: {e}")
        else:
            print(f"The directory '{directory_path}' does not exist.")
