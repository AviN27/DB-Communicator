import os
from typing import List, Tuple
from utils.load_config import LoadConfig
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from operator import itemgetter
from sqlalchemy import create_engine
from langchain_community.agent_toolkits import create_sql_agent
import langchain
langchain.debug = True

from langchain_google_genai import ChatGoogleGenerativeAI
import re

APPCFG = LoadConfig()


class ChatBot:
    """
    A ChatBot class capable of responding to messages using different modes of operation.
    It can interact with SQL databases, leverage language chain agents for Q&A,
    and use embeddings for Retrieval-Augmented Generation (RAG) with ChromaDB.
    """
    def extract_sql(response: str) -> str:
        identifiers = [
            "sql", "sqlite", "postgres", "postgresql", "mysql", "mongodb", "mongo"
        ]
        pattern = r"```(?:{}|)\s*\n(.*?)```".format("|".join(identifiers))
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        else:
            return response.strip()    

    @staticmethod
    def respond(chatbot: List, message: str, chat_type: str, app_functionality: str) -> Tuple:
        """
        Respond to a message based on the given chat and application functionality types.

        Args:
            chatbot (List): A list representing the chatbot's conversation history.
            message (str): The user's input message to the chatbot.
            chat_type (str): Describes the type of the chat (interaction with SQL DB or RAG).
            app_functionality (str): Identifies the functionality for which the chatbot is being used (e.g., 'Chat').

        Returns:
            Tuple[str, List, Optional[Any]]: A tuple containing an empty string, the updated chatbot conversation list,
                                             and an optional 'None' value.
        """
        if app_functionality == "Chat":
            # Q&A with stored SQL-DB
            if chat_type == "Q&A with stored SQL-DB":
                if os.path.exists(APPCFG.sqldb_directory):
                    # 1. Connect DB and tools
                    db = SQLDatabase.from_uri(f"sqlite:///{APPCFG.sqldb_directory}")
                    execute_query = QuerySQLDataBaseTool(db=db)
                    write_query = create_sql_query_chain(APPCFG.langchain_llm, db)
                    prompt_template = ChatPromptTemplate.from_messages([
                        ("system", APPCFG.agent_llm_system_role),
                        ("human", "Question: {question}\nSQL Query: {query}\nSQL Result: {result}\nAnswer:")
                    ])
                    clean_sql = RunnableLambda(ChatBot.extract_sql)

                    # 2. Generate raw SQL
                    raw_sql = write_query.invoke({"question": message})
                    # 3. Clean SQL (strip fences, etc.)
                    sql = ChatBot.extract_sql(raw_sql)
                    # 4. Execute SQL
                    result = execute_query.invoke(sql)

                    # 5. Build prompt inputs for final answer
                    prompt_inputs = {
                        "question": message,
                        "query": sql,
                        "result": result
                    }
                    # 6. Invoke LLM to get answer
                    ai_message = (prompt_template | APPCFG.langchain_llm).invoke(prompt_inputs)
                    response = ai_message.content

                    # 7. Append to chat history, include SQL and result for transparency
                    #    First element is user message, second is assistant reply + debug info
                    display_text = f"{response}\n\nSQL Query: {sql}\nSQL Result: {result}"
                    chatbot.append((message, display_text))

                    return "", chatbot

                else:
                    chatbot.append(
                        (message, "SQL DB does not exist. Please first create the 'sqldb.db'."))
                    return "", chatbot, None

            # Q&A with Uploaded or stored CSV/XLSX SQL-DB
            elif chat_type in ("Q&A with Uploaded CSV/XLSX SQL-DB", "Q&A with stored CSV/XLSX SQL-DB"):
                # Determine which DB path to use
                if chat_type == "Q&A with Uploaded CSV/XLSX SQL-DB":
                    if os.path.exists(APPCFG.uploaded_files_sqldb_directory):
                        engine = create_engine(f"sqlite:///{APPCFG.uploaded_files_sqldb_directory}")
                        db = SQLDatabase(engine=engine)
                    else:
                        chatbot.append(
                            (message, "SQL DB from the uploaded csv/xlsx files does not exist. Please first upload the csv files from the chatbot."))
                        return "", chatbot, None
                else:  # "Q&A with stored CSV/XLSX SQL-DB"
                    if os.path.exists(APPCFG.stored_csv_xlsx_sqldb_directory):
                        engine = create_engine(f"sqlite:///{APPCFG.stored_csv_xlsx_sqldb_directory}")
                        db = SQLDatabase(engine=engine)
                    else:
                        chatbot.append(
                            (message, "SQL DB from the stored csv/xlsx files does not exist. Please first execute `src/prepare_csv_xlsx_sqlitedb.py`."))
                        return "", chatbot, None

                # Use create_sql_agent with Gemini-based llm
                agent_executor = create_sql_agent(
                    APPCFG.langchain_llm, db=db, agent_type="openai-tools", verbose=True
                )
                result = agent_executor.invoke({"input": message})
                # agent_executor.invoke returns a dict with 'output'
                response = result.get("output")

            # RAG with stored CSV/XLSX ChromaDB
            elif chat_type == "RAG with stored CSV/XLSX ChromaDB":
                # 1. Embed query using Gemini embeddings
                query_embed = APPCFG.embeddings.embed_query(message)

                # 2. Retrieve top-k documents from Chroma
                vectordb = APPCFG.chroma_client.get_collection(name=APPCFG.collection_name)
                results = vectordb.query(query_embeddings=query_embed, n_results=APPCFG.top_k)

                # 3. Build prompt including system role
                prompt = f"Search results:\n{results}\n\nUser question: {message}"
                ai_message = APPCFG.langchain_llm.invoke([
                    ("system", APPCFG.rag_llm_system_role),
                    ("human", prompt)
                ])
                response = ai_message.content

            else:
                # If chat_type is something else under Chat, you can handle or default
                response = "Unsupported chat type."
            # Append and return
            chatbot.append((message, response))
            return "", chatbot

        # Non-Chat functionality can be handled here if needed
        return "", chatbot, None
