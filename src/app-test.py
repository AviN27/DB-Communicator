from utils.load_config import LoadConfig

APPCFG = LoadConfig()

llm = APPCFG.langchain_llm
ai_message = llm.invoke([("system", "You are a test."), ("human", "Hello!")])
print("Gemini reply:", ai_message.content)
