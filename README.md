# DB-Communicator 💬

A LangChain-based conversational assistant for SQL databases and tabular data, powered by Google Gemini.

This app allows users to:
- Ask questions in plain English about SQL databases or uploaded CSV/XLSX files.
- Get SQL queries auto-generated, executed, and explained.
- Use retrieval-augmented generation (RAG) on tabular data for semantic answers.

---

## 🚀 Features

- 🧾 Natural language to SQL generation (via Gemini LLM)
- 🗃️ Query `.db` files or uploaded `.csv`/`.xlsx`
- 🔍 ChromaDB-powered RAG over tabular data
- 🧠 Gemini models (`gemini-2.0-flash`, `gemini-embedding-exp-03-07`) used for chat + embedding
- 🧑‍💻 Clean UI with Gradio, including file upload and multi-mode selection

---

## 🛠️ Setup

### 1. Clone this repo
```bash
git clone https://github.com/AviN27/DB-Communicator.git
cd DB-Communicator
```

### 2. Provide the required data
- XLSX or CSV files
- .sql or .db files

### 3. Run the Gradio chatbot
```bash
py src/app.py
```

## 📌 Notes
- Currently supports SQLite only.
- Optimized for small- to medium-sized tabular datasets.
- Gemini is used both for generating SQL and for semantic retrieval (RAG).

## 📣 Credits
- Developed with support from Youtube tutorials (Farzad from AI RoundTable)
- [Flaticons](https://www.flaticon.com)
