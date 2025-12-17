import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA


load_dotenv()  # ensure OPENAI_API_KEY is loaded

# --------------------------
# 1. Load your document
# --------------------------
with open("document.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

docs = [Document(page_content=raw_text)]

# --------------------------
# 2. Split into chunks
# --------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=100
)
chunks = splitter.split_documents(docs)

# --------------------------
# 3. Embed + Store in FAISS
# --------------------------
embeddings = OpenAIEmbeddings()
vector_db = FAISS.from_documents(chunks, embeddings)

retriever = vector_db.as_retriever(search_kwargs={"k": 3})

# Optional test
# docs = retriever.get_relevant_documents("What is RAG?")
# print(docs)

# --------------------------
# 4. Build RAG Chain
# --------------------------
llm = ChatOpenAI(model="gpt-4o-mini")  # choose your model

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# --------------------------
# 5. Ask a Question
# --------------------------
def ask(question: str):
    result = qa.invoke({"query": question})
    print("\nANSWER:\n", result["result"])
    print("\n--- Retrieved Chunks ---")
    for i, d in enumerate(result["source_documents"]):
        print(f"\nChunk {i+1}:\n{d.page_content[:300]}...\n")


# Example usage
if __name__ == "__main__":
    q = input("Ask a question: ")
    ask(q)
