from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv
import asyncio
import os

load_dotenv()

# Create a RAG tool using LlamaIndex
documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()

async def search_documents(query: str) -> str:
    """Useful for answering natural language questions about an personal essay written by Paul Graham."""
    response = await query_engine.aquery(query)
    return str(response)

agent = FunctionAgent(
    llm=OpenAI(temperature=0),
    tools=[search_documents],
    system_prompt="You are a helpful assistant that can search for relevant documents based on a query"
)

async def main():
    response = await agent.run("What did the author do in college?")
    print(response)

if __name__ == "__main__":
    asyncio.run(main())

