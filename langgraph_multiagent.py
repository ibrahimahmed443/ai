import os
import requests
import trafilatura
from typing import TypedDict
from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END

load_dotenv()

class ResearchState(TypedDict):
    topic: str
    research: str
    summary: str
    article: str

llm = OpenAI(model="gpt-4o-mini", temperature=0.2)


def serper_search(query):

    url = "https://google.serper.dev/search"
    payload = {"q": query}

    headers = {
        "X-API-KEY": os.environ["SERPER_API_KEY"],
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)
    results = response.json()
    articles = []

    for r in results.get("organic", [])[:3]:
        link = r["link"]
        downloaded = trafilatura.fetch_url(link)

        if downloaded:
            text = trafilatura.extract(downloaded)
            if text:
                # limit article size
                words = text.split()
                start = words[:400]
                middle = words[len(words)//2 : len(words)//2 + 200]
                end = words[-200:]

                limited_text = " ".join(start + middle + end)
                articles.append(limited_text)

    return "\n\n".join(articles)


### Agents
def researcher(state: ResearchState):
    topic = state["topic"]
    research_notes = serper_search(topic)

    return {
        "research": research_notes
    }

def summarizer(state: ResearchState):
    research = state["research"]
    prompt = f"Summarize the following research notes into clear bullet points:\n\n{research}"

    response = llm.invoke(prompt)

    return {
        "summary": response
    }

def writer(state: ResearchState):
    summary = state["summary"]

    prompt = f"Write a blog article based on the summary below:\n\n{summary}"
    response = llm.invoke(prompt)

    return {
        "article": response
    }

### Build Graph
workflow = StateGraph(ResearchState)
workflow.add_node("researcher", researcher)
workflow.add_node("summarizer", summarizer)
workflow.add_node("writer", writer)

workflow.set_entry_point("researcher")

workflow.add_edge("researcher", "summarizer")
workflow.add_edge("summarizer", "writer")
workflow.add_edge("writer", END)

app = workflow.compile()

result = app.invoke({
    "topic": "Role of AI in finance"
})

print(result["article"])
