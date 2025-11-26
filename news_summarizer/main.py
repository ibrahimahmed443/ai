import requests
import trafilatura
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl, model_validator
from typing import Optional
from transformers import pipeline

app = FastAPI(
    title="News Summarizer API",
    version="1.0.0",
    description="Summarizes text or news articles from a URL using Hugging Face models."
)

# Load summarizer model once
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


# ---------------------------
# Request Schema
# ---------------------------
class SummarizeRequest(BaseModel):
    url: Optional[HttpUrl] = None
    text: Optional[str] = None

    @model_validator(mode="after")
    def check_text_or_url(self):
        if not self.url and not self.text:
            raise ValueError("Either 'url' or 'text' is required.")
        return self


# ---------------------------
# Summary Function
# ---------------------------
def extract_text_from_url(url: str) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        )
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
    except Exception:
        raise HTTPException(400, detail="Unable to fetch URL.")

    if response.status_code != 200:
        raise HTTPException(400, detail="Unable to fetch URL.")

    # Feed raw HTML into Trafilatura
    text = trafilatura.extract(response.text)
    if not text:
        raise HTTPException(400, detail="Unable to extract article content.")

    return text


# ---------------------------
# API Route
# ---------------------------
@app.post("/summarize")
def summarize(req: SummarizeRequest):
    # If URL provided, extract article text
    if req.url:
        article_text = extract_text_from_url(req.url)
    else:
        article_text = req.text

    if len(article_text.split()) < 10:
        raise HTTPException(400, detail="Text too short to summarize.")

    summary = summarizer(
        article_text,
        max_length=300,
        min_length=100,
        do_sample=False
    )[0]["summary_text"]

    return {
        "summary": summary,
        "source": req.url if req.url else "text"
    }


# ---------------------------
# Health Check
# ---------------------------
@app.get("/")
def root():
    return {"message": "News Summarizer API is running!"}


"""
Run app: uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Usage:
1.
curl -X POST "http://localhost:8000/summarize" \
     -H "Content-Type: application/json" \
     -d '{
           "url": "https://www.bbc.com/news/articles/cj4w44w42j5o"
         }'

2.
curl -X POST "http://localhost:8000/summarize" \
     -H "Content-Type: application/json" \
     -d '{
           "text": "Apple announced new AI features today, focusing on on-device processing..."
         }'

"""
