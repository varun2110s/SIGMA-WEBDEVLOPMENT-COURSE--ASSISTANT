import os
import re
import time
import joblib
import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity

# ---- Config (env vars, no hardcoded secrets) ----
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
OLLAMA_EMBED_URL = os.getenv("OLLAMA_EMBED_URL", "http://host.docker.internal:11434/api/embed")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "bge-m3")
EMBEDDINGS_PATH = os.getenv("EMBEDDINGS_PATH", "embeddings.joblib")

app = FastAPI(title="Sigma RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load embeddings once at startup, not per-request
df = joblib.load(EMBEDDINGS_PATH)


class Query(BaseModel):
    question: str


def create_embedding(text_list):
    r = requests.post(OLLAMA_EMBED_URL, json={
        "model": OLLAMA_EMBED_MODEL,
        "input": text_list
    }, timeout=30)
    r.raise_for_status()
    return r.json()["embeddings"]


def inference_llm(prompt, max_retries=3):
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": GROQ_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7
                },
                timeout=30
            )
            break
        except requests.exceptions.ConnectionError as e:
            if attempt == max_retries:
                raise RuntimeError(f"Could not reach Groq API: {e}")
            time.sleep(3)
        except requests.exceptions.Timeout:
            if attempt == max_retries:
                raise RuntimeError("Groq API timed out repeatedly.")
            time.sleep(3)

    response = r.json()
    if "error" in response:
        raise RuntimeError(response["error"])
    return response["choices"][0]["message"]["content"]


def clean_text(text):
    replacements = {
        "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "-",
        "\u2011": "-", "\u2212": "-",
        "\u2022": "-",
        "\u25aa": "-", "\u25ab": "-",
        "\u25a0": "-", "\u25a1": "-",
        "\u2026": "...",
        "\u202f": " ", "\u00a0": " ",
        "\u2192": "->",
        "\ufe0f": "", "\u200b": "",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)

    emoji_pattern = re.compile(
        "["
        "\U0001F300-\U0001FAFF"
        "\U00002600-\U000026FF"
        "\U00002700-\U000027BF"
        "\U0001F1E6-\U0001F1FF"
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub("", text)


def format_timestamp(seconds):
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if hours > 0:
        return f"{hours} hr {minutes} min {secs} sec"
    return f"{minutes} min {secs} sec"


@app.get("/health")
def health():
    return {"status": "ok"}


SIMILARITY_THRESHOLD = 0.3  # tune this based on testing with your embedding model


@app.post("/ask")
def ask(query: Query):
    try:
        question_embedding = create_embedding([query.question])[0]

        similarities = cosine_similarity(
            np.vstack(df['embedding'].values), [question_embedding]
        ).flatten()

        top_results = 8
        max_indx = similarities.argsort()[::-1][0:top_results]
        top_similarities = similarities[max_indx]

        # If even the best match is weak, skip the LLM call entirely
        if top_similarities.max() < SIMILARITY_THRESHOLD:
            return {"answer": "I can only answer questions related to the Sigma Web Development course. Please ask something about the course content."}

        new_df = df.iloc[max_indx].copy()
        new_df["start_mmss"] = new_df["start"].apply(format_timestamp)
        new_df["end_mmss"] = new_df["end"].apply(format_timestamp)

        prompt = f'''  I am teaching web devlopment using sigma web devlopment course, here are video chunks containing video title , video number , start time , end time (both given in minutes:seconds format) and the text at that time :

{new_df[["title", "number", "text", "start_mmss", "end_mmss"]].to_json(orient="records")}
------------------------------

"{query.question}"
User asked this question related to the video chunks. Answer in a clear, human, conversational way (dont mention the above format, its just for you). Focus ONLY on the chunk(s) that are most directly relevant to the question - ignore chunks that only loosely or partially relate. Do not mix in unrelated details from other chunks just because they were retrieved. Keep the answer concise and to the point: name the video and the exact timestamp(s), and briefly describe what is taught there. Always mention timestamps exactly as given (e.g. "6 min 28 sec"), never convert to raw seconds or any other format. If none of the chunks are actually relevant to the question, tell the user you can only answer questions related to the course.

'''

        response = inference_llm(prompt)
        response = clean_text(response)
        return {"answer": response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))