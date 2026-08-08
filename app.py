import os
import re
import time
from collections import defaultdict
import torch
torch.set_num_threads(2)  # avoid thread contention on shared/limited CPU
import joblib
import numpy as np
import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ---- Config ----
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
EMBEDDINGS_PATH = os.getenv("EMBEDDINGS_PATH", "embeddings_hf.joblib")
SIMILARITY_THRESHOLD = 0.3

app = FastAPI(title="Sigma RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading embedding model (bge-m3)... this happens once at startup.")
embed_model = SentenceTransformer("BAAI/bge-m3")

print("Loading precomputed course embeddings...")
df = joblib.load(EMBEDDINGS_PATH)


class Query(BaseModel):
    question: str

RATE_LIMIT_MAX_REQUESTS = 15
RATE_LIMIT_WINDOW_SECONDS = 60
request_log = defaultdict(list)


def check_rate_limit(client_ip: str):
    now = time.time()
    request_log[client_ip] = [t for t in request_log[client_ip] if now - t < RATE_LIMIT_WINDOW_SECONDS]
    if len(request_log[client_ip]) >= RATE_LIMIT_MAX_REQUESTS:
        raise HTTPException(status_code=429, detail="Too many requests. Please slow down and try again shortly.")
    request_log[client_ip].append(now)


def create_embedding(text):
    return embed_model.encode([text])[0]


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


@app.post("/ask")
def ask(query: Query, request: Request):
    client_ip = request.client.host if request.client else "unknown"
    check_rate_limit(client_ip)

    try:
        question_embedding = create_embedding(query.question)

        similarities = cosine_similarity(
            np.vstack(df['embedding'].values), [question_embedding]
        ).flatten()

        top_results = 6
        max_indx = similarities.argsort()[::-1][0:top_results]
        top_similarities = similarities[max_indx]

        if top_similarities.max() < SIMILARITY_THRESHOLD:
            return {"answer": "I can only answer questions related to the Sigma Web Development course. Please ask something about the course content."}

        new_df = df.iloc[max_indx].copy()
        new_df["start_mmss"] = new_df["start"].apply(format_timestamp)
        new_df["end_mmss"] = new_df["end"].apply(format_timestamp)
        # Construct a direct YouTube link for each chunk using the playlist + video number
        new_df["youtube_url"] = new_df["number"].apply(
            lambda n: f"https://www.youtube.com/watch?list=PLu0W_9lII9agq5TrH9XLIKQvv0iaF2X3w&index={n}"
        )

        prompt = f'''  I am teaching web devlopment using sigma web devlopment course, here are video chunks containing video title , video number , youtube url , start time , end time (both given in readable time format) and the text at that time :

{new_df[["title", "number", "youtube_url", "text", "start_mmss", "end_mmss"]].to_json(orient="records")}
------------------------------

"{query.question}"
User asked this question related to the video chunks. Include EVERY chunk that is genuinely relevant to the question - if multiple different videos cover the topic, mention all of them, not just one.

IMPORTANT: If the SAME video number appears multiple times (multiple relevant timestamps within the same video), combine them into ONE block for that video, and include the link only ONCE at the end of that block. Do not repeat the same video's block or link multiple times. However, do NOT merge the timestamps into one vague combined description - for EACH timestamp range within that video, describe specifically and separately what is taught in THAT exact range, using its own chunk text. Never write one generic summary that mixes multiple timestamp ranges together.

For EACH unique relevant video, format it EXACTLY like this (in plain text, not markdown):

Video No: <number>
Title: <title>
Timestamp: <start_mmss> to <end_mmss> - <specific description of what is taught in exactly this range>
(repeat the "Timestamp: ... - ..." line separately for each additional relevant timestamp range in this same video, each with its own specific description)
Watch here: <youtube_url>

Leave a blank line between each video block. Do not skip the "Video No:" line for any video you mention - it must always be present. Do not mix in unrelated details from chunks that only loosely or partially relate. Always mention timestamps exactly as given (e.g. "6 min 28 sec"), never convert to raw seconds or any other format. If none of the chunks are actually relevant to the question, just say you can only answer questions related to the course - do not use the format above in that case.

'''

        response = inference_llm(prompt)
        response = clean_text(response)
        return {"answer": response}

    except HTTPException:
        raise
    except Exception as e:
        # Log full details server-side for debugging, but never leak
        # internals (stack traces, file paths) to the client.
        print(f"[ERROR] /ask failed: {e}")
        raise HTTPException(status_code=500, detail="Something went wrong processing your question. Please try again.")
