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

        # NOTE: change "url" below if your df's video-URL column has a different name
        # (e.g. "link" or "youtube_url").
        URL_COLUMN = "url"

        prompt = f'''  I am teaching web devlopment using sigma web devlopment course, here are video chunks containing video title , video number , start time , end time (both given in minutes:seconds format) and the text at that time :

{new_df[["title", "number", "text", "start_mmss", "end_mmss"]].to_json(orient="records")}
------------------------------

"{query.question}"
User asked this question related to the video chunks above. Write a clear, human, conversational answer as flowing plain paragraphs (dont mention the above format, its just for you, and dont use markdown tables, headers, or labels like "Video No:" or "Timestamp:").

Focus ONLY on the chunk(s) that are genuinely relevant to the question - ignore chunks that only loosely or partially relate.

If a video has multiple relevant timestamp ranges, do NOT merge them into one vague combined description. Instead, describe each timestamp range separately and specifically - say what is actually taught in THAT exact range, using its own chunk text, not a generic summary that mixes ranges together.

Always name the video (its number and title) and mention its timestamp(s) exactly as given (e.g. "6 min 28 sec"), never convert to raw seconds or any other format.

Do NOT include any links or URLs anywhere in your answer - just the explanation text. Links will be added separately.

If none of the chunks are actually relevant to the question, simply tell the user you can only answer questions related to the course.
'''

        response = inference_llm(prompt)
        response = clean_text(response)

        # Attach real video URLs ourselves (never LLM-generated -> no hallucination risk).
        # Only for the unique videos that were actually relevant (above threshold).
        relevant = new_df[top_similarities >= SIMILARITY_THRESHOLD]
        if URL_COLUMN in relevant.columns:
            unique_videos = relevant.drop_duplicates(subset=["number"]).sort_values("number")
            links = []
            for _, row in unique_videos.iterrows():
                if row.get(URL_COLUMN):
                    links.append((row["title"], row[URL_COLUMN]))

            if len(links) == 1:
                response = f"{response}\n\nYou can watch it here: {links[0][1]}"
            elif len(links) > 1:
                link_lines = "\n".join(f"{title}: {url}" for title, url in links)
                response = f"{response}\n\nYou can watch these here:\n{link_lines}"

        return {"answer": response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))