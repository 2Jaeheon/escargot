#!/usr/bin/env python3
import os
import json
import requests
from fastapi import FastAPI, Request
import uvicorn

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL = os.getenv("OLLAMA_MODEL", "llama3")

app = FastAPI()

@app.post("/review")
async def review_code(req: Request):
    """
    Receives: { "diff": "<git diff contents>" }
    Returns: array of inline review comments:
    [
      {"path": "src/BuiltinIterator.cpp", "position": 12, "body": "Consider null-check here."}
    ]
    """
    try:
        data = await req.json()
    except Exception as e:
        return {"error": f"Invalid JSON: {e}"}

    diff = data.get("diff", "")
    if not diff.strip():
        return []

    # LLM prompt to produce correct JSON array
    prompt = f"""
You are a code reviewer for the Escargot JavaScript engine project.

Analyze the following git diff and produce inline code review comments.
The output must be a valid JSON array of objects with:
- "path": the relative file path from the repo root
- "position": the position in the diff (integer, counting from 1 in the diff hunk)
- "body": the comment text

Only comment on meaningful issues.
Example:
[
  {{"path": "src/BuiltinIterator.cpp", "position": 12, "body": "Consider adding null-check to avoid crash."}}
]

Diff to review:
{diff}
"""

    try:
        ollama_resp = requests.post(
            OLLAMA_URL,
            json={"model": MODEL, "prompt": prompt, "stream": False},
            timeout=180
        )
        ollama_resp.raise_for_status()
    except Exception as e:
        return {"error": f"Failed to call Ollama API: {e}"}

    try:
        raw_response = ollama_resp.json().get("response", "").strip()
        review_array = json.loads(raw_response)
        if not isinstance(review_array, list):
            return {"error": "LLM did not return an array", "raw": raw_response}
    except Exception as e:
        return {"error": f"Invalid JSON from Ollama: {e}", "raw": raw_response}

    return review_array

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3000)