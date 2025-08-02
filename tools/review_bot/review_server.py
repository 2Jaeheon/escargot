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
    Returns: JSON array of inline review comments
    Example:
    [
      {"path": "src/BuiltinIterator.cpp", "line": 123, "comment": "Consider null-check here."}
    ]
    """
    try:
        data = await req.json()
    except Exception as e:
        return {"error": f"Invalid JSON received: {e}"}

    diff = data.get("diff", "")
    if not diff.strip():
        return []

    prompt = f"""
You are a strict code reviewer for the Escargot JavaScript engine project.

The diff is from a C++ ECMAScript engine implementation.
Analyze the changes and produce inline code review comments for the modified lines.

Rules:
- Output ONLY valid JSON array
- Each element must have:
  - "path": exact file path from the diff (relative to repo root)
  - "line": the line number in the NEW file where the issue is found
  - "comment": a concise suggestion explaining the issue and how to improve it
- Do NOT include explanations outside JSON
- Focus on ECMAScript spec compliance, performance, style, and potential bugs

Example output:
[
  {{"path": "src/BuiltinIterator.cpp", "line": 123, "comment": "Consider null-check here to follow ECMAScript spec."}}
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
        # Ensure valid JSON
        review_json = json.loads(raw_response)
        if not isinstance(review_json, list):
            return {"error": "LLM did not return a list"}
    except Exception as e:
        return {"error": f"Invalid JSON from Ollama: {e}", "raw": raw_response}

    return review_json

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3000)