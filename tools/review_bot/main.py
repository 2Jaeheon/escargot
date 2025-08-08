import json
import os

import ollama
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI()

class DiffRequest(BaseModel):
    diff: str

@app.post("/review")
async def review_diff(request: DiffRequest):
    try:
        model_name = os.getenv("OLLAMA_MODEL", "llama3")
        
        prompt = f"""
        You are a senior software engineer performing a code review.
        Examine the following git diff (-U0) and provide inline review comments.

        OUTPUT FORMAT (valid JSON ONLY, no markdown, no prose):
        [
          {{
            "path": "relative/path/to/file.cpp",
            "side": "RIGHT",
            "line": 42,
            "body": "Why & how to improve"
          }},
          {{
            "path": "relative/path/to/file.cpp",
            "side": "RIGHT",
            "start_line": 10,
            "line": 15,
            "body": "Why & how to improve (multi-line)"
          }}
        ]

        Rules:
        1. Use line numbers of the NEW file (right side).
        2. For multi-line comments include both start_line and line.
        3. Do NOT output markdown or explanations, only the JSON array.

        Git diff (-U0):
        ```diff
        {request.diff}
        ```
        """
        
        response = ollama.chat(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )
        
        raw = response["message"]["content"].strip()
        print("=== LLM raw response ===")
        print(raw[:2000])  # truncate to avoid huge logs

        try:
            comments = json.loads(raw)
        except json.JSONDecodeError:
            print("JSON decode failed; sending fallback general comment")
            comments = [{"path": "_general", "line": 1, "side": "RIGHT", "body": raw}]

        if not comments:
            print("LLM returned empty comments array; inserting placeholder")
            comments = [{"path": "_general", "line": 1, "side": "RIGHT", "body": "LLM did not generate specific comments."}]

        print("Parsed comments:", comments)
        return {"comments": comments}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)