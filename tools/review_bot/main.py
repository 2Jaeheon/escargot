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
        Examine the following git diff and provide inline review comments.

        Output FORMAT (must be **valid JSON only**, no markdown, no explanation):
        [
          {{
            "path": "relative/file/path.cpp",  // file path in repository
            "line": 42,                          // line number in the NEW (right) side
            "side": "RIGHT",                   // always "RIGHT"
            "body": "Explain what should be improved and why."
          }},
          ... (one object per comment)
        ]

        Only output the JSON array. Do NOT wrap it in markdown.

        Git diff to review:
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