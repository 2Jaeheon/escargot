import ollama
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
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
        Please review the following code changes (git diff) and provide feedback.
        Focus on identifying potential bugs, performance issues, and violations of best practices.
        The output should be in Markdown format, suitable for a GitHub comment.
        When you find a section of code to comment on, please use the git diff format to show the lines you are referring to.
        For example:
        ```diff
        - old line of code
        + new line of code
        ```
        
        Here is the diff:
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
        
        return {"review": response["message"]["content"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
