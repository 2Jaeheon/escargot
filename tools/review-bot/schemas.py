from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel

class ReviewRequest(BaseModel):
    base_sha: str
    head_sha: str
    pull_request_number: int

class LineMapping(BaseModel):
    target_id: int
    line_type: Literal['added', 'removed', 'context']
    content: str
    source_line_no: Optional[int] = None
    target_line_no: Optional[int] = None

class LLMReviewComment(BaseModel):
    target_id: int
    body: str
    confidence: float

class GitHubComment(BaseModel):
    path: str
    body: str
    commit_id: str
    line: int
    side: Literal["LEFT", "RIGHT"]
