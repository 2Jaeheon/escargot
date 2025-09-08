import os
import re
import json
import subprocess
import signal
import time
from typing import List, Dict, Any, Optional, Literal
from dotenv import load_dotenv

import ollama
from fastapi import HTTPException
from unidiff import Hunk
from logger import get_logger

load_dotenv()


# Settings/Environment Variables

REPO_PATH = os.getenv("REPO_PATH")
if not REPO_PATH or not os.path.isdir(REPO_PATH):
    raise ValueError(f"REPO_PATH '{REPO_PATH}' is not a valid directory")

MODEL_NAME = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")

# Review Bot Settings
DIFF_CONTEXT = int(os.getenv("DIFF_CONTEXT", "10"))
REVIEW_INCLUDE_PATHS = [p.strip() for p in os.getenv("REVIEW_INCLUDE_PATHS", "src/").split(",") if p.strip()]
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.1"))
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
OLLAMA_NUM_BATCH = int(os.getenv("OLLAMA_NUM_BATCH", "256"))
OLLAMA_REPEAT_PENALTY = float(os.getenv("OLLAMA_REPEAT_PENALTY", "1.1"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.8"))
ALIGN_SEARCH_WINDOW = int(os.getenv("ALIGN_SEARCH_WINDOW", "25"))
OLLAMA_TIMEOUT_SECONDS = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "480"))
OLLAMA_MAX_RETRIES = int(os.getenv("OLLAMA_MAX_RETRIES", "2"))
INTER_REQUEST_DELAY_SECONDS = float(os.getenv("INTER_REQUEST_DELAY_SECONDS", "5"))

# Regular Expressions
IDENT_RE = re.compile(r'\b[A-Za-z_]\w*\b')
FENCE_RE = re.compile(r"^\s*```(?:json|diff|[\w-]+)?\s*|\s*```\s*$", re.MULTILINE)
JSON_ARRAY_RE = re.compile(r"\[[\s\S]*?\]")

SYMBOL_ONLY = {'{', '}', '};'}

logger = get_logger("review-bot.helpers")

# Git Helpers
def run_git_command(command: List[str]) -> str:
    """Executes a Git command in the repository path and returns the output."""
    try:
        logger.debug(f"GIT exec: git {' '.join(command)} (cwd={REPO_PATH})")
        out = subprocess.check_output(["git"] + command, cwd=REPO_PATH, text=True)
        logger.debug(f"GIT ok: len={len(out)}")
        return out
    except subprocess.CalledProcessError as e:
        logger.error(f"GIT command failed: git {' '.join(command)} -> {e}")
        raise HTTPException(status_code=500, detail="An internal Git command failed.")

def get_file_full_content(head_sha: str, path: str) -> str:
    return run_git_command(["show", f"{head_sha}:{path}"])


# Hunk -> Line mappings
class LineMappingLite:
    """Lightweight runtime struct (to avoid circular imports from schemas)."""
    def __init__(self, target_id: int, line_type: str, content: str,
                 source_line_no: Optional[int], target_line_no: Optional[int]) -> None:
        self.target_id = target_id
        self.line_type = line_type
        self.content = content
        self.source_line_no = source_line_no
        self.target_line_no = target_line_no

def create_line_mappings_for_hunk(hunk: Hunk) -> List[LineMappingLite]:
    """
    Counting line numbers directly from the target_start/source_start in the Hunk header as the starting point,
    calculate the absolute line numbers for the right (HEAD) and left (BASE) sides.
    """
    mappings: List[LineMappingLite] = []
    current_id = 1

    right_line = hunk.target_start  # +c (HEAD file start line)
    left_line  = hunk.source_start  # -a (BASE file start line)

    for line in hunk:
        if line.is_added:
            mappings.append(LineMappingLite(
                target_id=current_id,
                line_type='added',
                content=line.value,
                source_line_no=None,
                target_line_no=right_line,
            ))
            right_line += 1

        elif line.is_removed:
            mappings.append(LineMappingLite(
                target_id=current_id,
                line_type='removed',
                content=line.value,
                source_line_no=left_line,
                target_line_no=None,
            ))
            left_line += 1

        else:
            # context
            mappings.append(LineMappingLite(
                target_id=current_id,
                line_type='context',
                content=line.value,
                source_line_no=left_line,
                target_line_no=right_line,
            ))
            left_line  += 1
            right_line += 1

        current_id += 1

    return mappings


# Normalization Helpers
def normalize_for_compare(s: str) -> str:
    # For comparison: expand tabs to spaces and remove leading/trailing whitespace/EOL
    return (s or "").expandtabs(4).strip()

def line_without_prefix(raw: str) -> str:
    if not raw:
        return ""
    if raw[0] in {"+", "-"}:
        return raw[1:]
    return raw

def is_meaningful_code(raw: str) -> bool:
    s = (raw or "").strip()
    if not s:
        return False
    if s in SYMBOL_ONLY:
        return False
    return True


# Alignment helpers
def assert_head_alignment(head_sha: str, path: str, mapping: LineMappingLite,
                          head_cache: Dict[str, List[str]]) -> Optional[bool]:
    """
    For added lines: check if the target_line_no we calculated matches the actual line in the head_sha:path file.
    Return True/False/None (None = not a target for checking)
    """
    if mapping.line_type != 'added' or mapping.target_line_no is None:
        return None

    key = f"{head_sha}:{path}"
    if key not in head_cache:
        blob_text = run_git_command(["show", f"{head_sha}:{path}"])
        head_cache[key] = blob_text.splitlines()

    lines = head_cache[key]
    idx = mapping.target_line_no - 1
    if not (0 <= idx < len(lines)):
        logger.debug(f"Align out-of-range: {path}:{mapping.target_line_no} (len={len(lines)})")
        return False

    expected = normalize_for_compare(line_without_prefix(mapping.content))
    actual   = normalize_for_compare(lines[idx])

    if expected == actual:
        return True

    logger.debug(
        f"Align mismatch: {path}:{mapping.target_line_no} expected={expected!r} actual={actual!r}"
    )
    return False



def try_nearby_align(head_sha: str, path: str, mapping: LineMappingLite,
                     head_cache: Dict[str, List[str]]) -> Optional[int]:
    """
    When the mapping(target_line_no) is not aligned in the HEAD file,
    find the first line in the ±ALIGN_SEARCH_WINDOW that has the same text,
    and return the line number (1-based). If not found, return None.
    """
    if mapping.line_type != 'added' or mapping.target_line_no is None:
        return None

    key = f"{head_sha}:{path}"
    if key not in head_cache:
        blob_text = run_git_command(["show", f"{head_sha}:{path}"])
        head_cache[key] = blob_text.splitlines()

    lines = head_cache[key]
    total = len(lines)
    base_idx = mapping.target_line_no - 1
    expected = normalize_for_compare(mapping.content[1:])

    # First check the current position
    if 0 <= base_idx < total and normalize_for_compare(lines[base_idx]) == expected:
        return mapping.target_line_no

    # Expand outward
    for delta in range(1, ALIGN_SEARCH_WINDOW + 1):
        # Up
        up = base_idx - delta
        if 0 <= up < total and normalize_for_compare(lines[up]) == expected:
            return up + 1
        # Down
        down = base_idx + delta
        if 0 <= down < total and normalize_for_compare(lines[down]) == expected:
            return down + 1

    return None


# Prompt Builder Helpers
def build_hunk_based_prompt(path: str, hunk: Hunk, mappings: List[LineMappingLite]) -> str:
    """Build an LLM prompt based on the Hunk and mappings (only added lines with meaningful code)."""
    hunk_text = str(hunk)

    commentable = [
        m for m in mappings
        if m.line_type == 'added' and is_meaningful_code(m.content[1:])
    ]
    if commentable:
        commentable_catalog = [
            f"<ID {m.target_id} | {m.line_type.upper()}>: {line_without_prefix(m.content).strip()}"
            for m in commentable
        ]
        commentable_str = "\n".join(commentable_catalog)
    else:
        commentable_str = "(no added lines)"

    return f"""
You are a world-class C++ and JavaScript engine reviewer for the Escargot project. Your review must be strict and technically precise.

## Target File: `{path}`
## Review Task
Your task is to review the code changes within the `DIFF HUNK` section only. Use the diff purely; do NOT comment on any line outside of the 'Commentable Catalog'.

### Hard Rules
- Choose "target_id" ONLY from **Commentable Catalog (ADDED lines only)**.
- If no qualifying added line has an issue, return [].
- Do NOT mention or infer any line numbers (e.g., "line 47", "at 115"). Anchor only by exact tokens from the chosen line.

---
### 1. DIFF HUNK
```diff
{hunk_text}
```

---
### 2. Commentable Catalog (ADDED lines only; eligible IDs)
```
{commentable_str}
```

### Output (JSON array only)
Each object: "target_id", "body", "confidence". If none, return [].
""".strip()

# LLM Helpers
class OllamaTimeoutError(Exception):
    pass

# Handler that raises OllamaTimeoutError when SIGALRM signal is received
def _timeout_handler(signum, frame):
    raise OllamaTimeoutError("Ollama API call timed out after 5 minutes.")

# Register signal handler
signal.signal(signal.SIGALRM, _timeout_handler)

def _extract_content(resp) -> str:
    # Defensively handle Ollama python client variants
    msg = getattr(resp, "message", None)
    if msg is not None:
        c = getattr(msg, "content", None)
        if isinstance(c, str) and c:
            return c
        if isinstance(msg, dict):
            c = msg.get("content")
            if isinstance(c, str) and c:
                return c
    if isinstance(resp, dict):
        m = resp.get("message")
        if isinstance(m, dict):
            c = m.get("content")
            if isinstance(c, str) and c:
                return c
    return ""

def sanitize_llm_output(raw: str) -> str:
    """
    Only safely extract JSON arrays from responses.
    - ```json ... ``` If there is a fence, use it first.
    - If not, find all [ ... ] candidates in the text.
        Finally, adopt the array that json.loads succeeds.
    """
    s = raw or ""

    # 1) CodeFence First
    m = re.findall(r"```json\s*([\s\S]*?)\s*```", s, flags=re.IGNORECASE)
    for block in m:
        cand = block.strip()
        try:
            obj = json.loads(cand)
            if isinstance(obj, list):
                logger.debug(f"LLM fenced JSON extracted (len={len(cand)})")
                return cand
        except Exception:
            pass

    # 2) If there is no fence, use the last successful parsing among the array candidates.
    candidates = JSON_ARRAY_RE.findall(s)
    for cand in candidates:
        try:
            obj = json.loads(cand)
            if isinstance(obj, list):
                logger.debug(f"LLM inline JSON array extracted (len={len(cand)})")
                return cand
        except Exception:
            continue

    # Return original text if unsuccessful (handle empty list from above)
    logger.debug("LLM no JSON array could be extracted from response")
    return ""


def _find_complete_json_array_span(s: str) -> Optional[tuple]:
    """
    Return (start_idx, end_idx) for the first complete JSON array `[ ... ]` in s,
    respecting strings/escapes. If none, return None.
    """
    if not s:
        return None
    start = s.find("[")
    if start == -1:
        return None
    in_str = False
    esc = False
    depth = 0
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == '[':
            depth += 1
        elif ch == ']':
            depth -= 1
            if depth == 0:
                return (start, i)
    return None

def validate_comment_body(body: str) -> bool:
    """
    Validate that comment body doesn't contain prohibited references.
    Returns True if valid, False if should be filtered out.
    """
    if not isinstance(body, str):
        return False
    
    body_lower = body.lower()
    
    # Check for prohibited patterns
    prohibited_patterns = [
        r'\bid\s*\d+\b',
        r'\btarget_id\s*\d+\b',
        r'\bline\s*\d+\b',
        r'\blines\s*\d+\b',
        r'\bat\s*\d+\b',
        r'\b#\d+\b',
        r'\bcatalog\b',
        r'\bcode\s*catalog\b',
        r'\bcommentable\s*catalog\b',
    ]
    
    for pattern in prohibited_patterns:
        if re.search(pattern, body_lower):
            logger.debug(f"Filter: body contains prohibited pattern '{pattern}': {body[:100]}...")
            return False
    
    return True

def call_ollama_and_parse(prompt: str, system_prompt: str,
                          model_type: Literal['defect', 'refactor']) -> List[Dict[str, Any]]:
    """
    Call Ollama with timeout and retry logic, and parse the response.
    model_type: 'defect' | 'refactor'
    """
    for attempt in range(OLLAMA_MAX_RETRIES):
        try:
            logger.info(f"LLM request start model={MODEL_NAME} type={model_type}")
            logger.debug(f"LLM attempt {attempt + 1}/{OLLAMA_MAX_RETRIES} timeout={OLLAMA_TIMEOUT_SECONDS}s")
            
            signal.alarm(OLLAMA_TIMEOUT_SECONDS)

            stream = ollama.chat(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                options={
                    "temperature": OLLAMA_TEMPERATURE,
                    "num_ctx": OLLAMA_NUM_CTX,
                    "num_batch": OLLAMA_NUM_BATCH,
                    "repeat_penalty": OLLAMA_REPEAT_PENALTY
                },
                keep_alive="60m",
                stream=True,
            )

            buf_parts: List[str] = []
            parsed: Optional[List[Dict[str, Any]]] = None
            for chunk in stream:
                content = getattr(getattr(chunk, "message", None), "content", None)
                if content is None and isinstance(chunk, dict):
                    msg = chunk.get("message")
                    if isinstance(msg, dict):
                        content = msg.get("content")
                if not isinstance(content, str) or not content:
                    continue
                buf_parts.append(content)
                text = "".join(buf_parts)
                span = _find_complete_json_array_span(text)
                if span is not None:
                    start, end = span
                    array_text = text[start:end + 1]
                    try:
                        raw_comments = json.loads(array_text)
                        if isinstance(raw_comments, list):
                            parsed = [c for c in raw_comments if isinstance(c, dict)]
                            logger.debug("LLM stream-early-stop: json array complete")
                            logger.debug(f"LLM items={len(parsed)}")
                            break
                    except Exception:
                        pass
            
            # Disable the timeout alarm when the task completes successfully
            signal.alarm(0)

            if parsed is None:
                text = "".join(buf_parts)
                cleaned = sanitize_llm_output(text)
                if not cleaned:
                    logger.debug("LLM sanitize produced empty string; returning []")
                    return []
                raw_comments = json.loads(cleaned)
                if not isinstance(raw_comments, list):
                    logger.debug(f"LLM parsed non-list JSON: type={type(raw_comments)}")
                    return []
                parsed = [c for c in raw_comments if isinstance(c, dict)]

            logger.info(f"LLM parsed comments: count={len(parsed)} type={model_type}")
            try:
                logger.debug(f"LLM sample parsed: {parsed[:2]}")
            except Exception:
                pass
            
            # Return the result and exit the loop if successful
            return parsed

        except OllamaTimeoutError as e:
            logger.warning(f"LLM timeout: {e}")
            if attempt + 1 < OLLAMA_MAX_RETRIES:
                logger.info("LLM retrying...")
            else:
                logger.error("LLM max retries reached. Aborting.")
                return []
        
        except Exception as e:
            logger.error(f"LLM unexpected error: {e}")
            return []
        
        finally:
            # Disable the timeout alarm in all cases
            signal.alarm(0)
            # Optional spacing between consecutive chats
            if INTER_REQUEST_DELAY_SECONDS > 0:
                try:
                    time.sleep(INTER_REQUEST_DELAY_SECONDS)
                except Exception:
                    pass
            
    # Return an empty list if all retries fail
    return []