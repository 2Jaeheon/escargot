import uvicorn
import os
import asyncio
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from typing import List, Dict, Any
from unidiff import PatchSet
from schemas import (
    ReviewRequest,
    LLMReviewComment,
    GitHubComment
)
from prompt_defect import SYSTEM_PROMPT_DEFECT
from prompt_refactor import SYSTEM_PROMPT_REFACTOR
from helpers import (
    DIFF_CONTEXT,
    REVIEW_INCLUDE_PATHS,
    CONFIDENCE_THRESHOLD,
    create_line_mappings_for_hunk,
    build_hunk_based_prompt,
    run_git_command,
    call_ollama_and_parse,
    assert_head_alignment,
    try_nearby_align,
    validate_comment_body,
    fetch_upstream_with_fallback,
)
from logger import get_logger

app = FastAPI(title="Escargot Review Bot API", version="1.0")
logger = get_logger("review-bot.app")

# Limit concurrent /review executions per-process (queue overflowing requests)
REVIEW_MAX_CONCURRENCY = int(os.getenv("REVIEW_MAX_CONCURRENCY", "1"))
_review_semaphore = asyncio.Semaphore(REVIEW_MAX_CONCURRENCY)


@app.middleware("http")
async def _queue_review_requests(request: Request, call_next):
    # Only gate the /review POST endpoint (support optional trailing slash)
    path = request.url.path.rstrip("/") or "/"
    if path == "/review" and request.method.upper() == "POST":
        async with _review_semaphore:
            return await call_next(request)
    return await call_next(request)


@app.post("/review")
async def handle_review_request(request: ReviewRequest) -> JSONResponse:
    """
    Receives a request from GitHub Actions, analyzes the code changes,
    and returns AI-generated review comments for both defect and refactor passes.
    """
    logger.info(f"Start review PR=#{request.pull_request_number} {request.base_sha}..{request.head_sha}")

    # Fetch latest upstream data (robust to missing PR ref)
    logger.info("Fetching latest data from upstream...")
    fetch_upstream_with_fallback(request.pull_request_number, request.base_sha, request.head_sha)
    logger.info("Fetch complete.")

    # Create diff
    diff_text = run_git_command([
        "diff", "--no-color", "--no-ext-diff", "--text",
        f"-U{DIFF_CONTEXT}", request.base_sha, request.head_sha
    ])
    diff_text = diff_text.replace("\r\n", "\n")
    if not diff_text.endswith("\n"):
        diff_text += "\n"
    # Parse diff robustly; if parsing fails, return empty comments instead of 500
    try:
        patch_set = PatchSet.from_string(diff_text)
        try:
            logger.info(f"Diff created. files={len(patch_set)}")
        except Exception:
            logger.info("Diff created. (could not count files)")
    except Exception as e:
        logger.exception(f"Diff parse failed: {e}")
        return JSONResponse(content={"comments": []})

    all_github_comments: List[Dict[str, Any]] = []
    head_blob_cache: Dict[str, List[str]] = {}

    for patched_file in patch_set:
        file_path = patched_file.path
        if not any(file_path.startswith(p) for p in REVIEW_INCLUDE_PATHS):
            continue
        logger.info(f"Processing file: {file_path}")

        for hunk in patched_file:
            try:
                logger.debug(f"Hunk {file_path}: -{hunk.source_start},{getattr(hunk, 'source_length', '?')} -> +{hunk.target_start},{getattr(hunk, 'target_length', '?')}")
            except Exception:
                logger.debug(f"Hunk {file_path}: (unable to summarize hunk)")
            mappings = create_line_mappings_for_hunk(hunk)
            if not mappings:
                logger.debug("No mappings generated; skipping hunk")
                continue

            mapping_dict = {m.target_id: m for m in mappings}
            added_count = sum(1 for m in mappings if m.line_type == 'added')
            logger.debug(f"Mappings: total={len(mappings)}, added={added_count}")

            # --- Pass 1: Defect Review ---
            prompt_defect = build_hunk_based_prompt(file_path, hunk, mappings)
            logger.debug(f"Defect pass: prompt length={len(prompt_defect)}")
            raw_defects = call_ollama_and_parse(prompt_defect, SYSTEM_PROMPT_DEFECT, 'defect')
            try:
                sample_defects = raw_defects[:2]
                logger.debug(f"Defect raw count={len(raw_defects)} sample={sample_defects}")
            except Exception:
                logger.debug("Defect raw: (unable to print sample)")
            accepted_defect_ids = set()

            for c in raw_defects:
                # Schema validation (skip if invalid)
                try:
                    llm_comment = LLMReviewComment(**c)
                except Exception as e:
                    logger.debug(f"Skip(defect): schema invalid -> {e} | raw={c}")
                    continue

                if llm_comment.confidence < CONFIDENCE_THRESHOLD:
                    logger.debug(f"Skip(defect): low confidence {llm_comment.confidence:.2f} < {CONFIDENCE_THRESHOLD}")
                    continue

                # Body validation guard - check for prohibited ID/line references
                if not validate_comment_body(llm_comment.body):
                    logger.debug(f"Skip(defect): body contains prohibited references")
                    continue

                m = mapping_dict.get(llm_comment.target_id)
                if not m or m.line_type != 'added' or m.target_line_no is None:
                    logger.debug(f"Skip(defect): invalid target_id={llm_comment.target_id} or not added line")
                    continue

                # Verify alignment against HEAD; if mismatched, try nearby alignment
                line_no = m.target_line_no
                head_ok = assert_head_alignment(request.head_sha, file_path, m, head_blob_cache)
                if head_ok is False:
                    logger.debug(f"Align mismatch at ~{line_no}, trying nearby align...")
                    aligned = try_nearby_align(request.head_sha, file_path, m, head_blob_cache)
                    if aligned is None:
                        logger.debug("Skip(defect): nearby align failed")
                        continue
                    line_no = aligned

                final_comment = GitHubComment(
                    path=file_path,
                    body=llm_comment.body,
                    commit_id=request.head_sha,
                    line=line_no,
                    side="RIGHT"
                )
                all_github_comments.append(final_comment.model_dump())
                accepted_defect_ids.add(llm_comment.target_id)
                logger.debug(f"Accept(defect): id={llm_comment.target_id} -> line={line_no}")

            # --- Pass 2: Refactoring Review ---
            prompt_refactor = build_hunk_based_prompt(file_path, hunk, mappings)
            logger.debug(f"Refactor pass: prompt length={len(prompt_refactor)}")
            raw_refactors = call_ollama_and_parse(prompt_refactor, SYSTEM_PROMPT_REFACTOR, 'refactor')
            try:
                sample_ref = raw_refactors[:2]
                logger.debug(f"Refactor raw count={len(raw_refactors)} sample={sample_ref}")
            except Exception:
                logger.debug("Refactor raw: (unable to print sample)")

            for c in raw_refactors:
                try:
                    llm_comment = LLMReviewComment(**c)
                except Exception as e:
                    logger.debug(f"Skip(refactor): schema invalid -> {e} | raw={c}")
                    continue

                # If this target_id already produced a defect comment, skip refactor for the same ID
                if llm_comment.target_id in accepted_defect_ids:
                    logger.debug(f"Skip(refactor): already accepted as defect id={llm_comment.target_id}")
                    continue

                if llm_comment.confidence < CONFIDENCE_THRESHOLD:
                    logger.debug(f"Skip(refactor): low confidence {llm_comment.confidence:.2f} < {CONFIDENCE_THRESHOLD}")
                    continue

                # Body validation guard - check for prohibited ID/line references
                if not validate_comment_body(llm_comment.body):
                    logger.debug(f"Skip(refactor): body contains prohibited references")
                    continue

                m = mapping_dict.get(llm_comment.target_id)
                if not m or m.line_type != 'added' or m.target_line_no is None:
                    logger.debug(f"Skip(refactor): invalid target_id={llm_comment.target_id} or not added line")
                    continue

                # Verify alignment; if mismatched, try nearby alignment
                line_no = m.target_line_no
                head_ok = assert_head_alignment(request.head_sha, file_path, m, head_blob_cache)
                if head_ok is False:
                    logger.debug(f"Align mismatch at ~{line_no}, trying nearby align...")
                    aligned = try_nearby_align(request.head_sha, file_path, m, head_blob_cache)
                    if aligned is None:
                        logger.debug("Skip(refactor): nearby align failed")
                        continue
                    line_no = aligned

                comment_body = f"{llm_comment.body}"
                final_comment = GitHubComment(
                    path=file_path,
                    body=comment_body,
                    commit_id=request.head_sha,
                    line=line_no,
                    side="RIGHT"
                )
                all_github_comments.append(final_comment.model_dump())
                logger.debug(f"Accept(refactor): id={llm_comment.target_id} -> line={line_no}")

    logger.info(f"Generated {len(all_github_comments)} comments in total.")
    return JSONResponse(content={"comments": all_github_comments})

if __name__ == "__main__":
    # For local development
    uvicorn.run(app, host="0.0.0.0", port=8000)