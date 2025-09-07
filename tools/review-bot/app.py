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
)

app = FastAPI(title="Escargot Review Bot API", version="1.0")

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
    print(f"Starting review for PR #{request.pull_request_number}: {request.base_sha}..{request.head_sha}")

    # Fetch latest upstream data
    try:
        print("[REVIEW] Fetching latest data from upstream...")
        run_git_command(["fetch", "upstream", "--prune"])
        run_git_command(["fetch", "upstream", f"refs/pull/{request.pull_request_number}/head"])
        print("[REVIEW] Fetch complete.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch from upstream: {e}")

    # Create diff
    diff_text = run_git_command(["diff", f"-U{DIFF_CONTEXT}", request.base_sha, request.head_sha])
    patch_set = PatchSet(diff_text)
    try:
        print(f"[REVIEW] Diff created. Files in patch: {len(patch_set)}")
    except Exception:
        print("[REVIEW] Diff created. (could not count files)")

    all_github_comments: List[Dict[str, Any]] = []
    head_blob_cache: Dict[str, List[str]] = {}

    for patched_file in patch_set:
        file_path = patched_file.path
        if not any(file_path.startswith(p) for p in REVIEW_INCLUDE_PATHS):
            continue
        print(f"[REVIEW] Processing file: {file_path}")

        for hunk in patched_file:
            try:
                print(f"[REVIEW]  Hunk @ {file_path}: -{hunk.source_start},{getattr(hunk, 'source_length', '?')} -> +{hunk.target_start},{getattr(hunk, 'target_length', '?')}")
            except Exception:
                print(f"[REVIEW]  Hunk @ {file_path}: (unable to summarize hunk)")
            mappings = create_line_mappings_for_hunk(hunk)
            if not mappings:
                print("[REVIEW]   No mappings generated; skipping hunk")
                continue

            mapping_dict = {m.target_id: m for m in mappings}
            added_count = sum(1 for m in mappings if m.line_type == 'added')
            print(f"[REVIEW]   Mappings: total={len(mappings)}, added={added_count}")

            # --- Pass 1: Defect Review ---
            prompt_defect = build_hunk_based_prompt(file_path, hunk, mappings)
            print(f"[REVIEW]   Defect pass: prompt length={len(prompt_defect)}")
            raw_defects = call_ollama_and_parse(prompt_defect, SYSTEM_PROMPT_DEFECT, 'defect')
            try:
                sample_defects = raw_defects[:2]
                print(f"[REVIEW]   Defect raw count={len(raw_defects)} sample={sample_defects}")
            except Exception:
                print("[REVIEW]   Defect raw: (unable to print sample)")
            accepted_defect_ids = set()

            for c in raw_defects:
                # Schema validation (skip if invalid)
                try:
                    llm_comment = LLMReviewComment(**c)
                except Exception as e:
                    print(f"[REVIEW]    Skip(defect): schema invalid -> {e} | raw={c}")
                    continue

                if llm_comment.confidence < CONFIDENCE_THRESHOLD:
                    print(f"[REVIEW]    Skip(defect): low confidence {llm_comment.confidence:.2f} < {CONFIDENCE_THRESHOLD}")
                    continue

                # Body validation guard - check for prohibited ID/line references
                if not validate_comment_body(llm_comment.body):
                    print(f"[REVIEW]    Skip(defect): body contains prohibited references")
                    continue

                m = mapping_dict.get(llm_comment.target_id)
                if not m or m.line_type != 'added' or m.target_line_no is None:
                    print(f"[REVIEW]    Skip(defect): invalid target_id={llm_comment.target_id} or not added line")
                    continue

                # Verify alignment against HEAD; if mismatched, try nearby alignment
                line_no = m.target_line_no
                head_ok = assert_head_alignment(request.head_sha, file_path, m, head_blob_cache)
                if head_ok is False:
                    print(f"[REVIEW]    Align mismatch at ~{line_no}, trying nearby align...")
                    aligned = try_nearby_align(request.head_sha, file_path, m, head_blob_cache)
                    if aligned is None:
                        print("[REVIEW]    Skip(defect): nearby align failed")
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
                print(f"[REVIEW]    Accept(defect): id={llm_comment.target_id} -> line={line_no}")

            # --- Pass 2: Refactoring Review ---
            prompt_refactor = build_hunk_based_prompt(file_path, hunk, mappings)
            print(f"[REVIEW]   Refactor pass: prompt length={len(prompt_refactor)}")
            raw_refactors = call_ollama_and_parse(prompt_refactor, SYSTEM_PROMPT_REFACTOR, 'refactor')
            try:
                sample_ref = raw_refactors[:2]
                print(f"[REVIEW]   Refactor raw count={len(raw_refactors)} sample={sample_ref}")
            except Exception:
                print("[REVIEW]   Refactor raw: (unable to print sample)")

            for c in raw_refactors:
                try:
                    llm_comment = LLMReviewComment(**c)
                except Exception as e:
                    print(f"[REVIEW]    Skip(refactor): schema invalid -> {e} | raw={c}")
                    continue

                # If this target_id already produced a defect comment, skip refactor for the same ID
                if llm_comment.target_id in accepted_defect_ids:
                    print(f"[REVIEW]    Skip(refactor): already accepted as defect id={llm_comment.target_id}")
                    continue

                if llm_comment.confidence < CONFIDENCE_THRESHOLD:
                    print(f"[REVIEW]    Skip(refactor): low confidence {llm_comment.confidence:.2f} < {CONFIDENCE_THRESHOLD}")
                    continue

                # Body validation guard - check for prohibited ID/line references
                if not validate_comment_body(llm_comment.body):
                    print(f"[REVIEW]    Skip(refactor): body contains prohibited references")
                    continue

                m = mapping_dict.get(llm_comment.target_id)
                if not m or m.line_type != 'added' or m.target_line_no is None:
                    print(f"[REVIEW]    Skip(refactor): invalid target_id={llm_comment.target_id} or not added line")
                    continue

                # Verify alignment; if mismatched, try nearby alignment
                line_no = m.target_line_no
                head_ok = assert_head_alignment(request.head_sha, file_path, m, head_blob_cache)
                if head_ok is False:
                    print(f"[REVIEW]    Align mismatch at ~{line_no}, trying nearby align...")
                    aligned = try_nearby_align(request.head_sha, file_path, m, head_blob_cache)
                    if aligned is None:
                        print("[REVIEW]    Skip(refactor): nearby align failed")
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
                print(f"[REVIEW]    Accept(refactor): id={llm_comment.target_id} -> line={line_no}")

    print(f"[REVIEW] Generated {len(all_github_comments)} comments in total.")
    return JSONResponse(content={"comments": all_github_comments})

if __name__ == "__main__":
    # For local development
    uvicorn.run(app, host="0.0.0.0", port=8000)