"""
RePair I2T synthesis (paper Algorithm 1): work items from FP pools, joint QC,
edit instruction G(q, p_fp), M caption edits, CLIP-cosine selection (PQC).

- api.py: chat client (OpenAI-compatible), pipeline API client, token counts
- retry.py: the single retry layer of all API calls
- work_items.py: (query, FP) work items from an I2T FP pool
- i2t/: prompts, model-call steps, workflow, cosine selection
- runner.py: serial / thread-pool execution with resume
- store.py: per-item result directories and run summaries
"""
