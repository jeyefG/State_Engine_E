# Phase D → Phase E Context Contract
1. Phase D builds a single canonical context bundle (ctx_df + df_score_ctx).
2. LOOK_FOR_* columns are materialized only in Phase D (gating policy).
3. Validation happens before gating; upstream gating logic stays unchanged.
4. Every phase_d.look_fors.*.enabled:true must exist in ctx_df and be binary.
5. Missing required context columns raise errors in Phase D (no silent fallbacks).
6. Phase E consumes df_score_ctx as-is: no allow recomputation or skipping.
7. required_allow_by_family defines the only allow_id allowed in Phase E.
8. Events exist iff ctx[required_allow]==1 for their family.
9. Phase E validates the contract at startup and logs allow coverage.
10. Contract validation lives in state_engine/pipeline_phase_d.py and train_event_scorer.py.
11. Phase E derives active_allows only from Phase D output (ctx_df columns) and never from YAML.
12. Family_id is computed independently of allow_id; required_allow_by_family is static and non-circular.
