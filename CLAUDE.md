# Enzyme_agent — project operating instructions

## Where campaign results live (single source of truth)
ALL structural-modeling / MD / barrier / cofold **campaign run folders go under
`structural_modeling_agent/runs/`** — never top-level `runs/`. Going forward, to find any campaign's
results you look in exactly one place: `structural_modeling_agent/runs/`.

- Create every new campaign dir as `structural_modeling_agent/runs/<campaign_name>/` from the start
  (don't build it elsewhere and move it later).
- **Keep each campaign self-contained — do NOT fragment.** MD outputs, prep, figures, stats, and movies
  for one campaign all live inside that campaign's folder, with analysis artifacts in an `analysis/`
  subdir and movies in `movies/` (see the global `feedback_run_folder_artifacts_naming` rule). Candidate
  subfolders use `Gene_ACCESSION` names (e.g. `GlnA3_Q9KZC7/`).
- `structural_modeling_agent/runs/` is gitignored (as is top-level `runs/`), so this is organization,
  not version control. A `structural_modeling_agent/runs/README.md` indexes the campaigns on disk.
- Legacy note: older top-level `runs/...` dirs predate this rule; the glutarate+cadaverine step-1/step-2
  campaigns were consolidated into `structural_modeling_agent/runs/` on 2026-11-16.

## GPU / cluster courtesy
Follow the global rules: all GPU work goes through `~/bin/gpu_reserve` (never raw
`CUDA_VISIBLE_DEVICES`), one task per GPU, cap CPU threads per worker. Do NOT use lambda4 or lambda13
for fan-out MD. See `~/.claude/CLAUDE.md` and the `lambda-fleet` skill.

## Git attribution and remote sync
- Always commit and push under the repository's configured user identity (never under Gemini, Claude, Antigravity, Codex, or any other AI name/email). Never pass `--author`, override `GIT_AUTHOR_*`/`GIT_COMMITTER_*`, or add `Co-Authored-By` / AI-attribution trailers to commits or PRs.
- Commit and push to `origin` (`github.com/Ray16/Enzyme_agent`) after every substantial code change (more than ~30 lines of code), staging only the relevant files and never force-pushing.
