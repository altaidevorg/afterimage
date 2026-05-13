# Context-to-Skill Tutorial

This tutorial shows how to run Afterimage's context-to-skill workflow and how to
compare generated skills against an original Ctx2Skill run.

The workflow has three parts:

1. Convert Ctx2Skill-style contexts into Afterimage documents.
2. Run Afterimage skill discovery.
3. Optionally compare Afterimage skills with Ctx2Skill skills and a no-skill baseline.

All commands below assume they are run from the Afterimage repository root unless
otherwise stated.

## Requirements

Install the project dependencies:

```bash
uv sync
```

Set the API key expected by your config:

```bash
export GEMINI_API_KEY=...
```

If you use a different provider or environment variable, update the config's
`model.api_key_env` field.

## Configure the Model

Use `examples/configs/context_to_skill_compare.yaml` as a working example.

The top-level model is used for every stage unless a stage-specific override is
provided:

```yaml
model:
  provider: gemini
  model_name: gemini-3.1-flash-lite-preview
  api_key_env: GEMINI_API_KEY
```

To configure each stage independently:

```yaml
skill:
  models:
    challenger: gemini-3.1-flash-lite-preview
    reasoner: gemini-3.1-flash-lite-preview
    judge: gemini-3.1-flash-lite-preview
    proposer: gemini-3.1-flash-lite-preview
    generator: gemini-3.1-flash-lite-preview
```

Stage meanings:

- `challenger` or `probe_generator`: generates probe tasks and rubrics.
- `reasoner`: answers probes during discovery.
- `judge`: grades answers against rubrics.
- `proposer`: alias for both proposer stages.
- `reasoner_proposer`: proposes respondent-side skills from failed probes.
- `challenger_proposer`: proposes challenger-side skills from solved probes.
- `generator`: alias for both generator stages.
- `reasoner_generator`: writes respondent-side skill markdown.
- `challenger_generator`: writes challenger-side skill markdown.
- `selector_reasoner`: answers replay probes during skill selection.
- `selector_judge` or `selector`: judges replay probes during skill selection.

A stage can also specify its own provider and API key:

```yaml
skill:
  models:
    judge:
      provider: gemini
      model_name: gemini-3.1-flash-lite-preview
      api_key_env: GEMINI_API_KEY
```

## Convert Ctx2Skill Contexts

Ctx2Skill stores contexts as chat-style messages. Afterimage skill discovery
expects document rows with a text field. Convert the context JSONL first:

```bash
uv run python scripts/ctx2skill_to_afterimage_docs.py \
  --input ../Ctx2Skill/CL-bench-context-dedup.jsonl \
  --output /tmp/ctx2skill-afterimage-docs.jsonl \
  --max-rows 10
```

Arguments:

- `--input`: Source Ctx2Skill JSONL file.
- `--output`: Converted Afterimage-compatible JSONL file.
- `--max-rows`: Number of contexts to convert.

The converted rows contain:

- `id`: context id
- `text`: formatted conversation/context
- `metadata`: source metadata
- `rubrics`: source benchmark rubrics, when present

## Run Afterimage Skill Discovery

Make sure your config points at the converted JSONL:

```yaml
documents:
  provider: jsonl
  path: /tmp/ctx2skill-afterimage-docs.jsonl
  content_key: text
  preserve_ids: true
  include_metadata: true

skill:
  output_dir: ./skills-compare-afterimage
  iterations: 3
  probes_per_context: 1
  max_contexts: 10
  bootstrap_when_no_failures: true
  use_source_rubrics: true
```

Run discovery:

```bash
uv run afterimage skill discover \
  -c examples/configs/context_to_skill_compare.yaml
```

Purpose:

This runs the context-to-skill loop:

1. Generate context-grounded probes.
2. Answer probes with the reasoner.
3. Judge answers against rubrics.
4. Update reasoner skills from failed probes.
5. Update challenger skills from solved probes.
6. Replay candidate skills and select the best respondent-side skill.

Important config fields:

- `skill.output_dir`: Directory where generated skills are written.
- `skill.iterations`: Number of self-improvement rounds per context.
- `skill.probes_per_context`: Number of probes generated per round.
- `skill.max_contexts`: Number of contexts to process.
- `skill.bootstrap_when_no_failures`: Writes a skill even if all probes pass.
- `skill.use_source_rubrics`: Uses source rubrics as hints during probe generation.

Expected output:

```text
skills-compare-afterimage/<context_id>/SKILL.md
```

Each context directory may contain:

- `probes.jsonl`: generated probe tasks and rubrics
- `results.jsonl`: probe answers and judge results
- `versions.jsonl`: respondent skill versions
- `challenger_versions.jsonl`: challenger skill versions
- `skill-iter-*.md`: respondent skill markdown by iteration
- `challenger-skill-iter-*.md`: challenger skill markdown by iteration
- `selection.json`: replay selection scores
- `SKILL.md`: final selected respondent skill

## Inspect Generated Skills

List final selected skills:

```bash
find skills-compare-afterimage -mindepth 2 -maxdepth 2 -name SKILL.md -print
```

Read one selected skill:

```bash
cat skills-compare-afterimage/<context_id>/SKILL.md
```

Read its selection result:

```bash
cat skills-compare-afterimage/<context_id>/selection.json
```

Useful manual checks:

- The skill should mention constraints, procedures, or failure modes from the context.
- The skill should be reusable, not just a memorized answer to one probe.
- The selected skill should be consistent with the replay scores in `selection.json`.

## Optional: Run Original Ctx2Skill

If you want to compare Afterimage with the original Ctx2Skill implementation,
run Ctx2Skill on the same number of contexts.

From the Ctx2Skill repository:

```bash
python selfplay_loop.py \
  --challenger-model gemini-3.1-flash-lite-preview \
  --reasoner-model gemini-3.1-flash-lite-preview \
  --judge-model gemini-3.1-flash-lite-preview \
  --proposer-model gemini-3.1-flash-lite-preview \
  --generator-model gemini-3.1-flash-lite-preview \
  --input ./CL-bench-context-dedup.jsonl \
  --output outputs/loop_data/compare_10ctx.jsonl \
  --num-iterations 3 \
  --num-tasks 1 \
  --skills-dir skills-compare-10ctx \
  --max-samples 10 \
  --workers 1
```

The generated Ctx2Skill respondent skills are expected under:

```text
skills-compare-10ctx/reasoner/<context_id>/SKILL.md
```

## Compare Skill Texts

This is a cheap lexical comparison between selected `SKILL.md` files.

```bash
uv run python scripts/compare_skill_outputs.py \
  --ctx2skill-root ../Ctx2Skill/skills-compare-10ctx/reasoner \
  --afterimage-root ./skills-compare-afterimage \
  --output output/skill_text_compare_10ctx.json
```

The report includes:

- `paired_contexts`: contexts with skills on both sides
- `ctx2skill_only`: contexts only present in Ctx2Skill output
- `afterimage_only`: contexts only present in Afterimage output
- `avg_token_jaccard`: token overlap between skill texts
- `avg_sequence_ratio`: character sequence similarity

Lexical similarity is only a sanity check. Two useful skills can have low text
overlap if they express the same procedure differently.

## Evaluate Baseline vs Ctx2Skill vs Afterimage

Use pass-rate evaluation to test whether skills improve task performance.

```bash
uv run python scripts/evaluate_skill_pass_rates.py \
  --config examples/configs/context_to_skill_compare.yaml \
  --docs /tmp/ctx2skill-afterimage-docs.jsonl \
  --ctx2skill-root ../Ctx2Skill/skills-compare-10ctx/reasoner \
  --afterimage-root ./skills-compare-afterimage \
  --output output/base_vs_ctx2skill_vs_afterimage_eval_10ctx_mixed.json \
  --limit-contexts 10 \
  --limit-tasks 6
```

Variants:

- `baseline`: no skill injected
- `ctx2skill`: Ctx2Skill skill injected
- `afterimage`: Afterimage skill injected

Task sources:

- `ctx2skill-hard-set`: hard tasks generated by Ctx2Skill
- `afterimage-probe`: probes generated by Afterimage during discovery

Metrics:

- `tasks`: number of evaluated tasks
- `passed`: number of tasks where all rubrics passed
- `pass_rate`: `passed / tasks`
- `avg_score`: strict pass score averaged across tasks
- `avg_rubric_satisfaction`: average fraction of satisfied rubrics

Example interpretation:

```text
afterimage  tasks=59 passed=30 pass_rate=0.508 avg_rubric_satisfaction=0.811
baseline    tasks=59 passed=26 pass_rate=0.441 avg_rubric_satisfaction=0.776
ctx2skill   tasks=59 passed=30 pass_rate=0.508 avg_rubric_satisfaction=0.792

source              variant       tasks passed pass_rate
--------------------------------------------------------
afterimage-probe   afterimage      30     22     0.733
afterimage-probe   baseline        30     18     0.600
afterimage-probe   ctx2skill       30     20     0.667
ctx2skill-hard-set afterimage      29      8     0.276
ctx2skill-hard-set baseline        29      8     0.276
ctx2skill-hard-set ctx2skill       29     10     0.345
```

This means Afterimage produced functional context-specific skills and improved
over the no-skill baseline on the mixed task set. In this example, Afterimage
matched Ctx2Skill on strict pass rate while achieving higher average rubric
satisfaction. The source breakdown is important: Afterimage is strongest on
`afterimage-probe`, while Ctx2Skill is stronger on `ctx2skill-hard-set`.

## Choosing Run Size

Start with a small run:

```yaml
skill:
  iterations: 1
  probes_per_context: 1
  max_contexts: 3
```

Then increase context count and iterations:

```yaml
skill:
  iterations: 3
  probes_per_context: 1
  max_contexts: 10
```

Evaluation call count is approximately:

```text
contexts * tasks_per_context * variants * 2
```

The `2` accounts for one answer call and one judge call per variant.

For example:

```text
10 contexts * 2 tasks * 3 variants * 2 = 120 LLM calls
```

Do not rely on lexical similarity alone. The pass-rate evaluation is the more
important signal.
