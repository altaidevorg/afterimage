# Reddit drafts — OpenSimula

Copy the **Title** and **Body** blocks below into each subreddit. Check subreddit rules (flair, project tags) before posting.

---

## r/MachineLearning

### Title

`[P] OpenSimula — open implementation of Simula-style mechanism design for synthetic data (in AfterImage)`

### Body

Hi r/MachineLearning,

We added **OpenSimula** to our open-source dataset tool **AfterImage**: an **experimental** Python implementation of the **Simula** mechanism-design recipe from Davidson et al. (TMLR, [PDF](https://openreview.net/pdf?id=NALsdGEPhB); framing also in this [research blog](https://research.google/blog/designing-synthetic-datasets-for-the-real-world-mechanism-design-and-reasoning-from-first-principles/)).

**Problem it targets:**

For some SFT/eval setups you care less about “one prompt → one answer” and more about **controlled diversity** over a reasoning space: which axes of variation exist, how you **joint-sample** them, and how you **stress-test** generations before they land in a JSONL file.

**What the code actually does (high level):**

LLM-built **factor taxonomies** → **weighted mix sampling** over factors → **meta-prompt** diversification (+ optional complexification) → **requirement critic** loop with refinement → optional **double-critic** gate for **verifiable MCQ**. Artifacts are a versioned **`opensimula/`** checkpoint (manifest, taxonomy bundle, sampling strategy) plus append-only **JSONL** for accepted points. You can plug in the same **`GenerationMonitor`** we use elsewhere for per-call metrics, or bridge scenarios into **`ConversationGenerator`** via a small callback.

**Hard disclaimers (please read):**

- This is **not** a Google product, **not** a reference port of anything internal—just our read of the **published** recipe.
- API is explicitly **experimental** and may change.
- Cost and latency explode if you remove the caps on taxonomy width/depth; wide trees are **many** structured calls unless you tune bounds.
- “Mechanism design” here helps **structure** the data-generating process; it does not magically fix model collapse or bad teacher models.

**Code & docs:**

- Repo (whole library): https://github.com/altaidevorg/afterimage
- Simula examples: https://github.com/altaidevorg/afterimage/tree/main/examples/simula
- Short overview: https://afterimage.altai.dev/opensimula.html
- API autodoc: https://afterimage.altai.dev/api/simula.html

I’d genuinely like pointers from anyone working on **synthetic data with explicit diversity controls** or **evals for structured synthetic pipelines**—especially how you separate “looks diverse” from “actually helps downstream training.”

---

## r/datasets

### Title

`[Project] Versioned “opensimula/” checkpoints + JSONL streams for Simula-style synthetic QA/MCQ (AfterImage)`

### Body

Posting for people who care about **dataset layout and reproducibility**, not another “synthetic data platform.”

**OpenSimula** (module **`afterimage.simula`** in **AfterImage**) is an **experimental** implementation of the **Simula**-style pipeline from Davidson et al. (TMLR, [PDF](https://openreview.net/pdf?id=NALsdGEPhB)): factor taxonomies, weighted joint sampling strategies, meta-prompts, critic + refine loops, optional **double-critic** for MCQ, then **append-only JSONL** for rows you accept.

**Why it might matter for r/datasets:**

- On-disk layout is meant to be **inspectable**: `opensimula/manifest.json` (digests + file names), `taxonomy_bundle.json`, `sampling_strategy.json`, optional typed `run_config.json`.
- You can **push/pull** that subtree to the Hub (`push_checkpoint_to_hub` / `pull_checkpoint_from_hub`; async **`apush_checkpoint_to_hub`** if you do not want blocking I/O on an event loop).
- Monitoring hooks exist so you can log **per structured LLM call** (tokens, latency, `operation` tags) alongside the rest of AfterImage.

**Caveats in plain language:**

- **Not** affiliated with Google; **not** a drop-in “official Simula.”
- Taxonomy growth is the main **foot-gun** (runtime + cost); the examples/README spell out caps and progress logging.
- Quality still depends on your **teacher model** and task spec—this organizes *how* you sample and gate, not *whether* the model is any good.

**Links:**

- Examples + knobs: https://github.com/altaidevorg/afterimage/blob/main/examples/simula/README.md
- Library notes: https://github.com/altaidevorg/afterimage/blob/main/afterimage/simula/README.md
- Docs: https://afterimage.altai.dev/opensimula.html

If you maintain **dataset cards** or **Hub** datasets for synthetic corpora: what metadata do you wish authors *always* shipped (besides LICENSE)? I’m trying to align our default card + manifest fields with what curators actually grep for.

---

## Posting notes

- Space r/MachineLearning and r/datasets by **a day or two** so cross-posting does not look like spam.
- Re-read each subreddit’s **sidebar rules** (flair, self-promo, `[P]` requirements) before submit.
- First comments matter: answer comparisons and limitations technically; avoid defensive tone (see `afterimage-launch-playbook.md`).
