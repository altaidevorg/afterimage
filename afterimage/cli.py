"""AfterImage command-line interface.

Provides ``generate``, ``validate``, and ``export`` subcommands for
working with AfterImage datasets without writing Python code.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import click
import yaml
from huggingface_hub import HfApi

from .config import AfterImageConfig, load_config, resolve_api_key


@click.group()
@click.version_option(package_name="afterimage")
def main():
    """AfterImage -- synthetic conversation dataset generator."""


# ---------------------------------------------------------------------------
# skill
# ---------------------------------------------------------------------------


@main.group()
def skill():
    """Discover and inspect context-specific skills."""


@skill.command("discover")
@click.option(
    "-c",
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to YAML config file.",
)
def skill_discover(config_path: str):
    """Discover context-specific skills from configured documents."""
    try:
        cfg = load_config(config_path)
        raw = _load_raw_config(config_path)
    except (FileNotFoundError, ValueError) as exc:
        click.secho(f"Config error: {exc}", fg="red", err=True)
        raise SystemExit(1)

    if cfg.documents is None:
        click.secho("Skill discovery requires a documents section.", fg="red", err=True)
        raise SystemExit(1)

    try:
        api_key = resolve_api_key(cfg)
    except ValueError as exc:
        click.secho(str(exc), fg="red", err=True)
        raise SystemExit(1)

    if api_key is None and cfg.model.provider == "local":
        api_key = "not-needed"

    skill_cfg = raw.get("skill", {}) if isinstance(raw.get("skill", {}), dict) else {}
    output_dir = skill_cfg.get("output_dir", "./skills")
    iterations = int(skill_cfg.get("iterations", 3))
    probes_per_context = int(skill_cfg.get("probes_per_context", 5))
    max_contexts = skill_cfg.get("max_contexts")
    max_contexts = int(max_contexts) if max_contexts is not None else None
    selection_cfg = skill_cfg.get("selection", {})
    select_best = bool(
        selection_cfg.get("enabled", True) if isinstance(selection_cfg, dict) else True
    )
    bootstrap_when_no_failures = bool(skill_cfg.get("bootstrap_when_no_failures", True))

    from .config_to_generator import _build_document_provider, _llm_create_extras
    from .providers import LLMFactory
    from .skills import SkillDiscoveryPipeline
    from .skills.generation import SkillGenerator
    from .skills.judging import RubricJudge
    from .skills.probe_generation import SkillProbeGenerator
    from .skills.proposal import SkillProposer
    from .skills.selection import SkillSelector

    document_provider = _build_document_provider(cfg)
    llm = LLMFactory.create(
        provider=cfg.model.provider,
        model_name=cfg.model.model_name,
        api_key=api_key,
        **_llm_create_extras(cfg),
    )
    stage_llms = _build_skill_stage_llms(cfg, skill_cfg, default_llm=llm)
    judge = RubricJudge(stage_llms["judge"])
    pipeline = SkillDiscoveryPipeline(
        document_provider=document_provider,
        respondent_prompt=cfg.respondent.system_prompt,
        llm=llm,
        reasoner_llm=stage_llms["reasoner"],
        output_dir=output_dir,
        probe_generator=SkillProbeGenerator(stage_llms["probe_generator"]),
        judge=judge,
        proposer=SkillProposer(stage_llms["proposer"]),
        skill_generator=SkillGenerator(stage_llms["generator"]),
        selector=SkillSelector(
            judge=RubricJudge(stage_llms["selector_judge"]),
            reasoner_llm=stage_llms["selector_reasoner"],
        ),
    )

    click.echo(
        "Discovering skills "
        f"(iterations={iterations}, probes_per_context={probes_per_context}, "
        f"max_contexts={max_contexts or 'all'})..."
    )
    try:
        selections = asyncio.run(
            pipeline.discover(
                iterations=iterations,
                probes_per_context=probes_per_context,
                max_contexts=max_contexts,
                select_best=select_best,
                bootstrap_when_no_failures=bootstrap_when_no_failures,
                show_progress=True,
            )
        )
    except Exception as exc:
        _handle_generation_error(exc, cfg)
        raise SystemExit(1)

    click.secho(f"Done. Selected {len(selections)} skill(s).", fg="green")
    click.echo(f"Skills: {output_dir}")


def _load_raw_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return raw if isinstance(raw, dict) else {}


def _build_skill_stage_llms(cfg: AfterImageConfig, skill_cfg: dict, *, default_llm):
    """Build optional per-stage LLMs for skill discovery.

    `skill.models.<stage>` may be either a model name string, or a mapping with
    provider/model_name/api_key_env/base_url. Missing stages reuse the top-level
    `model` LLM.
    """
    stage_aliases = {
        "challenger": "probe_generator",
        "probe_generator": "probe_generator",
        "reasoner": "reasoner",
        "judge": "judge",
        "proposer": "proposer",
        "generator": "generator",
        "selector_reasoner": "selector_reasoner",
        "selector_judge": "selector_judge",
        "selector": "selector_judge",
    }
    canonical_stages = {
        "probe_generator",
        "reasoner",
        "judge",
        "proposer",
        "generator",
        "selector_reasoner",
        "selector_judge",
    }
    llms = {stage: default_llm for stage in canonical_stages}
    raw_models = skill_cfg.get("models", {})
    if not isinstance(raw_models, dict):
        return llms

    from .providers import LLMFactory

    for raw_stage, spec in raw_models.items():
        stage = stage_aliases.get(str(raw_stage))
        if stage is None:
            click.secho(
                f"Warning: ignoring unknown skill model stage: {raw_stage}",
                fg="yellow",
            )
            continue
        llms[stage] = _create_skill_stage_llm(cfg, spec, LLMFactory)

    return llms


def _create_skill_stage_llm(cfg: AfterImageConfig, spec, llm_factory):
    if isinstance(spec, str):
        provider = cfg.model.provider
        model_name = spec
        api_key_env = cfg.model.api_key_env
        base_url = cfg.model.base_url
    elif isinstance(spec, dict):
        provider = spec.get("provider", cfg.model.provider)
        model_name = spec.get("model_name") or spec.get("model") or cfg.model.model_name
        api_key_env = spec.get("api_key_env", cfg.model.api_key_env)
        base_url = spec.get("base_url", cfg.model.base_url)
    else:
        raise ValueError(
            "skill.models entries must be model name strings or mappings, "
            f"got {type(spec).__name__}"
        )

    if api_key_env is None:
        defaults = {
            "gemini": "GEMINI_API_KEY",
            "openai": "OPENAI_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
        }
        api_key_env = defaults.get(provider)

    api_key = None
    if provider == "local":
        api_key = "not-needed"
    elif api_key_env:
        api_key = os.environ.get(api_key_env)
        if api_key is None:
            raise ValueError(
                f"API key not found for skill stage model {model_name!r}. "
                f"Set environment variable: export {api_key_env}=your-key"
            )

    extras = {"base_url": base_url} if base_url else {}
    return llm_factory.create(
        provider=provider,
        model_name=model_name,
        api_key=api_key,
        **extras,
    )


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------


@main.command()
@click.option(
    "-c",
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to YAML config file.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Validate config and print plan without generating.",
)
def generate(config_path: str, dry_run: bool):
    """Generate synthetic conversation dataset from config."""
    try:
        cfg = load_config(config_path)
    except (FileNotFoundError, ValueError) as exc:
        click.secho(f"Config error: {exc}", fg="red", err=True)
        raise SystemExit(1)

    if dry_run:
        _print_plan(cfg)
        return

    # Resolve API key early for clear error messages
    try:
        resolve_api_key(cfg)
    except ValueError as exc:
        click.secho(str(exc), fg="red", err=True)
        raise SystemExit(1)

    from .config_to_generator import build_conversation_run

    run = build_conversation_run(cfg)
    if run.num_requested is not None:
        click.echo(
            f"Generating until stopping rules fire (progress target: {run.num_requested} conversations)..."
        )
    else:
        click.echo(
            "Generating until stopping rules fire (no fixed conversation count; progress bar is indeterminate)..."
        )
    start = time.time()

    try:
        asyncio.run(
            run.generator.generate(
                num_dialogs=None,
                max_turns=cfg.generation.max_turns,
                max_concurrency=cfg.generation.max_concurrency,
                stopping_criteria=run.stopping_criteria,
                num_requested=run.num_requested,
            )
        )
    except ConnectionRefusedError:
        base_url = cfg.model.base_url or "the configured endpoint"
        click.secho(
            f"Connection refused to {base_url}. Is your model server running?",
            fg="red",
            err=True,
        )
        raise SystemExit(1)
    except Exception as exc:
        _handle_generation_error(exc, cfg)
        raise SystemExit(1)

    elapsed = time.time() - start
    click.secho(f"Done! Finished in {elapsed:.1f}s", fg="green")
    click.echo(f"Output: {cfg.output.path}")

    # Post-generation hooks (fail-safe: never block generation)
    if cfg.output.export and cfg.output.export.formats:
        _run_auto_export(cfg)
    if cfg.analytics.auto_analyze:
        _run_auto_analyze(cfg)


def _print_plan(cfg: AfterImageConfig) -> None:
    """Display a summary of what generation would do."""
    click.echo("=== Generation Plan ===")
    click.echo(f"  Model:          {cfg.model.provider} / {cfg.model.model_name}")
    if cfg.model.base_url:
        click.echo(f"  Base URL:       {cfg.model.base_url}")
    nd = cfg.generation.num_dialogs
    click.echo(
        f"  num_dialogs:    {nd if nd is not None else 'null (no extra fixed cap)'}"
    )
    click.echo(f"  Max turns:      {cfg.generation.max_turns}")
    if cfg.generation.stopping:
        click.echo("  Stopping (OR; first satisfied rule ends the run):")
        for rule in cfg.generation.stopping:
            payload = rule.model_dump(exclude_none=True)
            t = payload.pop("type", "?")
            extra = ", ".join(f"{k}={v!r}" for k, v in payload.items())
            click.echo(f"    - {t}" + (f"  {extra}" if extra else ""))
    else:
        click.echo(
            "  Stopping:       (YAML rules only; fixed cap comes from num_dialogs)"
        )
    conc = cfg.generation.max_concurrency or "provider default"
    click.echo(f"  Concurrency:    {conc}")
    if cfg.documents:
        click.echo(
            f"  Documents:      {cfg.documents.provider} @ {cfg.documents.path or cfg.documents.collection}"
        )
    else:
        click.echo("  Documents:      none")
    click.echo(f"  Personas:       {'enabled' if cfg.personas.enabled else 'disabled'}")
    click.echo(
        f"  Auto-improve:   {'enabled' if cfg.quality.auto_improve else 'disabled'}"
    )
    click.echo(f"  Output:         {cfg.output.path}")


def _handle_generation_error(exc: Exception, cfg: AfterImageConfig) -> None:
    """Translate common exceptions into user-friendly messages."""
    msg = str(exc)
    lower = msg.lower()

    if "api key" in lower or "authentication" in lower or "unauthorized" in lower:
        env_var = cfg.model.api_key_env or "your API key environment variable"
        click.secho(
            f"Authentication failed. Check that {env_var} is set correctly.",
            fg="red",
            err=True,
        )
    elif "rate limit" in lower or "429" in msg:
        click.secho(
            "Rate limited by provider. Try lowering max_concurrency in your config.",
            fg="red",
            err=True,
        )
    elif "connection" in lower or "connect" in lower:
        base_url = cfg.model.base_url or "the API endpoint"
        click.secho(
            f"Connection error to {base_url}. Is the server running?",
            fg="red",
            err=True,
        )
    else:
        click.secho(f"Generation failed: {exc}", fg="red", err=True)


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


@main.command()
@click.option(
    "-c",
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to YAML config file.",
)
def validate(config_path: str):
    """Validate config file without running generation."""
    all_ok = True

    # 1. Parse config
    try:
        cfg = load_config(config_path)
        _check_ok("Config syntax")
    except Exception as exc:
        _check_fail("Config syntax", str(exc))
        raise SystemExit(1)

    # 2. API key
    if cfg.model.provider != "local":
        try:
            resolve_api_key(cfg)
            _check_ok("API key")
        except ValueError as exc:
            _check_fail("API key", str(exc))
            all_ok = False
    else:
        _check_ok("API key (not required for local)")

    # 3. Document paths
    if cfg.documents and cfg.documents.path:
        p = Path(cfg.documents.path)
        if p.exists():
            _check_ok(f"Documents path ({p})")
        else:
            _check_fail("Documents path", f"{p} does not exist")
            all_ok = False
    else:
        _check_ok("Documents (none configured)")

    # 3b. Generator wiring (instruction path + stopping callbacks)
    if all_ok:
        try:
            from .config_to_generator import build_conversation_run

            build_conversation_run(cfg)
            _check_ok("Generator wiring (instructions + stopping)")
        except Exception as exc:
            _check_fail("Generator wiring", str(exc))
            all_ok = False

    # 4. Output directory writable
    output_dir = Path(cfg.output.path).parent
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        _check_ok(f"Output directory ({output_dir})")
    except OSError as exc:
        _check_fail("Output directory", str(exc))
        all_ok = False

    # 5. Local model connectivity
    if cfg.model.provider == "local" and cfg.model.base_url:
        try:
            import urllib.request

            req = urllib.request.Request(
                cfg.model.base_url.rstrip("/") + "/models",
                method="GET",
            )
            urllib.request.urlopen(req, timeout=5)
            _check_ok(f"Local server ({cfg.model.base_url})")
        except Exception:
            _check_fail(
                "Local server",
                f"Cannot reach {cfg.model.base_url}. Is the model server running?",
            )
            all_ok = False

    if all_ok:
        click.secho("\nAll checks passed!", fg="green")
    else:
        click.secho("\nSome checks failed.", fg="red")
        raise SystemExit(1)


def _check_ok(label: str) -> None:
    click.echo(click.style("  [OK] ", fg="green") + label)


def _check_fail(label: str, reason: str) -> None:
    click.echo(click.style("  [FAIL] ", fg="red") + f"{label}: {reason}")


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


@main.command()
@click.option(
    "-i",
    "--input",
    "input_path",
    type=click.Path(exists=True),
    default=None,
    help="Path to AfterImage JSONL dataset.",
)
@click.option(
    "-f",
    "--format",
    "formats",
    multiple=True,
    help="Target format(s). Repeat for multiple: -f sharegpt -f alpaca",
)
@click.option(
    "--all",
    "export_all",
    is_flag=True,
    help="Export to all available formats.",
)
@click.option(
    "-o",
    "--output-dir",
    "output_dir",
    default=None,
    type=click.Path(),
    help="Output directory (default: same as input file).",
)
@click.option(
    "--split",
    default=None,
    type=float,
    help="Train/val split ratio, e.g. 0.1 for 10%% validation.",
)
@click.option(
    "--shuffle/--no-shuffle",
    default=True,
    help="Shuffle before splitting (default: shuffle).",
)
@click.option(
    "--seed",
    default=42,
    type=int,
    help="Random seed for reproducible splits.",
)
@click.option(
    "--system-prompt",
    default=None,
    type=str,
    help="System prompt to prepend to exported conversations.",
)
@click.option(
    "--list-formats",
    is_flag=True,
    help="Show all available export formats.",
)
def export(
    input_path,
    formats,
    export_all,
    output_dir,
    split,
    shuffle,
    seed,
    system_prompt,
    list_formats,
):
    """Convert AfterImage dataset to training tool formats."""
    from .integrations import get_exporter, list_formats as _list_fmts

    if list_formats:
        _print_formats_table(_list_fmts())
        return

    if input_path is None:
        click.secho("Error: -i/--input is required.", fg="red", err=True)
        raise SystemExit(1)

    if not formats and not export_all:
        click.secho("Specify at least one -f FORMAT or use --all.", fg="red", err=True)
        raise SystemExit(1)

    if export_all:
        formats = tuple(f["name"] for f in _list_fmts())

    inp = Path(input_path)
    out_dir = Path(output_dir) if output_dir else inp.parent

    results = []
    for fmt in formats:
        try:
            exporter = get_exporter(fmt)
        except ValueError as exc:
            click.secho(str(exc), fg="red", err=True)
            raise SystemExit(1)

        if split is not None:
            r = _export_with_split(
                exporter,
                inp,
                out_dir,
                fmt,
                split,
                shuffle,
                seed,
                system_prompt=system_prompt,
            )
            results.append(r)
        else:
            out_path = out_dir / f"{inp.stem}_{fmt}.jsonl"
            r = exporter.export_file(inp, out_path, system_prompt=system_prompt)
            results.append(r)
            if r.warnings and r.total_output == 0:
                click.secho(f"  ! {fmt}: {r.warnings[0]}", fg="yellow")
            else:
                click.secho(
                    f"  {fmt}: {r.total_input:,} conversations -> "
                    f"{r.total_output:,} rows -> {r.output_path}",
                    fg="green",
                )

    # Summary table
    if len(results) > 1:
        click.echo("\nExport complete:")
        click.echo(f"{'Format':<16} {'Rows':>8} {'Skipped':>8} {'Warnings':>9}  Output")
        click.echo("-" * 72)
        for r in results:
            click.echo(
                f"{r.format_name:<16} {r.total_output:>8,} {r.skipped:>8,} "
                f"{len(r.warnings):>9}  {Path(r.output_path).name}"
            )


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------


@main.command()
@click.option(
    "-i",
    "--input",
    "input_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to AfterImage JSONL dataset.",
)
@click.option(
    "-o",
    "--output",
    "output_path",
    default=None,
    type=click.Path(),
    help="Output HTML report path. Default: input path with .html extension.",
)
def analyze(input_path: str, output_path: str | None):
    """Generate an analytics report for a dataset."""
    from .analytics import DatasetAnalyzer, generate_report

    if output_path is None:
        output_path = str(Path(input_path).with_suffix(".html"))

    try:
        report = DatasetAnalyzer.from_jsonl(input_path)
        generate_report(report, output_path)
        click.secho(f"Report saved to {output_path}", fg="green")
    except Exception as exc:
        click.secho(f"Analysis failed: {exc}", fg="red", err=True)
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# push
# ---------------------------------------------------------------------------


@main.command()
@click.option(
    "-i",
    "--input",
    "input_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to AfterImage JSONL dataset.",
)
@click.option(
    "-f",
    "--format",
    "fmt",
    default="messages",
    help="Export format before pushing (default: messages).",
)
@click.option(
    "--repo",
    required=True,
    help="HuggingFace repo: username/dataset-name",
)
@click.option(
    "--private",
    is_flag=True,
    help="Create as private dataset.",
)
@click.option(
    "--split",
    default=0.1,
    type=float,
    help="Train/val split ratio (default: 0.1).",
)
def push(input_path, fmt, repo, private, split):
    """Export and push dataset to HuggingFace Hub."""
    import json
    import tempfile
    from .integrations import get_exporter

    try:
        exporter = get_exporter(fmt)
    except ValueError as exc:
        click.secho(str(exc), fg="red", err=True)
        raise SystemExit(1)

    inp = Path(input_path)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        r = _export_with_split(
            exporter,
            inp,
            tmp_dir,
            fmt,
            split,
            shuffle=True,
            seed=42,
        )

        train_path = tmp_dir / f"{inp.stem}_{fmt}_train.jsonl"
        val_path = tmp_dir / f"{inp.stem}_{fmt}_val.jsonl"

        api = HfApi()
        api.create_repo(repo, repo_type="dataset", private=private, exist_ok=True)

        api.upload_file(
            path_or_fileobj=str(train_path),
            path_in_repo="train.jsonl",
            repo_id=repo,
            repo_type="dataset",
        )
        api.upload_file(
            path_or_fileobj=str(val_path),
            path_in_repo="val.jsonl",
            repo_id=repo,
            repo_type="dataset",
        )

        # Dataset card
        first_row = ""
        with open(train_path, encoding="utf-8") as f:
            line = f.readline().strip()
            if line:
                first_row = json.dumps(json.loads(line), indent=2)

        n_train = r.train_export_rows
        n_val = r.val_export_rows

        import importlib.metadata

        try:
            version = importlib.metadata.version("afterimage")
        except importlib.metadata.PackageNotFoundError:
            version = "0.0.0"

        from datetime import date

        card = (
            "---\nlicense: apache-2.0\ntask_categories:\n  - conversational\n"
            "tags:\n  - synthetic\n  - afterimage\n---\n"
            f"# {repo.split('/')[-1]}\n\n"
            f"Generated with [AfterImage](https://github.com/altaidevorg/afterimage) v{version}\n\n"
            f"## Dataset details\n- Format: {fmt}\n- Train samples: {n_train}\n"
            f"- Validation samples: {n_val}\n- Generated: {date.today()}\n\n"
            f"## Sample\n```json\n{first_row}\n```\n\n"
            f"## Usage\n```python\nfrom datasets import load_dataset\n"
            f'ds = load_dataset("{repo}")\n```\n'
        )

        readme_path = tmp_dir / "README.md"
        readme_path.write_text(card)
        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=repo,
            repo_type="dataset",
        )

        click.secho(f"Pushed to https://huggingface.co/datasets/{repo}", fg="green")


# ---------------------------------------------------------------------------
# preference
# ---------------------------------------------------------------------------


@main.command()
@click.option(
    "-c",
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to YAML config file.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Validate config and print plan without generating.",
)
@click.option(
    "--num-pairs",
    default=None,
    type=int,
    help="Override config preference.num_pairs.",
)
@click.option(
    "--format",
    "output_format",
    default=None,
    type=click.Choice(["dpo", "chat_dpo", "ultrafeedback", "anthropic_hh", "orpo"]),
    help="Override output format.",
)
@click.option(
    "-o",
    "--output",
    "output_path",
    default=None,
    type=click.Path(),
    help="Override output file path.",
)
@click.option(
    "--save-log",
    is_flag=True,
    default=False,
    help="Save full generation log with all scored responses.",
)
def preference(
    config_path: str, dry_run: bool, num_pairs, output_format, output_path, save_log
):
    """Generate DPO/RLHF preference pairs from a config file."""
    try:
        cfg = load_config(config_path)
    except (FileNotFoundError, ValueError) as exc:
        click.secho(f"Config error: {exc}", fg="red", err=True)
        raise SystemExit(1)

    # Build PreferenceConfig from YAML + CLI overrides
    from .preference.types import PreferenceConfig

    pref_cfg_dict = {}
    if cfg.preference is not None:
        pref_cfg_dict = cfg.preference.model_dump()

    if num_pairs is not None:
        pref_cfg_dict["num_pairs"] = num_pairs
    if output_format is not None:
        pref_cfg_dict["output_format"] = output_format
    if output_path is not None:
        pref_cfg_dict["output_path"] = output_path
    if save_log:
        pref_cfg_dict["save_log"] = True

    pref_config = PreferenceConfig(**pref_cfg_dict)

    if dry_run:
        _print_preference_plan(cfg, pref_config)
        return

    try:
        resolve_api_key(cfg)
    except ValueError as exc:
        click.secho(str(exc), fg="red", err=True)
        raise SystemExit(1)

    click.echo(f"Generating {pref_config.num_pairs} preference pairs...")
    start = time.time()

    try:
        import asyncio

        from .config_to_generator import build_generator
        from .evaluator import ConversationJudge
        from .key_management import SmartKeyPool
        from .providers import LLMFactory

        api_key = resolve_api_key(cfg)
        if api_key is None and cfg.model.provider == "local":
            api_key = "not-needed"

        gen = build_generator(cfg)

        key_pool = (
            SmartKeyPool.from_single_key(api_key)
            if isinstance(api_key, str)
            else api_key
        )
        llm_extras: dict = {}
        if cfg.model.base_url:
            llm_extras["base_url"] = cfg.model.base_url
        judge_llm = LLMFactory.create(
            provider=cfg.model.provider,
            model_name=cfg.model.model_name,
            api_key=key_pool,
            **llm_extras,
        )
        from .evaluator import default_embedding_provider_config

        embed_cfg = default_embedding_provider_config(cfg.model.provider)
        judge = ConversationJudge.from_factory(
            judge_llm,
            key_pool=key_pool,
            model_provider_name=cfg.model.provider,
            embedding_provider_config=embed_cfg,
        )

        pref_gen = gen.to_preference_generator(judge=judge, config=pref_config)

        pairs, analytics = asyncio.run(pref_gen.generate())
        pref_gen.save_pairs(pairs, analytics)

    except Exception as exc:
        click.secho(f"Preference generation failed: {exc}", fg="red", err=True)
        raise SystemExit(1)

    elapsed = time.time() - start
    click.secho(
        f"Done! Generated {len(pairs)} preference pairs in {elapsed:.1f}s",
        fg="green",
    )
    click.echo(f"Output: {pref_config.output_path}")

    # Print stats
    if analytics.total_attempted > 0:
        click.echo(f"  Attempted:    {analytics.total_attempted}")
        click.echo(f"  Valid pairs:  {analytics.total_valid}")
        click.echo(
            f"  Discarded:    {analytics.total_discarded} "
            f"({analytics.discard_rate:.1%} discard rate)"
        )
    for warning in analytics.warnings:
        click.secho(f"  Warning: {warning}", fg="yellow")


def _print_preference_plan(cfg: AfterImageConfig, pref_config) -> None:
    """Print what preference generation would do."""
    click.echo("=== Preference Generation Plan ===")
    click.echo(f"  Model:          {cfg.model.provider} / {cfg.model.model_name}")
    click.echo(f"  Target pairs:   {pref_config.num_pairs}")
    click.echo(f"  Responses/prompt: {pref_config.num_responses}")
    click.echo(f"  Strategy:       {pref_config.strategy}")
    click.echo(f"  Min score gap:  {pref_config.min_score_gap}")
    click.echo(f"  Multi-turn:     {'yes' if pref_config.multi_turn else 'no'}")
    click.echo(f"  Output format:  {pref_config.output_format}")
    click.echo(f"  Output path:    {pref_config.output_path}")
    click.echo(f"  Save log:       {'yes' if pref_config.save_log else 'no'}")
    if cfg.documents:
        click.echo(
            f"  Documents:      {cfg.documents.provider} @ {cfg.documents.path or cfg.documents.collection}"
        )
    else:
        click.echo("  Documents:      none")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_auto_analyze(cfg: AfterImageConfig) -> None:
    """Run analytics after generation. Never raises — errors are logged."""
    try:
        from .analytics import DatasetAnalyzer, generate_report

        output_path = cfg.analytics.output_path
        if output_path is None:
            output_path = str(Path(cfg.output.path).with_suffix(".html"))

        report = DatasetAnalyzer.from_jsonl(cfg.output.path)
        generate_report(report, output_path)
        click.secho(f"Analytics report: {output_path}", fg="cyan")
    except Exception as exc:
        click.secho(f"Analytics failed (non-blocking): {exc}", fg="yellow", err=True)


def _run_auto_export(cfg: AfterImageConfig) -> None:
    """Run auto-export after generation. Never raises — errors are logged."""
    try:
        from .integrations import get_exporter

        export_cfg = cfg.output.export
        if export_cfg is None:
            return

        inp = Path(cfg.output.path)
        out_dir = Path(export_cfg.output_dir) if export_cfg.output_dir else inp.parent

        for fmt in export_cfg.formats:
            exporter = get_exporter(fmt)
            if export_cfg.split is not None:
                _export_with_split(
                    exporter,
                    inp,
                    out_dir,
                    fmt,
                    export_cfg.split,
                    export_cfg.shuffle,
                    export_cfg.seed,
                )
            else:
                out_path = out_dir / f"{inp.stem}_{fmt}.jsonl"
                exporter.export_file(inp, out_path)
            click.secho(f"  Auto-exported: {fmt}", fg="cyan")
    except Exception as exc:
        click.secho(f"Auto-export failed (non-blocking): {exc}", fg="yellow", err=True)


def _print_formats_table(formats: list[dict]) -> None:
    """Print a formatted table of available export formats."""
    click.echo("Available export formats:")
    click.echo("-" * 72)
    click.echo(f"{'Name':<16} {'Multi-turn':<12} {'System':<8} {'Tools':<8} Used by")
    click.echo("-" * 72)
    for f in formats:
        mt = "yes" if f["multi_turn"] else "-"
        sp = "yes" if f["system_prompt"] else "-"
        tc = "yes" if f["tool_calls"] else "-"
        click.echo(f"{f['name']:<16} {mt:<12} {sp:<8} {tc:<8} {f['used_by']}")
    click.echo("-" * 72)


def _jsonl_nonempty_line_starts(path: Path) -> list[int]:
    """Return byte offsets of each non-empty line in *path* (JSONL).

    Offsets are taken in binary mode so :func:`seek` / :func:`readline` stay aligned
    with the original file on all platforms.
    """
    starts: list[int] = []
    with open(path, "rb") as f:
        while True:
            pos = f.tell()
            chunk = f.readline()
            if not chunk:
                break
            if chunk.strip():
                starts.append(pos)
    return starts


def _export_with_split(
    exporter,
    input_path: Path,
    output_dir: Path,
    fmt: str,
    split_ratio: float,
    shuffle: bool,
    seed: int,
    *,
    system_prompt: str | None = None,
):
    """Export with train/val split. Returns the ExportResult for train."""
    import json
    import random

    from .integrations.base import ExportResult

    output_dir.mkdir(parents=True, exist_ok=True)

    line_starts = _jsonl_nonempty_line_starts(input_path)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(line_starts)

    n_val = max(1, int(len(line_starts) * split_ratio)) if line_starts else 0
    val_starts = line_starts[:n_val]
    train_starts = line_starts[n_val:]

    train_path = output_dir / f"{input_path.stem}_{fmt}_train.jsonl"
    val_path = output_dir / f"{input_path.stem}_{fmt}_val.jsonl"

    result = ExportResult(
        format_name=fmt,
        input_path=str(input_path),
        output_path=str(train_path),
    )
    result.total_input = len(line_starts)

    with open(input_path, "rb") as fin:
        for out_path, subset in [(train_path, train_starts), (val_path, val_starts)]:
            train_shard = out_path == train_path
            with open(out_path, "w", encoding="utf-8") as fout:
                for start in subset:
                    fin.seek(start)
                    raw_line = fin.readline().decode("utf-8").strip()
                    try:
                        row = json.loads(raw_line)
                        converted = exporter.convert_conversation(
                            row,
                            system_prompt=system_prompt,
                        )
                        for out_row in converted:
                            fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                            result.total_output += 1
                            if train_shard:
                                result.train_export_rows += 1
                            else:
                                result.val_export_rows += 1
                    except Exception as exc:
                        result.skipped += 1
                        result.warnings.append(str(exc))

    click.secho(
        f"  {fmt}: {len(train_starts)} train + {len(val_starts)} val "
        f"-> {train_path.name}, {val_path.name}",
        fg="green",
    )

    return result
