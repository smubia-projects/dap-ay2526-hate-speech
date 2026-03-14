"""Synthetic data generation CLI (orchestration-only entrypoint)."""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import requests

try:
    from .synthetic_generation import build_prompt, call_llm, parse_llm_output, validate_examples
    from .synthetic_processing import infer_target_groups, normalize_entry
except ImportError:
    from synthetic_generation import build_prompt, call_llm, parse_llm_output, validate_examples
    from synthetic_processing import infer_target_groups, normalize_entry

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "singlish"
PARSED_FILE = DATA_DIR / "singlish_parsed.jsonl"
OUTPUT_FILE = DATA_DIR / "synthetic_training.jsonl"

DEFAULT_OLLAMA_URL = "http://localhost:11434"
MAX_RETRIES_PER_TERM = 3


def load_already_generated(output_file: Path) -> set[str]:
    done = set()
    if output_file.exists():
        with open(output_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    done.add(entry.get("source_term", ""))
    return done


def run(model: str, n_per_label: int, limit: int | None, dry_run: bool,
        backend: str = "ollama", api_url: str = DEFAULT_OLLAMA_URL,
        api_key: str = "EMPTY", input_file: Path = PARSED_FILE,
        selected_terms: list[str] | None = None):
    if not input_file.exists():
        print(f"ERROR: {input_file} not found.")
        sys.exit(1)

    entries = []
    with open(input_file, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(normalize_entry(json.loads(line)))
    print(f"Loaded {len(entries)} dictionary entries from {input_file.name}")

    entries = [e for e in entries if e["term"]]
    if selected_terms:
        wanted = [t.strip() for t in selected_terms if t.strip()]
        entry_map = {e["term"].casefold(): e for e in entries}
        missing = [t for t in wanted if t.casefold() not in entry_map]
        if missing:
            print(f"WARNING: {len(missing)} selected terms not found: {', '.join(missing)}")
        entries = [entry_map[t.casefold()] for t in wanted if t.casefold() in entry_map]
        print(f"Selected {len(entries)} terms from filter list")

    already_done = load_already_generated(OUTPUT_FILE)
    if already_done:
        print(f"Already generated for {len(already_done)} terms (skipping)")

    remaining = [e for e in entries if e["term"] not in already_done]
    if limit:
        remaining = remaining[:limit]

    print(f"Will generate for {len(remaining)} terms ({n_per_label * 3} examples each)")
    print(f"Backend: {backend} | Model: {model}")
    print(f"API URL: {api_url}")
    print(f"Expected output: ~{len(remaining) * n_per_label * 3} examples")

    if dry_run:
        for entry in remaining[:3]:
            ctx = entry["context"]
            langs = entry["languages"]
            groups = infer_target_groups(langs)
            prompt = build_prompt(entry["term"], entry["categories"], ctx,
                                  n_per_label, groups, langs)
            print(f"\n{'=' * 60}")
            print(f"TERM: {entry['term']}")
            print(f"Languages: {langs} → Target groups: {groups}")
            print(f"{'=' * 60}")
            print(prompt)
        return

    if not remaining:
        print("Nothing to generate!")
        return

    try:
        if backend == "ollama":
            requests.get(f"{api_url}/api/tags", timeout=5)
        else:
            requests.get(f"{api_url}/models", timeout=5,
                         headers={"Authorization": f"Bearer {api_key}"})
    except requests.ConnectionError:
        print(f"ERROR: Cannot reach {backend} at {api_url}")
        print("Start it with: ollama serve" if backend == "ollama" else "Check your --api-url and that the server is running.")
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    total_generated = 0
    failed = 0

    with open(OUTPUT_FILE, "a", encoding="utf-8") as fh:
        try:
            for i, entry in enumerate(remaining):
                term = entry["term"]
                cats = entry["categories"]
                if "..." in term or "…" in term:
                    print(f"  [{i + 1}/{len(remaining)}] {term}... SKIP (pattern entry)")
                    continue

                ctx = entry["context"]
                if not ctx:
                    print(f"  [{i + 1}/{len(remaining)}] {term}... SKIP (no definitions)")
                    continue

                langs = entry["languages"]
                groups = infer_target_groups(langs)
                prompt = build_prompt(term, cats, ctx, n_per_label, groups, langs)
                print(f"  [{i + 1}/{len(remaining)}] {term} (→{','.join(groups)})...", end=" ", flush=True)

                best_examples = []
                dropped = 0
                raw_response = None
                for attempt in range(MAX_RETRIES_PER_TERM + 1):
                    raw_response = call_llm(prompt, model, backend, api_url, api_key)
                    if raw_response is None:
                        break

                    examples = parse_llm_output(raw_response)
                    if not examples:
                        break

                    valid, dropped = validate_examples(examples, term)
                    best_examples = valid
                    if dropped == 0:
                        break
                    if attempt < MAX_RETRIES_PER_TERM:
                        print(f"\n    ↻ retry {attempt + 1} ({dropped} examples missing term)...", end=" ", flush=True)

                if not best_examples:
                    failed += 1
                    print("FAILED (LLM error)" if raw_response is None else "FAILED (no valid examples)")
                    continue

                for ex in best_examples:
                    row = {
                        "text": ex["text"],
                        "label": ex["label"],
                        "target_group": ex.get("target_group", "none"),
                        "source_term": term,
                        "source_category": cats[0] if cats else "",
                    }
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")

                total_generated += len(best_examples)
                label_counts = {}
                for ex in best_examples:
                    label_counts[ex["label"]] = label_counts.get(ex["label"], 0) + 1
                dropped_str = f", {dropped} dropped" if dropped else ""
                print(f"OK → {len(best_examples)} examples ({label_counts}{dropped_str})")

                if (i + 1) % 5 == 0:
                    fh.flush()
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\n\nInterrupted! Progress saved. Re-run to resume.")

    print(f"\n{'=' * 60}")
    print(f"DONE — {total_generated} examples generated, {failed} failed")
    print(f"Output: {OUTPUT_FILE}")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic hate speech training data")
    parser.add_argument("--model", "-m", default="qwen3:8b", help="Model name")
    parser.add_argument("--backend", "-b", choices=["ollama", "openai"], default="ollama",
                        help="LLM backend: 'ollama' (local) or 'openai' (vLLM/TGI/etc.)")
    parser.add_argument("--api-url", default=DEFAULT_OLLAMA_URL,
                        help="API base URL (default: localhost ollama)")
    parser.add_argument("--api-key", default="EMPTY",
                        help="API key for openai backend (default: EMPTY for vLLM)")
    parser.add_argument("--per-term", "-n", type=int, default=2,
                        help="Examples per label per term (total = n*3). Default: 2 → 6 per term")
    parser.add_argument("--limit", "-l", type=int, help="Max terms to process")
    parser.add_argument("--input-file", default=str(PARSED_FILE),
                        help="Input JSONL path (default: data/singlish/singlish_parsed.jsonl)")
    parser.add_argument("--terms", default="",
                        help="Comma-separated list of terms to generate (case-insensitive)")
    parser.add_argument("--dry-run", "-d", action="store_true", help="Preview prompts only")
    args = parser.parse_args()

    api_url = args.api_url
    if args.backend == "openai" and api_url == DEFAULT_OLLAMA_URL:
        print("ERROR: --api-url required when using --backend openai")
        print("Example: --api-url https://your-ngrok-url.ngrok.io/v1")
        sys.exit(1)

    selected_terms = None
    if args.terms.strip():
        selected_terms = [t.strip() for t in re.split(r"[,\n;]", args.terms) if t.strip()]

    run(model=args.model, n_per_label=args.per_term, limit=args.limit,
        dry_run=args.dry_run, backend=args.backend, api_url=api_url,
        api_key=args.api_key, input_file=Path(args.input_file),
        selected_terms=selected_terms)


if __name__ == "__main__":
    main()
