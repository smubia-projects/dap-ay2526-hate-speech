"""
Singlish Dictionary Scraper (Raw)
==================================
Scrapes entries from singlishdict.app using Jina Reader API.
Saves RAW markdown per entry — parsing/formatting is a separate step.

Two phases:
1. Collect term lists from category/letter pages
2. Fetch raw markdown for each term via Jina Reader

Output:
- data/singlish/singlish_raw.jsonl  → {term, source_categories, raw_markdown}

Usage:
    # Default: scrape insults + vulgar categories
    python scraping/singlish_scraper.py

    # Scrape ALL terms A-Z
    python scraping/singlish_scraper.py --all

    # Specific categories or letters
    python scraping/singlish_scraper.py --categories insults vulgar "online slang"
    python scraping/singlish_scraper.py --letters A B C

    # Resume is automatic — re-run and it skips already-scraped terms
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import quote_plus, unquote_plus

import requests
from dotenv import load_dotenv

# ── Config ──────────────────────────────────────────────────────────────────

BASE_URL = "https://singlishdict.app"
JINA_READER_PREFIX = "https://r.jina.ai/"
RATE_LIMIT_DELAY = 1.5  # seconds between Jina API calls
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "singlish"

LETTERS = ["0-9"] + [chr(i) for i in range(ord("A"), ord("Z") + 1)]


# ── Phase 1: Collect term lists ─────────────────────────────────────────────

def fetch_listing(url: str, jina_key: str) -> str:
    """Fetch a listing page via Jina Reader."""
    headers = {"Authorization": f"Bearer {jina_key}"}
    resp = requests.get(f"{JINA_READER_PREFIX}{url}", headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.text


def extract_terms_from_listing(markdown_text: str) -> list[str]:
    """Extract term names from listing markdown.
    Links look like: [term](https://singlishdict.app/?q=term+name#anchor)
    """
    pattern = r'\[([^\]]+)\]\(https://singlishdict\.app/\?q=([^)#]+)[^)]*\)'
    matches = re.findall(pattern, markdown_text)
    seen = set()
    terms = []
    for _, url_param in matches:
        term = unquote_plus(url_param)
        if term not in seen:
            seen.add(term)
            terms.append(term)
    return terms


def collect_terms(categories: list[str] | None, letters: list[str] | None,
                  scrape_all: bool, jina_key: str) -> dict[str, list[str]]:
    """Collect terms and track which categories they came from.
    Returns {term: [source_categories]}
    """
    term_sources: dict[str, list[str]] = {}

    sources = []
    if scrape_all:
        sources = [("letter", l) for l in LETTERS]
    else:
        if categories:
            sources += [("category", c) for c in categories]
        if letters:
            sources += [("letter", l) for l in letters]

    for source_type, source_name in sources:
        if source_type == "category":
            url = f"{BASE_URL}/cat?c={quote_plus(source_name)}"
        else:
            url = f"{BASE_URL}/letter?l={quote_plus(source_name)}"

        print(f"  Fetching {source_type}: {source_name}...", end=" ")
        md = fetch_listing(url, jina_key)
        terms = extract_terms_from_listing(md)
        print(f"→ {len(terms)} terms")

        for t in terms:
            if t not in term_sources:
                term_sources[t] = []
            if source_type == "category":
                term_sources[t].append(source_name)

        time.sleep(RATE_LIMIT_DELAY)

    return term_sources


# ── Phase 2: Fetch raw entries ──────────────────────────────────────────────

def fetch_entry(term: str, jina_key: str, max_retries: int = 3) -> str | None:
    """Fetch a single dictionary entry via Jina Reader (with retries)."""
    url = f"{JINA_READER_PREFIX}{BASE_URL}/?q={quote_plus(term)}"
    headers = {"Authorization": f"Bearer {jina_key}"}

    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as e:
            if attempt < max_retries - 1:
                wait = (attempt + 1) * 2
                print(f"\n    ⚠ Retry {attempt+1}/{max_retries} for '{term}' (waiting {wait}s)")
                time.sleep(wait)
            else:
                print(f"\n    ✗ Failed after {max_retries} attempts: {e}")
                return None


# ── Resume support ──────────────────────────────────────────────────────────

def load_already_scraped(output_file: Path) -> set[str]:
    """Load terms already in the output file."""
    scraped = set()
    if output_file.exists():
        with open(output_file, "r") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    scraped.add(entry["term"])
    return scraped


# ── Main ────────────────────────────────────────────────────────────────────

def run_scraper(categories=None, letters=None, scrape_all=False):
    env_path = Path(__file__).parent / ".env"
    load_dotenv(env_path)
    jina_key = os.getenv("JINA_KEY") or os.getenv("JINA_API_KEY")
    if not jina_key:
        print("ERROR: Set JINA_KEY in scraping/.env")
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_file = DATA_DIR / "singlish_raw.jsonl"

    # Phase 1
    print("=" * 60)
    print("PHASE 1: Collecting term lists")
    print("=" * 60)
    term_sources = collect_terms(categories, letters, scrape_all, jina_key)
    print(f"\nTotal unique terms: {len(term_sources)}")

    # Resume
    already_done = load_already_scraped(output_file)
    if already_done:
        print(f"Already scraped: {len(already_done)} (skipping)")
    remaining = [t for t in term_sources if t not in already_done]
    print(f"Terms remaining: {len(remaining)}")

    if not remaining:
        print("All done!")
        return

    # Phase 2
    print("\n" + "=" * 60)
    print("PHASE 2: Fetching entries via Jina Reader")
    print("=" * 60)

    fh = open(output_file, "a")
    success = failed = 0

    try:
        for i, term in enumerate(remaining):
            pct = (i + 1) / len(remaining) * 100
            print(f"  [{i+1}/{len(remaining)}] ({pct:.1f}%) {term}...", end=" ", flush=True)

            raw_md = fetch_entry(term, jina_key)
            if raw_md is None:
                failed += 1
                print("FAILED")
                continue

            row = {
                "term": term,
                "source_categories": term_sources.get(term, []),
                "raw_markdown": raw_md,
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

            success += 1
            # Quick sanity: count how many "### definitions" sections are in the page
            n_def_sections = raw_md.lower().count("### definitions")
            print(f"OK ({len(raw_md)} chars, {n_def_sections} def sections)")

            if (i + 1) % 10 == 0:
                fh.flush()

            time.sleep(RATE_LIMIT_DELAY)

    except KeyboardInterrupt:
        print("\n\nInterrupted! Progress saved. Re-run to resume.")
    finally:
        fh.close()

    print("\n" + "=" * 60)
    print(f"DONE — {success} ok, {failed} failed")
    print(f"Output: {output_file}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Scrape singlishdict.app via Jina Reader")
    parser.add_argument("--categories", "-c", nargs="+", help="Categories to scrape")
    parser.add_argument("--letters", "-l", nargs="+", help="Letters to scrape")
    parser.add_argument("--all", "-a", action="store_true", help="Scrape ALL terms A-Z")
    args = parser.parse_args()

    if not args.categories and not args.letters and not args.all:
        print("No args → defaulting to --categories insults vulgar")
        args.categories = ["insults", "vulgar"]

    run_scraper(categories=args.categories, letters=args.letters, scrape_all=args.all)


if __name__ == "__main__":
    main()
