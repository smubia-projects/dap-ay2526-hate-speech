"""
Stage 1 Parser: Clean raw Jina markdown into structured sections.
=================================================================
Reads:  data/singlish/singlish_raw.jsonl  (raw scraped markdown)
Writes: data/singlish/singlish_parsed.jsonl (cleaned, sectioned)

Each output entry:
{
  "term": "chao keng",
  "categories": ["insults"],
  "languages": ["hokkien"],
  "alternatives": ["chao keng", "chaokeng"],
  "definitions": "verb\\n1. to malinger...\\n— 2018...",   # clean text blob
  "usage_notes": "Commonly used in NS...",                 # clean text or null
  "synonyms": ["keng", "siam"],
  "antonyms": ["on", "siao on"],
  "related_terms": ["kengster", "keng king"]
}

Usage:
  python scraping/parse_raw.py              # parse all, write JSONL
  python scraping/parse_raw.py --sample 5   # preview 5 random entries
"""

import argparse
import json
import random
import re
from pathlib import Path

RAW_FILE = Path(__file__).resolve().parent.parent / "data" / "singlish" / "singlish_raw.jsonl"
OUT_FILE = Path(__file__).resolve().parent.parent / "data" / "singlish" / "singlish_parsed.jsonl"

# Languages we care about for target-group inference
KNOWN_LANGS = [
    "hokkien", "teochew", "cantonese", "hakka", "hainanese", "mandarin",
    "min nan", "chinese", "malay", "bahasa", "jawi", "tamil", "hindi",
    "punjabi", "sanskrit", "urdu", "bengali", "tagalog", "filipino",
    "vietnamese", "burmese", "myanmar", "arabic", "japanese", "javanese",
    "indonesian", "thai", "korean",
]


# ── Step 1: Strip nav / header / footer ─────────────────────────────────────

def _strip_nav(md: str) -> str:
    """Remove everything before actual entry content."""
    marker = "Anyhow whack\n"
    idx = md.find(marker)
    if idx >= 0:
        md = md[idx + len(marker):]
    # Strip the [A-Z][Categories][Languages] breadcrumb line
    md = re.sub(r'^\[A-Z\].*?\n', '', md, count=1)
    return md


def _strip_footer(text: str) -> str:
    """Remove social share blocks and footer junk.
    There can be multiple share blocks (one in etymology, one after definitions).
    We remove ALL of them but keep content between them."""
    # Remove each share/social block (checkbox + Share + social links + Link copied)
    # These blocks look like: - [x]\n\nShare\n\n[](facebook...)[](twitter...)...Link copied!
    text = re.sub(
        r'- \[[ x]\]\s*\n\nShare\s*\n.*?(?:Link copied!|Thumbs up man!)',
        '', text, flags=re.DOTALL
    )
    # Remove standalone social link clusters
    text = re.sub(
        r'\[\]\(https://(?:www\.facebook|twitter|api\.whatsapp|t\.me|reddit)\.com[^)]*\)',
        '', text
    )
    # Clean up remnants
    for junk in ["Link copied!", "Thumbs up man!", "ko-fi.com", "forms.gle",
                 "Suggest a word"]:
        text = text.replace(junk, "")

    return text


# ── Step 2: Extract alternatives (variant spellings) ────────────────────────

def _extract_alternatives(text: str) -> tuple[list[str], str]:
    """Extract variant spellings from the 'alternatives:' block.
    Returns (alternatives_list, text_with_block_removed)."""
    alts = []

    # Pattern: "Commonly found variants..." disclaimer followed by "alternatives:" list
    disclaimer_re = re.compile(
        r'Commonly found variants.*?(?=alternatives:|### |\Z)', re.DOTALL | re.IGNORECASE
    )
    text = disclaimer_re.sub('', text)

    alt_re = re.compile(r'^alternatives:\s*\n(.*?)(?=^---|\n###|\n\S+\n---)', re.DOTALL | re.MULTILINE)
    m = alt_re.search(text)
    if m:
        block = m.group(1)
        alts = [line.strip() for line in block.split('\n') if line.strip()]
        text = text[:m.start()] + text[m.end():]

    # Also catch "redirected from\n\nterm" pattern
    redir_re = re.compile(r'^redirected from\s*\n+\s*(\S.*?)\s*\n', re.MULTILINE)
    rm = redir_re.search(text)
    if rm:
        alts.insert(0, rm.group(1).strip())
        text = text[:rm.start()] + text[rm.end():]

    return alts, text


# ── Step 3: Strip etymology block ───────────────────────────────────────────

def _strip_etymology(text: str) -> str:
    """Remove the etymology block (CJK, poj, arrow chains, etc.)
    between the term heading and ### definitions."""

    # Find where ### definitions starts
    def_idx = text.find('### definitions')
    if def_idx < 0:
        def_idx = text.find('### Definitions')
    if def_idx < 0:
        return text

    # Everything before ### definitions is preamble (term heading + etymology)
    # We keep the text after ### definitions intact
    preamble = text[:def_idx]
    rest = text[def_idx:]

    # From preamble, remove:
    # - CJK lines
    # - romanization (poj) lines
    # - arrow symbols (→, ←)
    # - standalone +, -, or/-- lines
    # - "#### word" etymology entries
    # - "from\n\nlanguage" blocks
    # - "expand for details" / "Detailed etymology" lines
    # - "### etymology notes" section
    # We basically discard the entire preamble since it's all etymology
    # The only useful thing in the preamble is the term heading line
    # but we already have the term from the entry metadata

    return rest


# ── Step 4: Clean markdown formatting ───────────────────────────────────────

def _clean_markdown(text: str) -> str:
    """Strip markdown syntax to plain text."""
    # Markdown links: [text](url) → text
    text = re.sub(r'\[_?([^\]]*?)_?\]\([^)]+\)', r'\1', text)
    # Bare URLs
    text = re.sub(r'https?://\S+', '', text)
    # Emphasis
    text = re.sub(r'[_*]{1,2}([^_*]+)[_*]{1,2}', r'\1', text)
    # Headers: ### foo → foo
    text = re.sub(r'^#{1,4}\s*', '', text, flags=re.MULTILINE)
    # Horizontal rules
    text = re.sub(r'^\* \* \*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^---+$', '', text, flags=re.MULTILINE)
    # ▼ expand markers
    text = re.sub(r'▼', '', text)
    # Image alt text artifacts
    text = re.sub(r'!\[.*?\]', '', text)
    # Checkbox artifacts
    text = re.sub(r'\[[ x]\]', '', text)
    # CJK lines (stray etymology characters)
    text = re.sub(r'^[\u2E80-\u9FFF\u3400-\u4DBF\U00020000-\U0002FA1F\s]+$', '', text, flags=re.MULTILINE)
    # Standalone romanization lines (bōe, chhió/chhiò, etc.)
    text = re.sub(r'^[a-zA-Zāáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜɔ̃ⁿ/()-]+$', '', text, flags=re.MULTILINE)
    # (poj) / (jyutping) annotations
    text = re.sub(r'\((?:poj|jyutping)\)', '', text)
    # 简/繁 toggle
    text = re.sub(r'简.*?繁', '', text)
    # Standalone arrows and connectors
    text = re.sub(r'^[→←+\-]$', '', text, flags=re.MULTILINE)
    # "or\n--" pattern from etymology
    text = re.sub(r'^or\n--$', '', text, flags=re.MULTILINE)
    # "expand for details" lines
    text = re.sub(r'expand for details.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Detailed etymology', '', text, flags=re.IGNORECASE)
    # Thumbs up / share remnants
    text = re.sub(r'\d+\s*Thumbs up.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Share\s*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
    # Social media link remnants: standalone "(Facebook)" etc.
    text = re.sub(r'^\("?\w+"?\)\s*$', '', text, flags=re.MULTILINE)
    # Language-only lines from etymology (standalone "Singlish", "english", etc.)
    text = re.sub(
        r'^(Singlish|English|Min [Nn]an|Hokkien|Malay|Tamil|Cantonese|'
        r'Mandarin|Teochew|Hakka|Hainanese|Hindi|Sanskrit|Arabic|'
        r'Japanese|Javanese|Indonesian|Korean|Thai|Latin|French|'
        r'Old [Ff]rench|Portuguese|Dutch|Bahasa|Bengali):?\s*$',
        '', text, flags=re.MULTILINE | re.IGNORECASE
    )
    # Collapse multiple blank lines → single blank
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Strip per-line whitespace
    text = '\n'.join(line.strip() for line in text.split('\n'))
    return text.strip()


# ── Step 5: Split into sections ─────────────────────────────────────────────

def _split_sections(text: str) -> dict[str, str]:
    """Split text into named sections by ### headers (before markdown cleaning)."""
    section_re = re.compile(
        r'^###\s+(definitions|usage notes|related:|categories:)',
        re.IGNORECASE | re.MULTILINE
    )

    sections = {}
    matches = list(section_re.finditer(text))

    for i, m in enumerate(matches):
        name = m.group(1).lower().rstrip(':').strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections[name] = text[start:end].strip()

    return sections


# ── Step 6: Extract synonyms / antonyms / related as lists ──────────────────

def _extract_term_list(text: str) -> list[str]:
    """Extract a list of terms from a block that may be run-together link text."""
    # If there are still markdown links, extract from those
    link_terms = re.findall(r'\[([^\]]+)\]', text)
    if link_terms:
        return [t.strip() for t in link_terms if t.strip()]
    # Otherwise try splitting on known separators
    terms = re.split(r'[,|;\n]+', text)
    return [t.strip() for t in terms if t.strip() and len(t.strip()) > 1]


def _extract_syn_ant(defs_text: str) -> tuple[list[str], list[str], str]:
    """Extract synonyms and antonyms from the definitions text blob.
    Returns (synonyms, antonyms, cleaned_defs_text)."""
    all_syns = []
    all_ants = []
    cleaned_lines = []

    lines = defs_text.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip().lower()

        if line == 'synonyms:':
            # Collect following lines (skip blanks) until next section boundary
            i += 1
            block = []
            while i < len(lines):
                sl = lines[i].strip()
                if not sl:
                    i += 1
                    continue  # skip blank lines within synonym block
                sl_lower = sl.lower()
                if sl_lower in ('antonyms:', 'synonyms:') or \
                   re.match(r'^\d+\.', sl) or sl.startswith('####') or \
                   sl.startswith('###'):
                    break
                block.append(lines[i])
                i += 1
            all_syns.extend(_extract_term_list('\n'.join(block)))
            continue

        if line == 'antonyms:':
            i += 1
            block = []
            while i < len(lines):
                al = lines[i].strip()
                if not al:
                    i += 1
                    continue  # skip blank lines
                al_lower = al.lower()
                if al_lower in ('synonyms:', 'antonyms:') or \
                   re.match(r'^\d+\.', al) or al.startswith('####') or \
                   al.startswith('###'):
                    break
                block.append(lines[i])
                i += 1
            all_ants.extend(_extract_term_list('\n'.join(block)))
            continue

        cleaned_lines.append(lines[i])
        i += 1

    # Deduplicate preserving order
    syn_seen, ant_seen = set(), set()
    syns, ants = [], []
    for s in all_syns:
        if s.lower() not in syn_seen:
            syn_seen.add(s.lower())
            syns.append(s)
    for a in all_ants:
        if a.lower() not in ant_seen:
            ant_seen.add(a.lower())
            ants.append(a)

    return syns, ants, '\n'.join(cleaned_lines).strip()


# ── Step 7: Extract languages ───────────────────────────────────────────────

def _extract_languages(md: str) -> list[str]:
    """Extract language-of-origin from the raw markdown."""
    pattern = r'(?:from|←)\s*\n*\s*(' + '|'.join(re.escape(l) for l in KNOWN_LANGS) + r')'
    matches = re.findall(pattern, md.lower())
    seen = set()
    result = []
    for m in matches:
        if m not in seen:
            seen.add(m)
            result.append(m)
    return result


# ── Main parse function ─────────────────────────────────────────────────────

def parse_entry(entry: dict) -> dict:
    """Stage 1: raw markdown → cleaned sections."""
    term = entry["term"]
    cats = entry["source_categories"]
    md = entry["raw_markdown"]

    # Languages from raw (before stripping etymology)
    languages = _extract_languages(md)

    # Pipeline
    text = _strip_nav(md)
    text = _strip_footer(text)
    alts, text = _extract_alternatives(text)
    text = _strip_etymology(text)

    # Split into sections BEFORE cleaning markdown (### headers still present)
    sections = _split_sections(text)

    # Definitions blob — extract syn/ant from RAW (links still present), then clean
    defs_raw = sections.get("definitions", "")
    synonyms, antonyms, defs_stripped = _extract_syn_ant(defs_raw)
    defs_text = _clean_markdown(defs_stripped)

    # Clean up definitions: collapse excess blanks
    defs_text = re.sub(r'\n{3,}', '\n\n', defs_text).strip()

    # Usage notes — clean markdown
    usage_raw = sections.get("usage notes", "")
    usage = _clean_markdown(usage_raw).strip() or None

    # Related terms
    related_raw = sections.get("related", "")
    related = _extract_term_list(related_raw)

    return {
        "term": term,
        "categories": cats,
        "languages": languages,
        "alternatives": alts if alts else None,
        "definitions": defs_text if defs_text else None,
        "usage_notes": usage,
        "synonyms": synonyms if synonyms else None,
        "antonyms": antonyms if antonyms else None,
        "related_terms": related if related else None,
    }


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stage 1: parse raw markdown → clean sections")
    parser.add_argument("--sample", "-s", type=int, help="Preview N random entries instead of writing all")
    args = parser.parse_args()

    entries = [json.loads(l) for l in open(RAW_FILE) if l.strip()]
    print(f"Loaded {len(entries)} raw entries")

    if args.sample:
        random.seed(42)
        sample = random.sample(entries, min(args.sample, len(entries)))
        for i, entry in enumerate(sample, 1):
            parsed = parse_entry(entry)
            print(f"\n{'='*70}")
            print(json.dumps(parsed, indent=2, ensure_ascii=False))
        # Quick stats
        print(f"\n{'='*70}")
        all_parsed = [parse_entry(e) for e in entries]
        no_defs = sum(1 for p in all_parsed if not p["definitions"])
        print(f"Entries without definitions: {no_defs}/{len(entries)}")
    else:
        OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT_FILE, "w") as f:
            for entry in entries:
                parsed = parse_entry(entry)
                f.write(json.dumps(parsed, ensure_ascii=False) + "\n")
        print(f"Wrote {len(entries)} parsed entries → {OUT_FILE}")


if __name__ == "__main__":
    main()
