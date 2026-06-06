import re

# Singapore target groups for structured hate speech generation
SG_TARGET_GROUPS = {
    "chinese": "Chinese Singaporeans (majority race, includes dialect groups: Hokkien, Teochew, Cantonese, Hakka, Hainanese)",
    "malay": "Malay Singaporeans (includes Muslim Malays)",
    "indian": "Indian Singaporeans (includes Tamil, Hindi, Sikh, Malayalee communities)",
    "others": "Eurasians, mixed-race, and other minority Singaporeans",
    "migrant_worker": "Foreign migrant workers (Bangladeshi, Chinese nationals/PRCs, Vietnamese, Filipino, Myanmar, Indonesian)",
    "religion": "Religious groups (Islam, Christianity, Buddhism, Hinduism, Taoism, Sikhism)",
}

# Map language origins to likely associated communities (loose heuristic)
LANG_TO_COMMUNITY = {
    "hokkien": "chinese", "teochew": "chinese", "cantonese": "chinese",
    "hakka": "chinese", "hainanese": "chinese", "mandarin": "chinese",
    "min nan": "chinese", "chinese": "chinese",
    "malay": "malay", "bahasa": "malay", "jawi": "malay",
    "tamil": "indian", "hindi": "indian", "punjabi": "indian",
    "sanskrit": "indian", "urdu": "indian",
    "bengali": "migrant_worker", "tagalog": "migrant_worker",
    "filipino": "migrant_worker", "vietnamese": "migrant_worker",
    "burmese": "migrant_worker", "myanmar": "migrant_worker",
    "arabic": "religion",
}


def extract_languages(raw_markdown: str) -> list[str]:
    all_langs = list(LANG_TO_COMMUNITY.keys()) + [
        "singlish", "english", "indonesian", "javanese", "japanese",
    ]
    pattern = r'(?:from|←)\s*\n*\s*(' + '|'.join(re.escape(l) for l in all_langs) + r')'
    matches = re.findall(pattern, raw_markdown.lower())
    seen = set()
    unique = []
    for m in matches:
        if m not in seen and m not in ("singlish", "english"):
            seen.add(m)
            unique.append(m)
    return unique


def infer_target_groups(languages: list[str]) -> list[str]:
    groups = set()
    for lang in languages:
        if lang in LANG_TO_COMMUNITY:
            groups.add(LANG_TO_COMMUNITY[lang])
    return list(groups) if groups else list(SG_TARGET_GROUPS.keys())


def _clean_markdown(text: str) -> str:
    text = re.sub(r'\[_?([^\]]*?)_?\]\([^)]+\)', r'\1', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'[_*]{1,2}([^_*]+)[_*]{1,2}', r'\1', text)
    text = re.sub(r'^\* \* \*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^---+$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\[[ x]\]', '', text)
    text = re.sub(r'^#{1,4}\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^[\u2E80-\u9FFF\u3400-\u4DBF\U00020000-\U0002FA1F\s]+$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^[a-zA-Zāáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜ/]+$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\(poj\)', '', text)
    text = re.sub(r'简.*?繁', '', text)
    text = re.sub(r'^[→←]$', '', text, flags=re.MULTILINE)
    text = re.sub(r'expand for details.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Detailed etymology', '', text)
    text = re.sub(r'\d+\s*Thumbs up.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Share\s*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r'!\[.*?\]', '', text)
    text = re.sub(r'^\+$', '', text, flags=re.MULTILINE)
    text = re.sub(r'▼', '', text)
    text = re.sub(r'^(Singlish|from|singlish|english|min nan|hokkien|malay|tamil):?\s*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r'^-\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = '\n'.join(line.strip() for line in text.split('\n'))
    return text.strip()


def extract_context(raw_markdown: str) -> str:
    lines = raw_markdown.split("\n")
    start = None
    for i, line in enumerate(lines):
        if "### definitions" in line.lower():
            start = i
            break
    if start is None:
        return ""

    relevant = []
    skip_sections = {
        "ko-fi", "buy me", "suggest a word", "submit media",
        "entries starting with", "link copied",
    }
    for line in lines[start:]:
        ll = line.lower().strip()
        if any(x in ll for x in ["facebook", "twitter.com/intent", "whatsapp.com",
                                  "t.me/share", "reddit.com/submit", "copyright",
                                  "login / register", "ko-fi.com", "forms.gle",
                                  "singlishdict.app/meta", "singlishdict.app/auth",
                                  "singlishdict.app/langs", "singlishdict.app/letter",
                                  "singlishdict.app/cat)"]):
            continue
        if line.strip() == "[](https://singlishdict.app/)":
            continue
        if any(x in ll for x in skip_sections):
            continue
        relevant.append(line)

    text = "\n".join(relevant)
    text = _clean_markdown(text)

    new_lines = []
    for line in text.split("\n"):
        ll = line.lower().strip()
        if ll in ("related:", "categories:"):
            continue
        if ll == "synonyms:":
            continue
        new_lines.append(line)
    text = "\n".join(new_lines)
    text = re.sub(r'\n{3,}', '\n\n', text)

    if len(text) > 1200:
        text = text[:1200] + "\n..."
    return text.strip()


def extract_context_from_parsed(entry: dict) -> str:
    definitions = (entry.get("definitions") or "").strip()
    usage_notes = (entry.get("usage_notes") or "").strip()
    synonyms = entry.get("synonyms") or []
    antonyms = entry.get("antonyms") or []

    parts = []
    if definitions:
        parts.append(f"Definitions:\n{definitions}")
    if usage_notes:
        parts.append(f"Usage notes:\n{usage_notes}")
    if synonyms:
        parts.append("Synonyms: " + ", ".join(synonyms))
    if antonyms:
        parts.append("Antonyms: " + ", ".join(antonyms))

    text = "\n\n".join(parts).strip()
    if len(text) > 1200:
        text = text[:1200] + "\n..."
    return text


def normalize_entry(entry: dict) -> dict:
    categories = entry.get("source_categories") or entry.get("categories") or []
    if isinstance(categories, str):
        categories = [categories]

    if "raw_markdown" in entry:
        languages = extract_languages(entry.get("raw_markdown", ""))
        context = extract_context(entry.get("raw_markdown", ""))
    else:
        languages = entry.get("languages") or []
        context = extract_context_from_parsed(entry)

    return {
        "term": entry.get("term", "").strip(),
        "categories": categories,
        "languages": languages,
        "context": context,
    }