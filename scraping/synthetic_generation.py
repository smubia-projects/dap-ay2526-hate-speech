import json
import re

import requests

try:
    from .synthetic_processing import SG_TARGET_GROUPS
except ImportError:
    from synthetic_processing import SG_TARGET_GROUPS

SYSTEM_PROMPT = """You generate Singapore social media posts for a hate speech detection research dataset.

You will be given a Singlish term and its meaning. Write posts that use that exact term.

Output ONLY a JSON array. No markdown, no explanation, no extra text.

Each object: {"text": "...", "label": "hate|offensive|normal", "target_group": "..."}

Labels:
- hate: targets a race/religion/nationality (chinese, malay, indian, migrant_worker, islam, etc.)
- offensive: rude but not targeting a protected group. target_group = "none"
- normal: neutral usage. target_group = "none"

Language rule (strict): Write in English only. You may use Singlish particles (lah, leh, lor, sia) and the given term, but do not write full Malay/Chinese/Tamil sentences.

Style: Singlish, 1-3 sentences, use particles (lah, leh, lor, sia). Reference SG things (HDB, MRT, NS, kopitiam)."""


def build_prompt(term: str, categories: list[str], context: str,
                 n_per_label: int, target_groups: list[str],
                 languages: list[str]) -> str:
    cat_str = ", ".join(categories) if categories else "general"
    lang_str = ", ".join(languages) if languages else "mixed/unclear"

    group_descriptions = []
    for g in target_groups:
        if g in SG_TARGET_GROUPS:
            group_descriptions.append(f"- {g}: {SG_TARGET_GROUPS[g]}")
    group_block = "\n".join(group_descriptions) if group_descriptions else "- any Singapore racial/religious/migrant group"

    return f"""Write {n_per_label * 3} Singapore social media posts that all use the Singlish term "{term}".

The word/phrase "{term}" MUST be written in every single post. This is mandatory.

What "{term}" means:
{context}

Categories: {cat_str} | Language origins: {lang_str}

For "hate" posts, target one of these groups:
{group_block}

Write exactly:
- {n_per_label} "hate" posts (targeting a specific racial/religious/migrant group)
- {n_per_label} "offensive" posts (rude but NOT targeting a protected group)
- {n_per_label} "normal" posts (neutral/benign usage)

Rules: every post MUST contain "{term}". Plain text only. Write in English Singlish only (English sentence structure; Singlish particles allowed). Keep each post 1-3 sentences.
Do NOT output full Malay/Chinese/Tamil sentences.

JSON array only, no other text:
[
  {{"text": "... {term} ...", "label": "hate", "target_group": "malay"}},
  {{"text": "... {term} ...", "label": "offensive", "target_group": "none"}},
  {{"text": "... {term} ...", "label": "normal", "target_group": "none"}}
]"""


def generate_with_ollama(prompt: str, model: str, api_url: str,
                         temperature: float = 0.8) -> str | None:
    payload = {
        "model": model,
        "prompt": prompt,
        "system": SYSTEM_PROMPT,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": temperature,
            "num_predict": 2048,
        },
    }
    try:
        resp = requests.post(f"{api_url}/api/generate", json=payload, timeout=180)
        resp.raise_for_status()
        return resp.json().get("response", "")
    except requests.RequestException as e:
        print(f"\n    ERROR calling ollama: {e}")
        return None


def generate_with_openai(prompt: str, model: str, api_url: str,
                         api_key: str = "EMPTY",
                         temperature: float = 0.8) -> str | None:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": 2048,
        "response_format": {"type": "json_object"},
    }
    try:
        url = f"{api_url}/chat/completions"
        resp = requests.post(url, json=payload, headers=headers, timeout=180)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except (requests.RequestException, KeyError, IndexError) as e:
        print(f"\n    ERROR calling OpenAI API: {e}")
        return None


def call_llm(prompt: str, model: str, backend: str, api_url: str,
             api_key: str = "EMPTY") -> str | None:
    if backend == "ollama":
        return generate_with_ollama(prompt, model, api_url)
    if backend == "openai":
        return generate_with_openai(prompt, model, api_url, api_key)
    print(f"Unknown backend: {backend}")
    return None


def term_appears_in_text(term: str, text: str) -> bool:
    text_lower = text.lower()
    term_lower = term.lower().strip()

    if term_lower in text_lower:
        return True

    clean_term = term_lower.replace("…", "").replace("...", "")
    core_words = [w for w in clean_term.split() if len(w) > 1]
    if core_words:
        matches = sum(1 for w in core_words if w in text_lower)
        if matches == len(core_words):
            return True

    stop_words = {"buay", "the", "and", "for", "a", "is", "to", "of", "in"}
    words = [w for w in term_lower.split() if len(w) > 2 and w not in stop_words
             and "." not in w and "…" not in w]
    if words:
        matches = sum(1 for w in words if w in text_lower)
        if matches >= max(1, len(words) * 0.6):
            return True

    if term.isupper() and len(term) <= 6 and term in text:
        return True

    return False


def validate_examples(examples: list[dict], term: str) -> tuple[list[dict], int]:
    valid = []
    dropped = 0
    for ex in examples:
        if term_appears_in_text(term, ex["text"]):
            valid.append(ex)
        else:
            dropped += 1
    return valid, dropped


def parse_llm_output(raw_text: str) -> list[dict]:
    text = raw_text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            if "text" in data and "label" in data:
                data = [data]
            else:
                for v in data.values():
                    if isinstance(v, list):
                        data = v
                        break
                else:
                    return []
        if not isinstance(data, list):
            return []
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if not match:
            return []
        try:
            data = json.loads(match.group())
        except json.JSONDecodeError:
            return []

    valid = []
    for item in data:
        if isinstance(item, dict) and "text" in item and "label" in item:
            label = str(item["label"]).strip().lower()
            if label in ("hate", "offensive", "normal"):
                if "target_group" not in item:
                    item["target_group"] = "none"
                item["label"] = label
                valid.append(item)
    return valid