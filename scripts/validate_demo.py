#!/usr/bin/env python3
"""
Playwright validation for the Streamlit demo.

Default behavior:
  - starts demo/app.py on localhost
  - disables Streamlit file watching to avoid transformers optional-import spam
  - writes server logs and screenshots under tmp/playwright-validation/
  - exercises Chat, Arena, Challenge, Leaderboard, and Samples
  - stops the server before exiting

Use --base-url to validate an already-running app instead.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from playwright.async_api import Page, async_playwright, expect


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "tmp/playwright-validation"
PAGES = ["Chat", "Arena", "Challenge", "Leaderboard", "Samples"]
HATE_TEXT = "knn those cina babi think they own the place"
BENIGN_TEXT = "wah this kopi damn shiok sia"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the Streamlit demo UI.")
    parser.add_argument("--backend", default=os.environ.get("GUARD_MODEL", "sg"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8765")))
    parser.add_argument("--base-url", default=os.environ.get("DEMO_BASE_URL"))
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--keep-server", action="store_true")
    parser.add_argument("--skip-ui", action="store_true")
    parser.add_argument("--timeout", type=float, default=90.0)
    return parser.parse_args()


def load_env() -> None:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    load_dotenv(ROOT / "demo/.env", override=True)


def wait_for_server(url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urlopen(url, timeout=2) as response:
                if response.status < 500:
                    return
        except (OSError, URLError) as exc:
            last_error = exc
        time.sleep(0.5)
    raise RuntimeError(f"Timed out waiting for {url}: {last_error}")


def start_server(args: argparse.Namespace, run_dir: Path) -> tuple[subprocess.Popen, Path]:
    log_path = run_dir / "streamlit.log"
    log_file = log_path.open("w")
    env = os.environ.copy()
    env["GUARD_MODEL"] = args.backend

    cmd = [
        str(ROOT / ".venv/bin/streamlit"),
        "run",
        "demo/app.py",
        "--server.port",
        str(args.port),
        "--server.fileWatcherType",
        "none",
        "--server.headless",
        "true",
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    proc._codex_log_file = log_file  # type: ignore[attr-defined]
    return proc, log_path


def stop_server(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    log_file = getattr(proc, "_codex_log_file", None)
    if log_file:
        log_file.close()


async def click_nav(page: Page, label: str) -> None:
    link = page.locator(f'[data-testid="stSidebarNav"] a:has-text("{label}")').first
    if await link.count() == 0:
        link = page.locator(f'a:has-text("{label}")').first
    await link.click()
    await settle(page)


async def settle(page: Page) -> None:
    await page.wait_for_timeout(800)
    try:
        await page.locator('[data-testid="stStatusWidget"]').wait_for(
            state="hidden", timeout=15_000
        )
    except Exception:
        pass
    await page.wait_for_timeout(400)


async def assert_clean_page(page: Page, label: str, run_dir: Path) -> None:
    n_exc = await page.locator('[data-testid="stException"]').count()
    n_404 = await page.locator("text=Page not found").count()
    await page.screenshot(path=run_dir / f"{label.lower()}-page.png", full_page=True)
    if n_exc or n_404:
        raise AssertionError(f"{label} failed: exceptions={n_exc}, page_not_found={n_404}")


async def test_page_navigation(page: Page, run_dir: Path) -> None:
    for label in PAGES:
        await click_nav(page, label)
        await assert_clean_page(page, label, run_dir)


async def test_chat(page: Page, run_dir: Path) -> None:
    await click_nav(page, "Chat")
    clear = page.get_by_role("button", name="Clear all")
    if await clear.count():
        await clear.click()
        await settle(page)

    chat_input = page.locator('textarea[placeholder="Type a message…"]').first
    await chat_input.fill(BENIGN_TEXT)
    await chat_input.press("Enter")
    await page.locator("text=Turn 1").wait_for(timeout=90_000)
    await assert_clean_page(page, "chat-benign", run_dir)

    await chat_input.fill(HATE_TEXT)
    await chat_input.press("Enter")
    await page.locator("text=Guardrail — input blocked").first.wait_for(timeout=60_000)
    await page.locator("text=/score: 0\\.\\d+/").first.wait_for(timeout=10_000)
    await page.screenshot(path=run_dir / "chat-blocked.png", full_page=True)
    await assert_clean_page(page, "chat-blocked-clean", run_dir)


async def test_arena(page: Page, run_dir: Path) -> None:
    await click_nav(page, "Arena")
    await page.locator('textarea[aria-label="Input"]').fill(BENIGN_TEXT)
    await page.get_by_role("button", name="Analyze All").click()
    await page.locator("text=SG SingBERT").first.wait_for(timeout=60_000)
    await page.locator("text=Allow").first.wait_for(timeout=60_000)
    await assert_clean_page(page, "arena-results", run_dir)


async def test_backend_arena_presence(page: Page, run_dir: Path, backend: str) -> None:
    await click_nav(page, "Arena")
    await page.locator('textarea[aria-label="Input"]').fill(BENIGN_TEXT)
    await page.get_by_role("button", name="Analyze All").click()

    sys.path.insert(0, str(ROOT / "demo"))
    from guard import BACKEND_LABELS

    await page.locator(f"text={BACKEND_LABELS[backend]}").first.wait_for(timeout=120_000)
    await assert_clean_page(page, f"arena-{backend}", run_dir)


async def test_challenge(page: Page, run_dir: Path) -> None:
    await click_nav(page, "Challenge")
    for round_number in range(1, 6):
        await page.locator(f"text=Round {round_number} / 5").wait_for(timeout=30_000)
        await page.get_by_role("button", name="Queue").click()
        await page.locator("text=How each guard ruled").wait_for(timeout=90_000)
        await page.locator("text=SG SingBERT").first.wait_for(timeout=30_000)
        await page.screenshot(
            path=run_dir / f"challenge-round-{round_number}.png",
            full_page=True,
        )
        if round_number < 5:
            await page.get_by_role("button", name="Next round →").click()
            await settle(page)
    await page.get_by_role("button", name="Next round →").click()
    await page.locator("text=Game over").wait_for(timeout=30_000)
    await assert_clean_page(page, "challenge-game-over", run_dir)


async def test_leaderboard_samples(page: Page, run_dir: Path) -> None:
    await click_nav(page, "Leaderboard")
    await page.get_by_role("heading", name="Leaderboard").wait_for(timeout=30_000)
    await assert_clean_page(page, "leaderboard", run_dir)

    await click_nav(page, "Samples")
    await page.get_by_role("heading", name="Dataset — Sample Browser").wait_for(
        timeout=30_000
    )
    await assert_clean_page(page, "samples", run_dir)


async def run_validation(base_url: str, headed: bool, run_dir: Path) -> None:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=not headed)
        context = await browser.new_context(viewport={"width": 1400, "height": 950})
        page = await context.new_page()
        page.on("pageerror", lambda exc: print(f"pageerror: {exc}", file=sys.stderr))

        await page.goto(base_url, wait_until="networkidle", timeout=60_000)
        await settle(page)

        await test_page_navigation(page, run_dir)
        await test_chat(page, run_dir)
        await test_arena(page, run_dir)
        await test_challenge(page, run_dir)
        await test_leaderboard_samples(page, run_dir)

        await browser.close()


async def run_backend_ui_validation(
    base_url: str,
    headed: bool,
    run_dir: Path,
    backend: str,
) -> None:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=not headed)
        context = await browser.new_context(viewport={"width": 1400, "height": 950})
        page = await context.new_page()
        page.on("pageerror", lambda exc: print(f"pageerror: {exc}", file=sys.stderr))

        await page.goto(base_url, wait_until="networkidle", timeout=60_000)
        await settle(page)
        await test_page_navigation(page, run_dir)
        await test_backend_arena_presence(page, run_dir, backend)

        await browser.close()


def run_direct_backend_check(backend: str, run_dir: Path) -> None:
    sys.path.insert(0, str(ROOT / "demo"))
    from guard import backend_status, run_guard

    lines = [f"backend={backend}"]
    ok, reason = backend_status(backend)
    lines.append(f"status={ok} reason={reason}")

    if not ok:
        lines.append("result=unavailable")
        (run_dir / f"backend-{backend}.txt").write_text("\n".join(lines) + "\n")
        if backend == "llmguard":
            print(f"SKIP: {backend} unavailable: {reason}")
            return
        raise AssertionError(f"{backend} unavailable: {reason}")

    hate = run_guard(HATE_TEXT, backend=backend)
    benign = run_guard(BENIGN_TEXT, backend=backend)
    lines.append(f"hate={hate}")
    lines.append(f"benign={benign}")
    (run_dir / f"backend-{backend}.txt").write_text("\n".join(lines) + "\n")

    if hate.get("error"):
        raise AssertionError(f"{backend} hate check errored: {hate['error']}")
    if benign.get("error"):
        raise AssertionError(f"{backend} benign check errored: {benign['error']}")
    if not isinstance(hate.get("score"), float) or not isinstance(benign.get("score"), float):
        raise AssertionError(f"{backend} did not return numeric scores")
    if backend in {"lionguard", "lionguard-2.1"} and not hate.get("fine_grained"):
        raise AssertionError(f"{backend} did not return fine_grained scores")
    if backend == "sea-guard" and not hate.get("verdict_label"):
        raise AssertionError("sea-guard did not return a verdict_label")
    if backend != "llmguard" and hate.get("passed") is not False:
        raise AssertionError(f"{backend} did not block the hate sample")
    if backend != "llmguard" and benign.get("passed") is not True:
        raise AssertionError(f"{backend} did not allow the benign sample")


async def async_main() -> int:
    args = parse_args()
    load_env()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = OUT_DIR / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir()

    base_url = args.base_url or f"http://localhost:{args.port}"
    proc: subprocess.Popen | None = None
    log_path: Path | None = None

    try:
        run_direct_backend_check(args.backend, run_dir)
        if not args.skip_ui:
            if not args.base_url:
                proc, log_path = start_server(args, run_dir)
                wait_for_server(base_url, args.timeout)

            if args.backend == "sg":
                await run_validation(base_url, args.headed, run_dir)
            else:
                await run_backend_ui_validation(base_url, args.headed, run_dir, args.backend)
    finally:
        if not args.keep_server:
            stop_server(proc)

    latest = OUT_DIR / "latest"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(run_dir.name)

    print(f"OK: validation passed")
    print(f"Artifacts: {run_dir}")
    if log_path:
        print(f"Streamlit log: {log_path}")
    return 0


def main() -> int:
    try:
        return asyncio.run(async_main())
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
