"""Isolated Chromium acceptance test; not automation of a user's browser."""

import subprocess
import json
import sys
import time
from pathlib import Path
from urllib.request import urlopen

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
with (ROOT / "reports/browser-server.log").open("w") as log:
    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "src.app.api:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
        ],
        cwd=ROOT,
        stdout=log,
        stderr=log,
    )
    try:
        deadline = time.monotonic() + 180
        while True:
            if server.poll() is not None:
                raise RuntimeError("Browser test server exited")
            try:
                with urlopen("http://127.0.0.1:8000/health", timeout=2):
                    break
            except OSError:
                if time.monotonic() > deadline:
                    raise TimeoutError("Server startup")
                time.sleep(1)
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1280, "height": 960})
            page.goto("http://127.0.0.1:8000")
            page.get_by_role("button", name="Build sample index").click()
            page.get_by_text("Indexed 3 documents", exact=False).wait_for()
            page.get_by_role("button", name="Find evidence").click()
            page.get_by_text("evidence found", exact=True).wait_for()
            assert page.locator("details").count() > 0
            page.locator("summary").first.click()
            assert "one-hour" in page.locator("#result").inner_text()
            assert "File SHA-256:" in page.locator("#result").inner_text()
            assert "lines" in page.locator("summary").first.inner_text()
            with page.expect_download(timeout=10000) as download:
                page.get_by_role("button", name="Save evidence record").click()
            exported = json.loads(Path(download.value.path()).read_text())
            assert exported["citations"][0]["source_sha256"]
            assert exported["corpus_sha256"]
            page.screenshot(path=str(ROOT / "reports/desk-desktop.png"), full_page=True)
            page.set_viewport_size({"width": 390, "height": 844})
            assert page.evaluate(
                "document.documentElement.scrollWidth <= window.innerWidth"
            )
            page.screenshot(path=str(ROOT / "reports/desk-mobile.png"), full_page=True)
            page.get_by_label("QUESTION", exact=True).fill(
                "What is the capital of Argentina?"
            )
            page.get_by_role("button", name="Find evidence").click()
            page.get_by_text("insufficient evidence", exact=True).wait_for()
            assert page.locator("details").count() == 0
            page.set_viewport_size({"width":1280,"height":960})
            page.get_by_label('Collection',exact=True).select_option('curl')
            page.get_by_role('button',name='Build reference index',exact=True).click()
            page.get_by_text('Indexed 8 documents',exact=False).wait_for()
            page.get_by_label('QUESTION',exact=True).fill('What certificate format does --cacert require?')
            page.get_by_role('button',name='Find evidence').click()
            page.get_by_text('evidence found',exact=True).wait_for(timeout=60000)
            assert 'PEM' in page.locator('#result').inner_text()
            page.locator('summary').first.click()
            page.screenshot(path=str(ROOT/'reports/desk-reference.png'),full_page=True)
            page.get_by_label("QUESTION", exact=True).fill(
                "What is the Business plan GPU quota?"
            )
            page.get_by_role("button", name="Find evidence").click()
            page.get_by_text("insufficient evidence", exact=True).wait_for()
            assert page.locator("details").count() == 0
            browser.close()
            print(
                "Chromium: indexing, answer, expandable sources, mobile overflow and abstention passed"
            )
    finally:
        server.terminate()
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=5)
