from __future__ import annotations

import threading
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from tempfile import TemporaryDirectory

from playwright.sync_api import sync_playwright
from pypdf import PdfWriter


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    site_dir = repo_root / "site"
    if not site_dir.is_dir():
        raise SystemExit("Missing site/ directory. Run `mkdocs build` first.")

    routes = [
        "index.html",
        "dag/index.html",
        "serialization/index.html",
        "contributing/index.html",
        "quick_start/index.html",
        "changelog/index.html",
    ]

    previous_cwd = Path.cwd()
    try:
        # Serve static files so browser rendering matches normal navigation.
        import os

        os.chdir(site_dir)
        server = ThreadingHTTPServer(("127.0.0.1", 0), SimpleHTTPRequestHandler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        output_pdf = repo_root / "tidyrun-docs.pdf"
        with TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            page_pdfs: list[Path] = []

            with sync_playwright() as p:
                browser = p.chromium.launch()
                try:
                    for idx, route in enumerate(routes, start=1):
                        page = browser.new_page()
                        url = f"http://127.0.0.1:{port}/{route}"
                        page.goto(url, wait_until="networkidle")
                        pdf_path = tmpdir_path / f"page-{idx:02d}.pdf"
                        page.pdf(
                            path=str(pdf_path),
                            format="A4",
                            print_background=True,
                            margin={
                                "top": "14mm",
                                "right": "12mm",
                                "bottom": "14mm",
                                "left": "12mm",
                            },
                        )
                        page.close()
                        page_pdfs.append(pdf_path)
                finally:
                    browser.close()

            writer = PdfWriter()
            for pdf in page_pdfs:
                writer.append(str(pdf))
            writer.write(str(output_pdf))

        print(f"Wrote {output_pdf}")
        return 0
    finally:
        try:
            server.shutdown()  # type: ignore[name-defined]
        except Exception:
            pass
        try:
            server.server_close()  # type: ignore[name-defined]
        except Exception:
            pass
        os.chdir(previous_cwd)


if __name__ == "__main__":
    raise SystemExit(main())
