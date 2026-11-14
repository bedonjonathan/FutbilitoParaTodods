"""Optional local preview (the real site is built by GitHub Actions).

    python run.py                 # build data for all leagues, then serve at http://localhost:8000
    python run.py --skip-build    # just serve what's already in site/
    python run.py --leagues E0    # quicker build: Premier League only
"""
import argparse
import functools
import http.server
import threading
import webbrowser
from pathlib import Path

from backend import export

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--leagues", nargs="*")
    a = ap.parse_args()
    site = Path(__file__).resolve().parent / "site"
    if not a.skip_build:
        export.main((["--leagues", *a.leagues] if a.leagues else []) + ["--out", str(site)])
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(site))
    threading.Timer(1.0, lambda: webbrowser.open(f"http://localhost:{a.port}")).start()
    print(f"Serving http://localhost:{a.port}  (Ctrl+C to stop)")
    http.server.ThreadingHTTPServer(("127.0.0.1", a.port), handler).serve_forever()
