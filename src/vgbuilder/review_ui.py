"""Tiny review web UI (standard library only): browse the library, look at the contact sheet, approve / reject.

Local use: binds to 127.0.0.1 by default, has no login, and refuses cross-origin POSTs. Do not expose it to a network
as is. Every value from the database is HTML-escaped; files are only served if they are registered for the asset.
"""
from __future__ import annotations

import html
import json
import mimetypes
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .library import Library

CSS = """
body{font:14px system-ui,sans-serif;background:#16171a;color:#ddd;margin:0}
header{padding:12px 20px;background:#202228;display:flex;gap:16px;align-items:center}
header a{color:#9ab}main{padding:16px 20px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px}
.card{background:#202228;border-radius:8px;padding:10px;border:2px solid #2c2f36}
.card img{width:100%;background:#111;border-radius:4px}
.b{display:inline-block;padding:1px 8px;border-radius:10px;font-size:12px;margin-right:4px;background:#333}
.passed,.approved{background:#245c35}.needs_review,.pending{background:#6b5a1f}.error,.rejected,.failed{background:#6b2525}
.k{color:#889}form{display:flex;gap:6px;margin-top:6px}input[type=text]{flex:1;background:#111;color:#ddd;border:1px solid #444;padding:4px}
button{background:#334;color:#ddd;border:0;padding:4px 10px;border-radius:4px;cursor:pointer}
table{border-collapse:collapse}td,th{padding:3px 10px;text-align:left;border-bottom:1px solid #2c2f36}
pre{background:#111;padding:8px;overflow:auto}
"""


def e(x) -> str:
    return html.escape(str(x), quote=True)


def page(title: str, body: str) -> bytes:
    nav = '<header><b>vgbuilder</b><a href="/">library</a><a href="/?review=pending">to review</a><a href="/queue">queue</a></header>'
    return (f"<!doctype html><meta charset=utf-8><title>{e(title)}</title><style>{CSS}</style>{nav}<main>{body}</main>").encode()


def badge(kind: str, text: str) -> str:
    return f'<span class="b {e(kind)}">{e(text)}</span>'


def summary_line(s: dict) -> str:
    keys = [k for k in s if k not in ("seconds", "iterations")]
    r = lambda v: round(v, 3) if isinstance(v, float) else v  # noqa: E731
    return ", ".join(f"{k}={r(s[k])}" for k in keys[:6])


def make_handler(lib: Library):
    class H(BaseHTTPRequestHandler):
        server_version = "vgbuilder-review"

        def log_message(self, *a):  # quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str = "text/html; charset=utf-8", extra: dict | None = None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urllib.parse.urlparse(self.path)
            q = urllib.parse.parse_qs(u.query)
            parts = [urllib.parse.unquote(p) for p in u.path.strip("/").split("/") if p]
            try:
                if not parts:
                    return self._send(200, self.index(q))
                if parts[0] == "api" and len(parts) == 2 and parts[1] == "assets":
                    rows = [{k: a[k] for k in ("name", "version", "kind", "status", "review", "tags", "summary", "critic")}
                            for a in lib.list(status=q.get("status", [None])[0], review=q.get("review", [None])[0])]
                    return self._send(200, json.dumps(rows).encode(), "application/json")
                if parts[0] == "queue":
                    return self._send(200, self.queue())
                if parts[0] == "asset" and len(parts) in (2, 3):
                    return self._send(200, self.detail(parts[1], int(parts[2]) if len(parts) == 3 else None))
                if parts[0] == "file" and len(parts) == 4:
                    return self.file(parts[1], int(parts[2]), parts[3])
            except (KeyError, ValueError):
                return self._send(404, page("not found", "<p>not found</p>"))
            return self._send(404, page("not found", "<p>not found</p>"))

        def do_POST(self):
            u = urllib.parse.urlparse(self.path)
            parts = [urllib.parse.unquote(p) for p in u.path.strip("/").split("/") if p]
            origin = self.headers.get("Origin")
            if origin and urllib.parse.urlparse(origin).netloc != self.headers.get("Host"):
                return self._send(403, page("forbidden", "<p>cross-origin request refused</p>"))
            if len(parts) == 3 and parts[0] == "review":
                n = int(self.headers.get("Content-Length") or 0)
                if n > 20000:
                    return self._send(413, page("too large", "<p>too large</p>"))
                form = urllib.parse.parse_qs(self.rfile.read(n).decode("utf-8", "replace"))
                try:
                    lib.review(parts[1], form.get("decision", [""])[0], form.get("note", [""])[0][:500], int(parts[2]))
                except (KeyError, ValueError) as ex:
                    return self._send(400, page("error", f"<p>{e(ex)}</p>"))
                return self._send(303, b"", extra={"Location": f"/asset/{urllib.parse.quote(parts[1])}/{parts[2]}"})
            return self._send(404, page("not found", "<p>not found</p>"))

        # ---- pages
        def index(self, q) -> bytes:
            rows = lib.list(status=q.get("status", [None])[0], review=q.get("review", [None])[0], tag=q.get("tag", [None])[0])
            if not rows:
                return page("library", "<p class=k>nothing here yet</p>")
            cards = []
            for a in rows:
                img = (f'<a href="/asset/{urllib.parse.quote(a["name"])}/{a["version"]}"><img loading=lazy src="/file/'
                       f'{urllib.parse.quote(a["name"])}/{a["version"]}/{urllib.parse.quote(a["thumb"])}"></a>') if a["thumb"] else ""
                crit = a["critic"]
                cs = (badge("error", "critic error") if crit and crit.get("error") else
                      badge("needs_review" if crit.get("below_min") else "passed", f"critic {crit['score']:.2f}") if crit else "")
                cards.append(f'<div class=card>{img}<div><b>{e(a["name"])}</b> v{a["version"]} '
                             f'{badge(a["status"], a["status"])}{badge(a["review"], a["review"])}{cs}</div>'
                             f'<div class=k>{e(a["kind"])} · {e(", ".join(t for t in a["tags"][:5]))}</div>'
                             f'<div class=k>{e(summary_line(a["summary"]))}</div></div>')
            return page("library", f'<div class=grid>{"".join(cards)}</div>')

        def detail(self, name: str, version: int | None) -> bytes:
            a = lib.get(name, version)
            if a is None:
                raise KeyError(name)
            nm, v = urllib.parse.quote(a["name"]), a["version"]
            img = f'<img style="max-width:100%" src="/file/{nm}/{v}/{urllib.parse.quote(a["thumb"])}">' if a["thumb"] else ""
            fails = "".join(f"<li>{e(f['code'])}: {e(f.get('detail', ''))}</li>" for f in a["failures"]) or "<li class=k>none</li>"
            files = "".join(f'<tr><td><a href="/file/{nm}/{v}/{urllib.parse.quote(f["file"])}">{e(f["file"])}</a></td>'
                            f'<td>{f["bytes"] or ""}</td><td class=k>{e((f["sha256"] or "")[:12])}</td></tr>' for f in a["files"])
            ev = "".join(f"<tr><td class=k>{e(x['ts'])}</td><td>{e(x['kind'])}</td><td>{e(x['detail'] or '')}</td></tr>" for x in a["events"])
            crit = a["critic"]
            critic = ("<p class=k>no critic run</p>" if not crit else
                      f"<p>{e(crit.get('error'))}</p>" if crit.get("error") else
                      f"<p>score {crit['score']:.2f} (min {crit.get('min_score')})<br>{e(crit.get('notes', ''))}</p>"
                      f"<ul>{''.join('<li>' + e(i) + '</li>' for i in crit.get('issues', []))}</ul>")
            vers = "".join(f'<a href="/asset/{nm}/{x["version"]}">v{x["version"]}</a> ' for x in lib.list(name=name, latest_only=False))
            form = "".join(
                f'<form method=post action="/review/{nm}/{v}"><input type=hidden name=decision value="{d}">'
                f'<input type=text name=note placeholder="note (optional)" value="">'
                f'<button>{label}</button></form>' for d, label in (("approved", "Approve"), ("rejected", "Reject"), ("pending", "Reset")))
            body = (f"<h2>{e(a['name'])} v{v} {badge(a['status'], a['status'])}{badge(a['review'], a['review'])}</h2>"
                    f"<p class=k>versions: {vers}</p>{img}<h3>Review</h3><p class=k>{e(a['review_note'])}</p>{form}"
                    f"<h3>QA failures left</h3><ul>{fails}</ul><h3>Critic</h3>{critic}"
                    f"<h3>Summary</h3><pre>{e(json.dumps(a['summary'], indent=2))}</pre>"
                    f"<h3>Files</h3><table>{files}</table><h3>History</h3><table>{ev}</table>"
                    f"<h3>Spec</h3><pre>{e(json.dumps(a['spec'], indent=2, default=str))}</pre>")
            return page(a["name"], body)

        def queue(self) -> bytes:
            rows = "".join(f"<tr><td>{r['id']}</td><td>{e(r['name'])}</td><td>{r['priority']}</td><td>{e(r['resource'] or '')}</td><td>{badge(r['state'], r['state'])}</td>"
                           f"<td>{r['attempts']}</td><td class=k>{e(r['error'] or '')}</td></tr>" for r in lib.queue_rows())
            return page("queue", f"<table><tr><th>#<th>job<th>prio<th>resource<th>state<th>tries<th>error</tr>{rows}</table>")

        def file(self, name: str, version: int, fname: str):
            a = lib.get(name, version)
            if a is None or fname not in {f["file"] for f in a["files"]}:
                return self._send(404, page("not found", "<p>not found</p>"))
            p = lib.resolve(a["final_dir"]) / fname
            if not p.is_file():
                return self._send(404, page("not found", "<p>file missing on disk</p>"))
            ctype = mimetypes.guess_type(fname)[0] or "application/octet-stream"
            return self._send(200, p.read_bytes(), ctype)

    return H


def serve(lib: Library, host: str = "127.0.0.1", port: int = 8765, log=print) -> ThreadingHTTPServer:
    srv = ThreadingHTTPServer((host, port), make_handler(lib))
    log(f"review UI on http://{host}:{srv.server_address[1]}/  (Ctrl-C to stop)")
    return srv
