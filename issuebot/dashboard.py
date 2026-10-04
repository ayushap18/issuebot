"""Static dashboard: one HTML page from results/*.json and eval/status.json, worst repo first."""
import argparse
import json
from datetime import datetime, timezone
from html import escape
from pathlib import Path

from issuebot.backtest import worst

CSS = """:root{color-scheme:light dark;--bg:#fff;--fg:#1f2328;--mute:#59636e;--line:#d1d9e0;--bad:#cf222e;--ok:#1a7f37}
@media (prefers-color-scheme:dark){:root{--bg:#0d1117;--fg:#e6edf3;--mute:#9198a1;--line:#3d444d;--bad:#ff7b72;--ok:#3fb950}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0 auto;max-width:1100px;padding:16px}
a{color:inherit}p,.mute{color:var(--mute)}.wrap{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;white-space:nowrap}
td.t{white-space:normal;min-width:12em}.shadow{color:var(--bad)}.comment{color:var(--ok)}"""


def _f(x, fmt="{:.0%}") -> str:
    return "-" if x is None else fmt.format(x)


def rows(results: Path, status: dict) -> list[dict]:
    live = lambda r: {"kept": (status.get(r) or {}).get("kept_rate"), "live_n": (status.get(r) or {}).get("n"),
                      "status": (status.get(r) or {}).get("status")}
    out, seen = [], set()
    for p in sorted(results.glob("*.json")):
        try:
            res = json.loads(p.read_text())
            m = res["metrics"]
        except (ValueError, KeyError, TypeError):
            continue  # not a results file
        repo = res.get("repo") or "-"  # run_eval results have no repo field
        seen.add(repo)
        w = next(iter(worst(res.get("cases") or [], 1)), None)
        out.append({"repo": repo, "run": res.get("name", p.stem), "n": m.get("n"), "acc": m.get("label_accuracy"),
                    "dup": m.get("dup_recall"), "judge": m.get("judge_mean"), "cpi": m.get("cost_per_issue"),
                    "worst": w, **live(repo)})
    out += [{"repo": r, "run": None, **live(r)} for r in status if r not in seen]
    # worst first: offline label accuracy, else live kept rate; rows with neither go last
    key = lambda r: r["acc"] if r.get("acc") is not None else r["kept"]
    return sorted(out, key=lambda r: (key(r) is None, key(r) or 0, r["repo"]))


def render(rs: list[dict], now: str) -> str:
    e = lambda x: escape(str(x), quote=True)
    link = lambda c: (f'<a href="{e(c["url"])}">#{e(c["number"])} {e(c.get("title", ""))}</a>'
                      if str(c.get("url", "")).startswith("https://github.com/") else f"#{e(c['number'])} {e(c.get('title', ''))}")
    if not rs:
        body = "<p><strong>No results yet.</strong> Run <code>python -m issuebot.backtest owner/repo</code> and commit " \
               "<code>results/*.json</code>; live status appears after the weekly feedback run.</p>"
    else:
        trs = "".join(
            f"<tr><td>{e(r['repo'])}</td><td>{e(r['run'] or '-')}</td><td>{e(r.get('n') or '-')}</td>"
            f"<td>{_f(r.get('acc'))}</td><td>{_f(r.get('dup'))}</td><td>{_f(r.get('judge'), '{:.2f}')}</td>"
            f"<td>{_f(r.get('cpi'), '${:.4f}')}</td><td>{_f(r['kept'])}{f' ({e(r['live_n'])})' if r['live_n'] else ''}</td>"
            f"<td class=\"{e(r['status'] or '')}\">{e(r['status'] or '-')}</td>"
            f"<td class=\"t\">{link(r['worst']) + ' (' + e(r['worst']['score']) + '/5)' if r.get('worst') else '-'}</td></tr>"
            for r in rs)
        body = ("<div class=\"wrap\"><table><thead><tr><th>repo</th><th>run</th><th>n</th><th>label acc</th><th>dup recall</th>"
                "<th>judge</th><th>$/issue</th><th>live kept</th><th>status</th><th>worst reply</th></tr></thead>"
                f"<tbody>{trs}</tbody></table></div>")
    return (f"<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            f"<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>issuebot dashboard</title>"
            f"<style>{CSS}</style></head><body><h1>issuebot dashboard</h1>"
            f"<p>Offline backtest/eval numbers per run and live label-kept status per repo, worst first. "
            f"Status: shadow &lt; label &lt; comment (comment needs 90% kept over 100 issues; under 75% demotes to shadow).</p>"
            f"{body}<p class=\"mute\">Built {e(now)} from results/*.json and eval/status.json.</p></body></html>\n")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="issuebot.dashboard")
    ap.add_argument("--results", default="results")
    ap.add_argument("--status", default="eval/status.json")
    ap.add_argument("--out", default="site")
    a = ap.parse_args(argv)
    sp = Path(a.status)
    status = json.loads(sp.read_text()) if sp.exists() else {}
    out = Path(a.out) / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(rows(Path(a.results), status), datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
