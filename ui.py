import base64
import html
import json
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import streamlit as st

from config import (
    OPENAI_API_KEY,
    ANTHROPIC_API_KEY,
    GOOGLE_API_KEY,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    DEEPSEEK_API_KEY,
    OLLAMA_BASE_URL,
    LLAMA_CPP_BASE_URL,
)
from health import check_llm_health, check_search_engines, check_tor_proxy
from llm import get_llm, refine_query, filter_results, generate_summary, PRESET_PROMPTS
from llm_utils import (
    BufferedStreamingHandler,
    get_model_choices,
    _llm_config_map,
    is_deepseek_endpoint,
    is_openrouter_endpoint,
)
from scrape import scrape_multiple
from search import get_search_results, SEARCH_ENGINES

# --- Investigation persistence ---

INVESTIGATIONS_DIR = Path("investigations")

PRESET_OPTIONS = {
    "Dark Web Threat Intel": "threat_intel",
    "Ransomware & Malware": "ransomware_malware",
    "Personal & Identity": "personal_identity",
    "Corporate Espionage": "corporate_espionage",
}
PRESET_PLACEHOLDERS = {
    "threat_intel": "e.g. Pay extra attention to cryptocurrency wallet addresses and exchange names.",
    "ransomware_malware": "e.g. Highlight any references to double-extortion tactics or known ransomware-as-a-service affiliates.",
    "personal_identity": "e.g. Flag any passport or government ID numbers and note which country they appear to be from.",
    "corporate_espionage": "e.g. Prioritize any mentions of source code repositories, API keys, or internal Slack/email dumps.",
}
STAGE_NAMES = ["LOAD · LLM", "REFINE", "SEARCH", "FILTER", "SCRAPE", "SUMMARY"]
CLOUD_PROVIDERS = [
    ("OpenAI", OPENAI_API_KEY, "OPENAI_API_KEY"),
    ("Anthropic", ANTHROPIC_API_KEY, "ANTHROPIC_API_KEY"),
    ("Google", GOOGLE_API_KEY, "GOOGLE_API_KEY"),
    ("OpenRouter", OPENROUTER_API_KEY, "OPENROUTER_API_KEY"),
    ("DeepSeek", DEEPSEEK_API_KEY, "DEEPSEEK_API_KEY"),
]


def save_investigation(query: str, refined_query: str, model: str, preset_label: str, sources: list, summary: str) -> str:
    """Save a completed investigation to disk. Returns the filename."""
    INVESTIGATIONS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"investigation_{timestamp}.json"
    data = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "refined_query": refined_query,
        "model": model,
        "preset": preset_label,
        "sources": sources,
        "summary": summary,
    }
    (INVESTIGATIONS_DIR / fname).write_text(json.dumps(data, indent=2))
    return fname


def load_investigations() -> list:
    """Return list of saved investigations sorted newest-first."""
    if not INVESTIGATIONS_DIR.exists():
        return []
    files = sorted(INVESTIGATIONS_DIR.glob("investigation_*.json"), reverse=True)
    investigations = []
    for f in files:
        try:
            data = json.loads(f.read_text())
            data["_filename"] = f.name
            investigations.append(data)
        except Exception:
            continue
    return investigations


@st.cache_data(show_spinner=False, max_entries=12)
def report_files(inv: dict) -> dict:
    """JSON + PDF report bytes for an investigation record (cached per record)."""
    from export_report import report_json_bytes, report_pdf_bytes

    source = inv.get("source_file") or inv.get("_filename") or ""
    return {
        "json": report_json_bytes(inv, source),
        "pdf": report_pdf_bytes(inv, source),
    }


def _dl_link(name: str, data: bytes, mime: str, label: str) -> str:
    b64 = base64.b64encode(data).decode()
    return f'<a class="dl" href="data:{mime};base64,{b64}" download="{name}">{label}</a>'


def report_downloads_html(inv: dict, with_markdown: bool = False) -> str:
    """Report panel download links: Report PDF / Report JSON (+ summary Markdown)."""
    from export_report import slugify

    files = report_files(inv)
    date = (inv.get("timestamp") or "")[:10] or datetime.now().strftime("%Y-%m-%d")
    stem = f"osint-{slugify(inv.get('query') or 'investigation')}-{date}"
    links = [
        _dl_link(f"{stem}.pdf", files["pdf"], "application/pdf", "Report · PDF"),
        _dl_link(f"{stem}.json", files["json"], "application/json", "Report · JSON"),
    ]
    if with_markdown:
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        links.append(_dl_link(f"summary_{now}.md", (inv.get("summary") or "").encode(),
                              "text/markdown", "Summary · Markdown"))
    return " ".join(links)


# Cache expensive backend calls
@st.cache_data(ttl=200, show_spinner=False)
def cached_search_results(refined_query: str, threads: int):
    return get_search_results(refined_query.replace(" ", "+"), max_workers=threads)


@st.cache_data(ttl=200, show_spinner=False)
def cached_scrape_multiple(filtered: list, threads: int):
    return scrape_multiple(filtered, max_workers=threads)


# --- Rendering helpers ---

def esc(value) -> str:
    return html.escape(str(value), quote=True)


def mono(value) -> str:
    return f'<b class="mono">{esc(value)}</b>'


def fmt_ms(ms) -> str:
    if ms is None:
        return "—"
    ms = int(ms)
    return f"{ms} MS" if ms < 1000 else f"{ms / 1000:.1f} S"


def env_is_set(value) -> bool:
    return bool(value and str(value).strip() and "your_" not in str(value))


def provider_of(model: str) -> str:
    cfg = _llm_config_map.get(str(model).strip().lower())
    if not cfg:
        return ""
    class_name = getattr(cfg["class"], "__name__", "")
    ctor = cfg.get("constructor_params", {}) or {}
    if "ChatAnthropic" in class_name:
        return "Anthropic"
    if "ChatGoogleGenerativeAI" in class_name:
        return "Google"
    if "ChatOpenAI" in class_name:
        base_url = ctor.get("base_url")
        if is_openrouter_endpoint(base_url, model):
            return "OpenRouter"
        if is_deepseek_endpoint(base_url, model):
            return "DeepSeek"
        return "OpenAI"
    return ""


def pill(text: str, kind: str) -> str:
    return f'<span class="pill p-{kind}">{esc(text)}</span>'


def ck_row(name: str, meta: str, state_pill: str) -> str:
    return (
        f'<div class="ck"><div class="ckl"><div class="name">{esc(name)}</div>'
        f'<div class="meta">{meta}</div></div>{state_pill}</div>'
    )


def panel_head(title: str, meta: str = "") -> str:
    """Header strip only — for panels whose chrome comes from a keyed container."""
    return f'<div class="panelhdr"><span class="t">{esc(title)}</span><span class="m">{meta}</span></div>'


def panel_html(title: str, meta: str, body: str) -> str:
    """Full panel — for read-only panels rendered through a single slot."""
    return f'<div class="pnl">{panel_head(title, meta)}{body}</div>'


def strip_html(phase, mode="idle", total_ms=None, archive_ts=None) -> str:
    cells = []
    for idx, name in enumerate(STAGE_NAMES):
        state = phase[idx]["state"] if idx < len(phase) else "pending"
        ms = phase[idx].get("ms") if idx < len(phase) else None
        dur = ""
        if state == "done" and ms is not None:
            dur = f'<span class="sd mono">{esc(fmt_ms(ms))}</span>'
        elif state == "live":
            dur = '<span class="sd mono">RUNNING</span>'
        elif state == "halt":
            dur = '<span class="sd mono">HALT</span>'
        cells.append(
            f'<div class="stage {state}"><span class="sn mono">{idx + 1:02d}</span>'
            f'<span class="st">{esc(name)}</span>{dur}</div>'
        )

    done_count = sum(1 for p in phase if p["state"] == "done")
    live_idx = next((i for i, p in enumerate(phase) if p["state"] == "live"), None)

    if mode == "running" and live_idx is not None:
        k = f"PHASE {live_idx + 1:02d} / 06"
        sub = ""
        pct = int(done_count / 6 * 100)
    elif mode == "complete":
        k = "MISSION COMPLETE"
        sub = f'<span class="sd mono">{esc(fmt_ms(total_ms))} TOTAL</span>'
        pct = 100
    elif mode == "halted":
        k = "MISSION HALTED"
        sub = ""
        pct = int(done_count / 6 * 100)
    elif mode == "archive":
        k = f"ARCHIVE · {esc(archive_ts or '—')}"
        sub = '<span class="sd mono">RESTORED RECORD</span>'
        pct = 0
    else:
        k = "READY · PRE-FLIGHT"
        sub = '<span class="sd mono">NOTHING LAUNCHED</span>'
        pct = 0

    return (
        '<div class="strip">' + "".join(cells) +
        f'<div class="sr2"><span class="k mono">{k}</span>{sub}'
        f'<div class="bar"><i style="width:{pct}%"></i></div></div></div>'
    )


# Brand mark embedded as a data URI so the header needs no static-file serving
_ARCHER_MARK_PATH = Path(__file__).parent / ".github" / "assets" / "robin_archer.png"
try:
    _ARCHER_MARK_URI = "data:image/png;base64," + base64.b64encode(_ARCHER_MARK_PATH.read_bytes()).decode("ascii")
except OSError:
    _ARCHER_MARK_URI = ""


def header_html(tor_state: str, elapsed=None, state_label="READY") -> str:
    tor_chip = {
        "checking": '<span class="torchip"><span class="dot"></span>TOR · CHECKING…</span>',
        "on": '<span class="torchip on"><span class="dot on"></span>TOR CONNECTED</span>',
        "off": '<span class="torchip off"><span class="dot off"></span>TOR UNREACHABLE</span>',
    }.get(tor_state, '<span class="torchip"><span class="dot"></span>TOR · UNKNOWN</span>')
    elapsed_txt = "—" if elapsed is None else f"{int(elapsed // 60)}m {int(elapsed % 60):02d}s"
    mark = f'<img class="mark" src="{_ARCHER_MARK_URI}" alt="">' if _ARCHER_MARK_URI else ""
    return (
        f'<header class="hdr"><div class="brand">{mark}<h1>ROBIN</h1>'
        '<span class="sub">AI-Powered Dark Web OSINT Tool</span></div>'
        f'<div class="hmeta">{tor_chip}'
        f'<div class="hm"><span class="lbl">Elapsed</span><span class="val mono">{elapsed_txt}</span></div>'
        f'<div class="hm"><span class="lbl">State</span><span class="val mono">{esc(state_label)}</span></div>'
        '</div></header>'
    )


def board_html(pf, model, model_options, engines_state="unknown") -> str:
    rows = []
    if pf is None or pf.get("tor") is None:
        rows.append(ck_row("Tor network", "Testing SOCKS5 127.0.0.1:9050…", pill("CHECKING", "neutral")))
    else:
        tor = pf["tor"]
        if tor["status"] == "up":
            rows.append(ck_row("Tor network", "Circuit established · SOCKS5 127.0.0.1:9050",
                               pill(f"GO · {tor['latency_ms']} MS", "go")))
        else:
            rows.append(ck_row("Tor network", f"SOCKS5 127.0.0.1:9050 unreachable · {esc(str(tor['error'])[:80])}",
                               pill("NO-GO", "nogo")))

    llm = (pf or {}).get("llm")
    selected_provider = ""
    if llm is None:
        rows.append(ck_row("LLM provider", f"Testing {esc(model)}…", pill("CHECKING", "neutral")))
    else:
        selected_provider = llm.get("provider") or provider_of(model)
        if llm["status"] == "up":
            meta = f"{esc(model)} selected · responding in {llm['latency_ms']} ms"
            rows.append(ck_row(f"{selected_provider} provider", meta, pill(f"GO · {llm['latency_ms']} MS", "go")))
        else:
            meta = f"{esc(model)} selected · {esc(str(llm.get('error') or 'no response')[:90])}"
            rows.append(ck_row(f"{selected_provider or 'LLM'} provider", meta, pill("NO-GO", "nogo")))

    sel_norm = selected_provider.split(" ")[0] if selected_provider else ""
    others = [n for n, v, _ in CLOUD_PROVIDERS if env_is_set(v) and n != sel_norm]
    missing = [n for n, v, _ in CLOUD_PROVIDERS if not env_is_set(v)]
    if others:
        rows.append(ck_row(" · ".join(others), "API key set · model not selected", pill("KEY SET", "neutral")))
    if missing:
        rows.append(ck_row(" · ".join(missing), "not configured · key not set", pill("NO-GO", "nogo")))

    local_configured = env_is_set(OLLAMA_BASE_URL) or env_is_set(LLAMA_CPP_BASE_URL)
    local_models = [m for m in model_options if str(m).lower() not in _llm_config_map]
    if local_configured and local_models:
        rows.append(ck_row("Local models · Ollama / llama.cpp",
                           f"{len(local_models)} models detected · added to the model list", pill("READY", "go")))
    elif local_configured:
        rows.append(ck_row("Local models · Ollama / llama.cpp",
                           "server configured · no models responding", pill("STANDBY", "amber")))
    else:
        rows.append(ck_row("Local models · Ollama / llama.cpp",
                           "optional path · not configured", pill("STANDBY", "amber")))

    total = len(SEARCH_ENGINES)
    if engines_state == "checking":
        rows.append(ck_row(f"Engines — / {total}", "Pinging all engines via Tor…", pill("CHECKING", "neutral")))
    elif engines_state == "skipped":
        rows.append(ck_row(f"Engines 0 / {total} checked", "Tor circuit down · engines not checked", pill("NOT CHECKED", "amber")))
    else:
        eng = (pf or {}).get("engines")
        if not eng:
            tor = (pf or {}).get("tor")
            if tor and tor["status"] == "down":
                rows.append(ck_row(f"Engines 0 / {total} checked", "Tor circuit down · engines not checked", pill("NOT CHECKED", "amber")))
            else:
                rows.append(ck_row(f"Engines — / {total}", "Not checked yet", pill("CHECKING", "neutral")))
        else:
            up = sum(1 for r in eng if r["status"] == "up")
            down_names = [r["name"] for r in eng if r["status"] != "up"]
            if up == total:
                meta, chip = "all engines answering · search at full width", pill("GO", "go")
            elif up > 0:
                meta, chip = f"{', '.join(down_names)} down · search degraded", pill("GO · DEGRADED", "amber")
            else:
                meta, chip = "no engine answering · search cannot run", pill("NO-GO", "nogo")
            rows.append(ck_row(f"Engines {up} / {total} reachable", meta, chip))

    return panel_head("Pre-flight · GO / NO-GO", "READ BEFORE LAUNCH") + "".join(rows)


def engines_html(pf) -> str:
    total = len(SEARCH_ENGINES)
    eng = (pf or {}).get("engines")
    if not eng:
        tor_down = pf is not None and (pf.get("tor") or {}).get("status") == "down"
        body = ('<div class="empt">Engines not checked'
                + (" — Tor circuit down." if tor_down else " — pre-flight has not run yet.")
                + "</div>")
        return panel_html("Engines", f"{total} ENGINES · NOT CHECKED", body)
    up = sum(1 for r in eng if r["status"] == "up")
    body = []
    for r in eng:
        if r["status"] == "up":
            body.append(f'<div class="eng"><span class="e">{esc(r["name"])}</span>'
                        f'<span class="l mono">{r["latency_ms"]} MS</span>'
                        f'<span class="chip c-go">UP</span></div>')
        else:
            body.append(f'<div class="eng"><span class="e down">{esc(r["name"])}</span>'
                        f'<span class="l mono">—</span>'
                        f'<span class="chip c-dn">DOWN</span></div>')
    return panel_html("Engines", f"{up} / {total} REACHABLE", "".join(body))


def log_html(entries, state_label: str) -> str:
    if not entries:
        body = '<div class="empt">No entries yet — the log fills when a mission launches.</div>'
    else:
        body = "".join(
            f'<div class="f"><span class="id mono">{i:02d}</span>'
            f'<span class="txt">{e}</span></div>'
            for i, e in enumerate(entries, 1)
        )
    return panel_html("Mission log", state_label, body)


def sources_html(sources, meta: str) -> str:
    if not sources:
        body = '<div class="empt">No sources yet.</div>'
    else:
        rows = []
        for i, item in enumerate(sources, 1):
            title = esc(item.get("title") or "Untitled")
            link = esc(item.get("link") or "")
            host = urlparse(item.get("link") or "").netloc
            rows.append(
                f'<div class="sr"><span class="k mono">{i:02d}</span>'
                f'<span class="h"><a href="{link}" target="_blank" rel="noopener noreferrer">{title}</a></span>'
                f'<span class="host mono">{esc(host)}</span></div>'
            )
        body = "".join(rows)
    return panel_html("Sources", meta, body)


def launch_meta_html(values: dict) -> str:
    parts = "".join(
        f'<div><div class="k">{esc(k)}</div><div class="v mono">{esc(v)}</div></div>'
        for k, v in values.items()
    )
    return f'<div class="ms">{parts}</div>'


def footer_html(left: str, right: str) -> str:
    return (
        f'<div class="ft"><span>{left}</span>'
        '<span class="credit">Made by '
        '<a href="https://www.linkedin.com/in/apurvsinghgautam/" target="_blank" rel="noopener noreferrer">Apurv Singh Gautam</a>'
        '</span>'
        f'<span class="mono">{right}</span></div>'
    )


# --- Page shell ---

st.set_page_config(
    page_title="Robin: AI-Powered Dark Web OSINT Tool",
    page_icon=".github/assets/robin_logo.png",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :root{
      --bg:#0B0D10; --panel:#15191F; --panel2:#0F1319; --line:#232830; --line2:#1D222A;
      --ink:#E8EAF0; --dim:#8C94A4; --faint:#7C8494; --act:#4C8DFF;
      --go:#3ED598; --warn:#FFB020; --down:#FF5A5A;
      --mono: ui-monospace, "SF Mono", SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
    }
    html, body, .stApp, [data-testid="stAppViewContainer"]{ background:var(--bg) !important; color:var(--ink); }
    .stApp{ font-family: system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }
    #MainMenu, [data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"],
    [data-testid="stStatusWidget"], footer, [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"]
      { display:none !important; }
    [data-testid="stMainBlockContainer"], .block-container{ padding: 0 24px !important; max-width: 100% !important; }
    [data-testid="stVerticalBlock"]{ gap: 14px; }
    .mono{ font-family: var(--mono); font-variant-numeric: tabular-nums; }
    ::selection{ background:var(--act); color:#06121F; }
    *{ caret-color: var(--act); }
    :focus-visible{ outline:1px solid var(--act) !important; outline-offset:2px; }
    a{ color:var(--act); text-decoration-color: rgba(76,141,255,.45); text-underline-offset:2px; }
    ::-webkit-scrollbar{ width:10px; height:10px; }
    ::-webkit-scrollbar-thumb{ background:#2A3038; border-radius:6px; }
    ::-webkit-scrollbar-track{ background:transparent; }

    /* header */
    .hdr{ display:flex; align-items:center; justify-content:space-between; gap:16px; min-height:72px;
          padding:12px 0; border-bottom:1px solid var(--line); flex-wrap:wrap; }
    .brand{ display:flex; align-items:baseline; gap:14px; flex-wrap:wrap; }
    .brand .mark{ height:48px; width:auto; align-self:center; display:block; }
    .brand h1{ font-size:26px; letter-spacing:.18em; font-weight:700; margin:0; }
    .brand .sub{ font-size:11px; letter-spacing:.22em; color:var(--faint); text-transform:uppercase; }
    .hmeta{ display:flex; gap:22px; align-items:center; flex-wrap:wrap; }
    .hm{ display:flex; flex-direction:column; gap:3px; }
    .hm .lbl{ font-size:10px; letter-spacing:.2em; color:var(--faint); text-transform:uppercase; }
    .hm .val{ font-size:15px; }
    .torchip{ display:inline-flex; align-items:center; gap:9px; padding:7px 12px; border-radius:2px;
              border:1px solid var(--line); background:var(--panel2); color:var(--dim);
              font-size:12px; letter-spacing:.08em; font-weight:600; }
    .torchip.on{ border-color:#24382A; background:#101A14; color:var(--ink); }
    .torchip.off{ border-color:#43222A; background:#1A1113; color:var(--ink); }
    .dot{ width:8px; height:8px; border-radius:50%; background:var(--faint); display:inline-block; }
    .dot.on{ background:var(--go); }
    .dot.off{ background:var(--down); }

    /* phase strip */
    .strip{ display:flex; border-bottom:1px solid var(--line); background:#0E1216; flex-wrap:wrap; }
    .stage{ flex:1; min-width:118px; display:flex; flex-direction:column; justify-content:center; gap:3px;
            padding:10px 16px; border-right:1px solid var(--line2); border-top:2px solid transparent; }
    .stage .sn{ font-size:10px; letter-spacing:.22em; color:var(--faint); }
    .stage .st{ font-size:12.5px; letter-spacing:.08em; font-weight:600; color:var(--dim); text-transform:uppercase; }
    .stage .sd{ font-size:10px; color:var(--faint); }
    .stage.done .st{ color:#B8BFCC; }
    .stage.done .sn{ color:var(--go); }
    .stage.live{ background:rgba(76,141,255,.10); border-top-color:var(--act); }
    .stage.live .st{ color:var(--ink); }
    .stage.live .sn{ color:var(--act); }
    .stage.halt{ background:rgba(255,176,32,.08); border-top-color:var(--warn); }
    .stage.halt .st{ color:var(--ink); }
    .stage.halt .sn{ color:var(--warn); }
    .sr2{ flex:0 0 230px; min-width:0; overflow:hidden; display:flex; flex-direction:column; justify-content:center; gap:7px; padding:10px 16px; }
    .sr2 .k{ font-size:10px; letter-spacing:.2em; color:var(--faint); white-space:nowrap; }
    .sr2 .sd{ white-space:nowrap; }
    .bar{ height:3px; background:#1B2026; position:relative; }
    .bar i{ position:absolute; left:0; top:0; bottom:0; background:var(--act); }

    /* panels */
    .pnl{ background:var(--panel); border:1px solid var(--line); border-radius:2px; overflow:hidden; }
    .panelhdr{ height:38px; display:flex; align-items:center; justify-content:space-between; gap:10px;
               padding:0 14px; border-bottom:1px solid var(--line); }
    .panelhdr .t{ font-size:11px; letter-spacing:.2em; font-weight:600; color:#9AA2B2; text-transform:uppercase; }
    .panelhdr .m{ font-size:10px; letter-spacing:.12em; color:var(--faint); font-family:var(--mono);
                  text-transform:uppercase; text-align:right; }

    /* checklist rows */
    .ck{ display:flex; align-items:center; justify-content:space-between; gap:10px; min-height:50px;
         padding:10px 16px; border-bottom:1px solid var(--line2); }
    .ck:last-child{ border-bottom:0; }
    .ck .name{ font-size:15px; font-weight:500; }
    .ck .meta{ font-size:11px; color:var(--faint); margin-top:3px; line-height:1.45; }
    .pill{ font-size:11px; letter-spacing:.14em; font-weight:700; padding:5px 10px; border-radius:2px;
           white-space:nowrap; text-transform:uppercase; }
    .p-go{ color:#06170E; background:var(--go); }
    .p-nogo{ color:#1F0505; background:var(--down); }
    .p-amber{ color:var(--warn); border:1px solid #4A3A14; background:rgba(255,176,32,.08); }
    .p-neutral{ color:var(--dim); border:1px solid var(--line); background:var(--panel2); }

    /* log / tables */
    .f{ display:flex; gap:12px; padding:9px 16px; border-bottom:1px solid var(--line2); }
    .f:last-child{ border-bottom:0; }
    .f .id{ color:var(--act); width:26px; flex:none; font-size:12px; }
    .f .txt{ font-size:13px; color:#B8BFCC; line-height:1.55; }
    .empt{ padding:14px 16px; font-size:12.5px; color:var(--faint); }
    .sr{ display:grid; grid-template-columns:26px minmax(0,1fr) minmax(0,150px); gap:0 10px; align-items:center;
         min-height:32px; padding:4px 16px; border-bottom:1px solid var(--line2); }
    .sr:last-child{ border-bottom:0; }
    .sr .k{ font-size:12px; color:var(--faint); }
    .sr .h{ overflow:hidden; text-overflow:ellipsis; white-space:nowrap; font-size:12.5px; }
    .sr .host{ font-size:11px; color:var(--faint); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .eng{ display:grid; grid-template-columns:minmax(0,1fr) 64px 62px; gap:0 10px; align-items:center;
          min-height:30px; padding:0 16px; border-bottom:1px solid var(--line2); }
    .eng:last-child{ border-bottom:0; }
    .eng .e{ font-size:12.5px; color:#B8BFCC; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .eng .e.down{ color:var(--faint); }
    .eng .l{ font-size:11.5px; color:var(--dim); text-align:right; }
    .chip{ font-size:10px; letter-spacing:.12em; font-weight:700; padding:4px 8px; border-radius:2px;
           text-align:center; text-transform:uppercase; }
    .c-go{ color:var(--go); background:rgba(62,213,152,.08); border:1px solid #1E3A2C; }
    .c-dn{ color:var(--down); background:rgba(255,90,90,.08); border:1px solid #43222A; }

    /* launch */
    .ms{ display:flex; gap:26px; padding:12px 2px 2px; border-top:1px solid var(--line2); margin-top:12px; flex-wrap:wrap; }
    .ms .k{ font-size:10px; letter-spacing:.2em; color:var(--faint); text-transform:uppercase; }
    .ms .v{ font-size:14px; margin-top:3px; }

    /* report */
    .rstate{ font-size:11px; letter-spacing:.14em; color:var(--faint); text-transform:uppercase; }
    .rstate .mono{ letter-spacing:0; text-transform:none; color:var(--dim); }
    .dl{ display:inline-block; font-size:11px; letter-spacing:.14em; font-weight:700; text-transform:uppercase;
         color:var(--ink); text-decoration:none; border:1px solid var(--line); background:var(--panel2);
         padding:10px 14px; border-radius:2px; }
    .dl:hover{ border-color:var(--act); color:var(--act); }
    .dl + .dl{ margin-left:8px; }
    .alert{ border:1px solid #43222A; background:#170F11; border-radius:2px; padding:14px 16px; }
    .alert .ah{ font-size:12px; letter-spacing:.18em; font-weight:700; color:var(--down); text-transform:uppercase; }
    .alert .am{ font-size:13px; color:#D8A8A8; margin-top:8px; word-break:break-word; font-family:var(--mono); }
    .alert ul{ margin:10px 0 0; padding-left:18px; }
    .alert li{ font-size:12.5px; color:var(--dim); line-height:1.6; }

    /* footer */
    .ft{ display:flex; align-items:center; justify-content:space-between; gap:12px; margin-top:18px;
         border-top:1px solid var(--line); padding:10px 0 14px; flex-wrap:wrap; }
    .ft span{ font-size:10.5px; letter-spacing:.14em; color:var(--faint); text-transform:uppercase; }
    .ft .credit{ text-transform:none; letter-spacing:.04em; }
    .ft a{ color:var(--faint); text-decoration-color:rgba(124,132,148,.5); }
    .ft a:hover{ color:var(--ink); }

    /* report prose measure */
    .st-key-report_body [data-testid="stMarkdownContainer"]{ max-width:72ch; }

    /* native widget skin */
    [data-testid="stWidgetLabel"] p{ font-size:10.5px !important; letter-spacing:.18em !important;
        text-transform:uppercase; color:var(--faint) !important; font-weight:600; }
    [data-testid="stTextInput"] > div > div{ border:1px solid #2E3A4C !important; border-radius:2px !important;
        background:var(--panel2) !important; }
    [data-testid="stTextInput"] input{ font-family:var(--mono); font-size:17px; color:var(--ink) !important; height:52px; }
    input::placeholder{ color:var(--faint) !important; font-family:var(--mono); }
    [data-testid="stSelectbox"] [data-baseweb="select"] > div{ background:var(--panel2) !important;
        border:1px solid #2A3038 !important; border-radius:2px !important; min-height:42px; }
    [data-baseweb="popover"] li{ font-family:var(--mono); }
    [data-baseweb="slider"] [role="slider"]{ background:var(--ink) !important; border:3px solid var(--act) !important; }
    [data-testid="stSliderThumbValue"], [data-testid="stSliderTickBar"]{ font-family:var(--mono); }
    [data-testid="stSliderThumbValue"]{ color:var(--ink) !important; font-size:12px; }
    .stButton > button, [data-testid="stFormSubmitButton"] > button{ border-radius:2px; text-transform:uppercase;
        letter-spacing:.12em; font-size:11.5px; font-weight:700; height:40px; border:1px solid var(--line);
        background:var(--panel2); color:#C9D0DC; }
    .stButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover{ border-color:var(--act);
        color:var(--ink); background:var(--panel2); }
    [data-testid="stFormSubmitButton"]{ width:100% !important; flex:1 1 100%; }
    [data-testid="stFormSubmitButton"] > button{ width:100%; height:54px; font-size:14px; font-weight:800;
        letter-spacing:.14em; }
    [data-testid="stFormSubmitButton"] > button[kind="primaryFormSubmit"]{ background:var(--act); color:#06121F;
        border:0; }
    [data-testid="stFormSubmitButton"] > button[kind="primaryFormSubmit"]:hover{ background:#3D7FF0; color:#06121F; }
    .st-key-preflight_panel .stButton > button{ height:30px; min-height:30px; font-size:10px; background:transparent;
        border-color:var(--line); color:var(--faint); padding:0 10px; }
    .st-key-preflight_panel .stButton > button:hover{ color:var(--ink); border-color:var(--faint); }
    [data-testid="stBaseButton-pills"], [data-testid="stBaseButton-pillsActive"]{ border-radius:2px; font-size:12px;
        padding:7px 11px; background:var(--panel2); border:1px solid #2A3038; color:#A6AEBD; letter-spacing:.02em; }
    [data-testid="stBaseButton-pillsActive"]{ border-color:#2F5FA8; background:rgba(76,141,255,.16); color:var(--ink);
        font-weight:600; }
    [data-testid="stExpander"] details{ border:0; border-top:1px solid var(--line); border-radius:0;
        background:transparent; }
    [data-testid="stExpander"] summary{ font-size:10.5px; letter-spacing:.18em; text-transform:uppercase;
        color:#9AA2B2; font-weight:600; }
    [data-testid="stExpander"] summary:hover{ color:var(--ink); }
    [data-testid="stTextArea"] textarea{ background:var(--panel2) !important; border:1px solid #2A3038 !important;
        color:var(--ink) !important; font-family:var(--mono); font-size:12.5px; border-radius:2px; }
    [data-testid="stTextArea"] textarea:disabled{ color:var(--dim) !important; -webkit-text-fill-color:var(--dim) !important;
        background:#0D1116 !important; cursor:not-allowed; }
    [data-testid="stCaptionContainer"] p{ color:var(--faint) !important; font-size:11px !important; }
    [data-testid="stForm"]{ border:0; padding:0; }
    [data-testid="stForm"] > [data-testid="stVerticalBlock"]{ align-items:stretch; }
    [data-testid="stForm"] [data-testid="stElementContainer"]:has([data-testid="stFormSubmitButton"]){ width:100%; }
    [data-testid="stSpinner"]{ color:var(--dim); }

    /* native-widget panels */
    .st-key-config_panel, .st-key-domains_panel, .st-key-prompt_panel, .st-key-archive_panel,
    .st-key-preflight_panel, .st-key-launch_panel, .st-key-report_panel{ background:var(--panel);
        border:1px solid var(--line); border-radius:2px; overflow:hidden; gap:0; }
    .st-key-config_panel [data-testid="stVerticalBlock"], .st-key-domains_panel [data-testid="stVerticalBlock"],
    .st-key-prompt_panel [data-testid="stVerticalBlock"], .st-key-archive_panel [data-testid="stVerticalBlock"],
    .st-key-preflight_panel [data-testid="stVerticalBlock"], .st-key-launch_panel [data-testid="stVerticalBlock"],
    .st-key-report_panel [data-testid="stVerticalBlock"]{ gap:0; }
    .st-key-config_body, .st-key-domains_body, .st-key-prompt_body, .st-key-launch_body,
    .st-key-archive_body, .st-key-report_body{ padding:12px 14px 14px; }
    .st-key-config_body [data-testid="stVerticalBlock"]{ gap:12px; }
    .st-key-report_body [data-testid="stVerticalBlock"]{ gap:10px; }

    @media (max-width: 760px){
      [data-testid="stMainBlockContainer"], .block-container{ padding: 0 14px !important; }
      .stage{ flex:1 1 33%; min-width:0; padding:8px 10px; }
      .sr2{ flex:1 1 100%; }
      .ck{ flex-direction:column; align-items:flex-start; gap:6px; }
      .eng{ grid-template-columns:minmax(0,1fr) 58px 58px; padding:0 12px; }
      .sr{ grid-template-columns:22px minmax(0,1fr); }
      .sr .host{ display:none; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Boot state ---

_ss = st.session_state
if "phase" not in _ss or len(_ss["phase"]) != 6:
    _ss["phase"] = [{"state": "pending", "ms": None} for _ in range(6)]
_ss.setdefault("log", [])
_ss.setdefault("run_meta", None)
_ss.setdefault("preflight", None)

model_options = get_model_choices()

header_slot = st.empty()
header_slot.markdown(header_html("checking"), unsafe_allow_html=True)
strip_slot = st.empty()

if not model_options:
    strip_slot.markdown(strip_html(_ss["phase"]), unsafe_allow_html=True)
    st.markdown(
        '<div class="alert" style="margin-top:18px"><div class="ah">NO LLM MODELS AVAILABLE</div>'
        '<div class="am">No API keys or local providers are configured. '
        'Set at least one in your .env file and restart Robin.</div>'
        '<ul>'
        + "".join(
            f'<li>{esc(name)} — {"configured" if env_is_set(value) else "key not set"}</li>'
            for name, value, _ in CLOUD_PROVIDERS
        )
        + '<li>Ollama / llama.cpp — '
        + ("configured" if (env_is_set(OLLAMA_BASE_URL) or env_is_set(LLAMA_CPP_BASE_URL)) else "not configured")
        + "</li></ul></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        footer_html("No models available · set a provider key in .env and restart Robin", "STANDBY · NO MODELS"),
        unsafe_allow_html=True,
    )
    st.stop()

# Guards so a changed model/provider list never strands a stale widget value
if _ss.get("model_select") not in model_options:
    _ss.pop("model_select", None)
_preset_labels = list(PRESET_OPTIONS.keys())
if _ss.get("preset_pills") not in _preset_labels:
    _ss.pop("preset_pills", None)

default_model_index = next(
    (idx for idx, name in enumerate(model_options) if name.lower() == "gpt4o"), 0
)

# --- Pre-flight ---

if _ss.get("preflight") is None:
    _ss["preflight"] = {"tor": None, "llm": None, "engines": None, "model": None, "at": None}
pf = _ss["preflight"]

left, center, right = st.columns([336, 660, 380], gap="medium")

# Left rail: configuration, research domains, prompt, archive
with left:
    with st.container(key="config_panel"):
        st.markdown(panel_head("Config", f"{len(model_options)} MODELS AVAILABLE"), unsafe_allow_html=True)
        with st.container(key="config_body"):
            model = st.selectbox(
                "Select LLM model",
                model_options,
                index=default_model_index,
                key="model_select",
            )
            if any(name not in {"gpt4o", "gpt-4.1", "claude-3-5-sonnet-latest", "llama3.1", "gemini-2.5-flash"} for name in model_options):
                st.caption("Locally detected Ollama models are automatically added to this list.")
            threads = st.slider("Scraping threads", 1, 16, 4, key="thread_slider")
            max_results = st.slider(
                "Max results to filter", 10, 100, 50, key="max_results_slider",
                help="Cap the number of raw search results passed to the LLM filter step.",
            )
            max_scrape = st.slider(
                "Max pages to scrape", 3, 20, 10, key="max_scrape_slider",
                help="Cap the number of filtered results that get scraped for content.",
            )

    _preset_now = _ss.get("preset_pills") or _preset_labels[0]
    with st.container(key="domains_panel"):
        st.markdown(panel_head("Research domains", esc(_preset_now).upper()), unsafe_allow_html=True)
        with st.container(key="domains_body"):
            _picked = st.pills(
                "Research domain",
                _preset_labels,
                default=_preset_labels[0],
                key="preset_pills",
                label_visibility="collapsed",
            )
            if _picked is None:
                _picked = _ss.get("preset_pills") or _preset_labels[0]
            selected_preset_label = _picked
            selected_preset = PRESET_OPTIONS[selected_preset_label]

    with st.container(key="prompt_panel"):
        st.markdown(panel_head("Prompt", "SYSTEM + CUSTOM"), unsafe_allow_html=True)
        with st.container(key="prompt_body"):
            with st.expander("Prompt settings"):
                st.text_area(
                    "System prompt",
                    value=PRESET_PROMPTS[selected_preset].strip(),
                    height=200,
                    disabled=True,
                    key="system_prompt_display",
                )
                custom_instructions = st.text_area(
                    "Custom instructions (optional)",
                    placeholder=PRESET_PLACEHOLDERS[selected_preset],
                    height=100,
                    key="custom_instructions",
                )

    saved_investigations = load_investigations()
    with st.container(key="archive_panel"):
        st.markdown(
            panel_head("Past investigations", f"{len(saved_investigations)} RECORDS" if saved_investigations else "EMPTY"),
            unsafe_allow_html=True,
        )
        with st.container(key="archive_body"):
            if saved_investigations:
                inv_labels = [
                    f"{inv['_filename'].replace('investigation_', '').replace('.json', '')} — {inv['query'][:40]}"
                    for inv in saved_investigations
                ]
                if _ss.get("inv_select") not in ["(none)"] + inv_labels:
                    _ss.pop("inv_select", None)
                selected_inv_label = st.selectbox(
                    "Load investigation", ["(none)"] + inv_labels, key="inv_select"
                )
                if selected_inv_label != "(none)":
                    selected_inv_idx = inv_labels.index(selected_inv_label)
                    if st.button("Load record", use_container_width=True, key="load_inv_btn"):
                        _ss["loaded_investigation"] = saved_investigations[selected_inv_idx]
                        st.rerun()
            else:
                st.caption("No saved investigations yet.")

# --- Center: pre-flight board, launch, report ---

with center:
    with st.container(key="preflight_panel"):
        board_slot = st.empty()
        _current_model = _ss.get("model_select", model_options[default_model_index])
        board_slot.markdown(board_html(pf, _current_model, model_options), unsafe_allow_html=True)

        # Re-ping the LLM whenever the selected model changes
        if pf.get("model") is not None and pf.get("model") != _current_model:
            pf["model"] = _current_model
            pf["llm"] = None
            board_slot.markdown(board_html(pf, _current_model, model_options), unsafe_allow_html=True)
            with st.spinner(f"PRE-FLIGHT · TESTING {_current_model}…"):
                pf["llm"] = check_llm_health(_current_model)
            board_slot.markdown(board_html(pf, _current_model, model_options), unsafe_allow_html=True)

        if pf.get("tor") is None:
            with st.spinner("PRE-FLIGHT · ESTABLISHING TOR CIRCUIT…"):
                pf["tor"] = check_tor_proxy()
            board_slot.markdown(board_html(pf, _current_model, model_options), unsafe_allow_html=True)
            with st.spinner(f"PRE-FLIGHT · TESTING {_current_model}…"):
                pf["llm"] = check_llm_health(_current_model)
            if pf["tor"]["status"] == "up":
                board_slot.markdown(
                    board_html(pf, _current_model, model_options, engines_state="checking"),
                    unsafe_allow_html=True,
                )
                with st.spinner("PRE-FLIGHT · PINGING 16 ENGINES VIA TOR…"):
                    pf["engines"] = check_search_engines()
            else:
                pf["engines"] = None
            pf["model"] = _current_model
            pf["at"] = datetime.now().isoformat()

        board_slot.markdown(board_html(pf, _current_model, model_options), unsafe_allow_html=True)
        if st.button("Re-run pre-flight", key="preflight_rerun"):
            _ss["preflight"] = None
            st.rerun()

    with st.container(key="launch_panel"):
        st.markdown(panel_head("Launch", "QUERY INPUT"), unsafe_allow_html=True)
        with st.container(key="launch_body"):
            with st.form("search_form", clear_on_submit=True):
                query = st.text_input(
                    "Enter dark web search query",
                    placeholder="Enter dark web search query",
                    label_visibility="collapsed",
                    key="query_input",
                )
                run_button = st.form_submit_button("Search dark web", type="primary")
            if run_button and not query:
                st.markdown(
                    '<div class="empt" style="padding:8px 2px 0">No query entered — type a lead before launching.</div>',
                    unsafe_allow_html=True,
                )
            launch_meta_slot = st.empty()
            status_slot = st.empty()

    err_slot = st.empty()

    with st.container(key="report_panel"):
        report_meta_slot = st.empty()
        with st.container(key="report_body"):
            report_state_slot = st.empty()
            summary_slot = st.empty()
            notes_slot = st.empty()
            dl_slot = st.empty()

# --- Right: mission log, sources, engines ---

def _log_state_label():
    if _ss.get("run_meta"):
        if any(p["state"] == "live" for p in _ss["phase"]):
            return "STREAMING"
        if all(p["state"] == "done" for p in _ss["phase"]):
            return f"{len(_ss['log'])} ENTRIES · COMPLETE"
    if any(p["state"] == "halt" for p in _ss["phase"]):
        return "HALTED"
    if _ss.get("loaded_investigation"):
        return "ARCHIVE"
    return "AWAITING LAUNCH"


with right:
    log_slot = st.empty()
    sources_slot = st.empty()
    engines_slot = st.empty()
    engines_slot.markdown(engines_html(pf), unsafe_allow_html=True)

footer_slot = st.empty()


# --- Refresh helpers ---

def _provider_label(model_name: str) -> str:
    return (pf.get("llm") or {}).get("provider") or provider_of(model_name)


def _refresh_strip(mode="idle", total_ms=None, archive_ts=None):
    strip_slot.markdown(strip_html(_ss["phase"], mode=mode, total_ms=total_ms, archive_ts=archive_ts), unsafe_allow_html=True)


def _refresh_header(state_label, elapsed=None):
    tor_state = "checking"
    if pf.get("tor"):
        tor_state = "on" if pf["tor"]["status"] == "up" else "off"
    header_slot.markdown(header_html(tor_state, elapsed=elapsed, state_label=state_label), unsafe_allow_html=True)


def _refresh_log():
    log_slot.markdown(log_html(_ss["log"], _log_state_label()), unsafe_allow_html=True)


def _refresh_launch_meta(phase_label="STANDBY", pages="—", filtered="—", discarded="—"):
    launch_meta_slot.markdown(
        launch_meta_html({
            "Phase": phase_label,
            "Threads": f"{_ss.get('thread_slider', 4)} / 16",
            "Pages": pages,
            "Filtered": filtered,
            "Hits discarded": discarded,
        }),
        unsafe_allow_html=True,
    )


def _refresh_report_meta(state):
    report_meta_slot.markdown(panel_head("Report · Findings", state), unsafe_allow_html=True)


def _log_add(text):
    _ss["log"].append(text)
    _refresh_log()


def _phase_start(idx):
    _ss["phase"][idx]["state"] = "live"
    _refresh_strip("running")


def _phase_done(idx, ms):
    _ss["phase"][idx]["state"] = "done"
    _ss["phase"][idx]["ms"] = ms
    _refresh_strip("running")


def _render_pipeline_error(stage: str, err: Exception) -> None:
    message = str(err).strip() or err.__class__.__name__
    lower_msg = message.lower()
    hints = [
        "- Confirm the relevant API key is set in your `.env` or shell before launching Streamlit.",
        "- Keys copied from dashboards often include hidden spaces; re-copy if authentication keeps failing.",
        "- Restart the app after updating environment variables so the new values are picked up.",
    ]

    if any(token in lower_msg for token in ("credit balance", "billing", "quota", "insufficient_quota", "payment required")):
        hints.insert(0, "- The provider accepted the key, but the account behind it is out of credit or quota — top up that provider's billing, or switch to another model in the Config rail.")
    elif any(token in lower_msg for token in ("anthropic", "x-api-key", "invalid api key", "authentication")):
        hints.insert(0, "- Claude/Anthropic models require a valid `ANTHROPIC_API_KEY`.")
    elif "openrouter" in lower_msg or "user not found" in lower_msg or "code: 401" in lower_msg:
        hints.insert(0, "- OpenRouter 401/User not found usually means the API key is invalid/expired or has leading/trailing characters.")
        hints.insert(1, "- Set `OPENROUTER_API_KEY` without extra spaces and verify the key is active in your OpenRouter account.")
        hints.insert(2, "- Keep `OPENROUTER_BASE_URL` as `https://openrouter.ai/api/v1` unless you intentionally use a custom gateway.")
    elif "deepseek" in lower_msg:
        hints.insert(0, "- DeepSeek models require a valid `DEEPSEEK_API_KEY`.")
        hints.insert(1, "- Keep `DEEPSEEK_BASE_URL` as `https://api.deepseek.com/v1` unless you use a custom gateway.")
    elif "openai" in lower_msg or "gpt" in lower_msg:
        hints.insert(0, "- OpenAI models require `OPENAI_API_KEY` with access to the chosen model.")
    elif "google" in lower_msg or "gemini" in lower_msg:
        hints.insert(0, "- Google Gemini models need `GOOGLE_API_KEY` or Application Default Credentials.")

    err_slot.markdown(
        '<div class="alert"><div class="ah">PIPELINE HALTED · FAILED TO ' + esc(stage.upper()) + "</div>"
        '<div class="am">Error: ' + esc(message) + "</div>"
        "<ul>" + "".join(f"<li>{esc(h.lstrip('- '))}</li>" for h in hints) + "</ul></div>",
        unsafe_allow_html=True,
    )
    footer_slot.markdown(
        footer_html("Investigation halted on error · nothing is saved", f"HALTED · {esc(stage.upper())}"),
        unsafe_allow_html=True,
    )
    st.stop()


def _initial_layout():
    """Fill the idle / archive view of every panel."""
    loaded = _ss.get("loaded_investigation")
    if loaded:
        ts = (loaded.get("timestamp") or "")[:16]
        _refresh_header("ARCHIVE", elapsed=None)
        _refresh_strip("archive", archive_ts=ts)
        _ss["log"] = [
            f"Restored investigation {esc(ts)} · query “{esc(loaded.get('query'))}” · "
            f"{len(loaded.get('sources') or [])} sources · model {esc(loaded.get('model'))}."
        ]
        _refresh_log()
        sources_slot.markdown(
            sources_html(loaded.get("sources") or [], f"{len(loaded.get('sources') or [])} SOURCES · ARCHIVE"),
            unsafe_allow_html=True,
        )
        _refresh_report_meta(f"ARCHIVE · {ts}")
        report_state_slot.markdown(
            f'<div class="rstate"><span class="mono">{esc(loaded.get("query"))}</span> · '
            f'<span class="mono">{esc(loaded.get("model"))}</span> · {esc(loaded.get("preset"))}</div>',
            unsafe_allow_html=True,
        )
        summary_slot.markdown(loaded.get("summary") or "")
        with notes_slot.container():
            with st.expander("Run notes"):
                st.markdown(f"**Refined query:** `{loaded.get('refined_query', '')}`")
                st.markdown(f"**Model:** `{loaded.get('model', '')}` &nbsp;&nbsp; **Domain:** {loaded.get('preset', '')}")
                st.markdown(f"**Sources:** {len(loaded.get('sources') or [])}")
        with dl_slot.container():
            try:
                st.markdown(report_downloads_html(loaded), unsafe_allow_html=True)
            except Exception as e:
                _log_add(f"Report export unavailable · {esc(str(e))}")
            if st.button("Clear record", key="clear_record"):
                del _ss["loaded_investigation"]
                st.rerun()
        footer_slot.markdown(
            footer_html("Investigation record · loaded from disk", f"{esc(loaded.get('_filename', ''))} · LOADED"),
            unsafe_allow_html=True,
        )
        _refresh_launch_meta(phase_label="ARCHIVE")
    else:
        _refresh_header("READY", elapsed=None)
        _refresh_strip("idle")
        _refresh_log()
        sources_slot.markdown(sources_html([], "NO SEARCH RUN"), unsafe_allow_html=True)
        _refresh_report_meta("EMPTY")
        report_state_slot.markdown(
            '<div class="rstate">No report yet · enter a query and press SEARCH DARK WEB</div>',
            unsafe_allow_html=True,
        )
        summary_slot.markdown("")
        notes_slot.empty()
        dl_slot.empty()
        footer_slot.markdown(
            footer_html("Investigations save to disk · report downloads as PDF / JSON / Markdown", "STANDBY · NO INVESTIGATION RUN"),
            unsafe_allow_html=True,
        )
        _refresh_launch_meta()


_initial_layout()

# --- Run the pipeline ---

if run_button and query:
    _ss.pop("loaded_investigation", None)
    for k in ["refined", "results", "filtered", "scraped", "streamed_summary"]:
        _ss.pop(k, None)
    _ss["log"] = []
    _ss["phase"] = [{"state": "pending", "ms": None} for _ in range(6)]
    run_started = time.monotonic()
    _ss["run_meta"] = {
        "query": query,
        "started_at": datetime.now().isoformat(),
        "model": model,
        "preset": selected_preset_label,
    }
    notes_slot.empty()
    dl_slot.empty()
    err_slot.empty()
    _refresh_report_meta("STREAMING")
    report_state_slot.markdown(
        f'<div class="rstate"><span class="mono">{esc(query)}</span> · '
        f'<span class="mono">{esc(model)}</span> · {esc(selected_preset_label)}</div>',
        unsafe_allow_html=True,
    )

    # Stage 1 - Load LLM
    _phase_start(0)
    _refresh_header("RUNNING", elapsed=time.monotonic() - run_started)
    with status_slot.container():
        with st.spinner("LOADING LLM…"):
            _t0 = time.monotonic()
            try:
                llm = get_llm(model)
            except Exception as e:
                _ss["phase"][0]["state"] = "halt"
                _refresh_strip("halted")
                _refresh_header("HALTED", elapsed=time.monotonic() - run_started)
                _render_pipeline_error("load the selected LLM", e)
            _ms = round((time.monotonic() - _t0) * 1000)
    _phase_done(0, _ms)
    _provider = _provider_label(model)
    _log_add(f"Loaded {mono(model)}{' · ' + esc(_provider) if _provider else ''} in {_ms} ms.")
    _refresh_launch_meta(phase_label="01 LOAD")

    # Stage 2 - Refine query
    _phase_start(1)
    _refresh_header("RUNNING", elapsed=time.monotonic() - run_started)
    with status_slot.container():
        with st.spinner("REFINING QUERY…"):
            _t0 = time.monotonic()
            try:
                _ss["refined"] = refine_query(llm, query)
            except Exception as e:
                _ss["phase"][1]["state"] = "halt"
                _refresh_strip("halted")
                _refresh_header("HALTED", elapsed=time.monotonic() - run_started)
                _render_pipeline_error("refine the query", e)
            _ms = round((time.monotonic() - _t0) * 1000)
    _phase_done(1, _ms)
    _log_add(
        f"Refined query → {mono(_ss['refined'])} · {len(str(_ss['refined']).split())} terms · {_ms} ms."
    )
    _refresh_launch_meta(phase_label="02 REFINE")

    # Stage 3 - Search dark web
    _phase_start(2)
    _refresh_header("RUNNING", elapsed=time.monotonic() - run_started)
    with status_slot.container():
        with st.spinner("SEARCHING 16 ENGINES OVER TOR…"):
            _t0 = time.monotonic()
            _ss["results"] = cached_search_results(_ss["refined"], threads)
            _ms = round((time.monotonic() - _t0) * 1000)
    # Cap results before LLM filter step
    if len(_ss["results"]) > max_results:
        _ss["results"] = _ss["results"][:max_results]
    _phase_done(2, _ms)
    _eng = pf.get("engines") or []
    _up = sum(1 for r in _eng if r["status"] == "up")
    _log_add(
        f"Searched {len(SEARCH_ENGINES)} engines over Tor · {mono(len(_ss['results']))} unique hits · "
        f"{_up}/{len(SEARCH_ENGINES)} engines answering · {_ms} ms."
    )
    _refresh_launch_meta(phase_label="03 SEARCH")

    # Stage 4 - Filter results
    _phase_start(3)
    _refresh_header("RUNNING", elapsed=time.monotonic() - run_started)
    with status_slot.container():
        with st.spinner("FILTERING HITS…"):
            _t0 = time.monotonic()
            try:
                _ss["filtered"] = filter_results(llm, _ss["refined"], _ss["results"])
            except Exception as e:
                _ss["phase"][3]["state"] = "halt"
                _refresh_strip("halted")
                _refresh_header("HALTED", elapsed=time.monotonic() - run_started)
                _render_pipeline_error("filter the results", e)
            _ms = round((time.monotonic() - _t0) * 1000)
    _kept = len(_ss["filtered"])
    if len(_ss["filtered"]) > max_scrape:
        _ss["filtered"] = _ss["filtered"][:max_scrape]
    _phase_done(3, _ms)
    _discarded = len(_ss["results"]) - _kept
    _log_add(
        f"LLM filter kept {mono(_kept)} of {len(_ss['results'])} hits · "
        f"{_discarded} discarded as noise · {_ms} ms."
    )
    _refresh_launch_meta(phase_label="04 FILTER", filtered=str(_kept), discarded=str(_discarded))

    # Stage 5 - Scrape content
    _phase_start(4)
    _refresh_header("RUNNING", elapsed=time.monotonic() - run_started)
    with status_slot.container():
        with st.spinner("SCRAPING PAGES OVER TOR…"):
            _t0 = time.monotonic()
            _ss["scraped"] = cached_scrape_multiple(_ss["filtered"], threads)
            _ms = round((time.monotonic() - _t0) * 1000)
    _phase_done(4, _ms)
    _scraped_n = len(_ss["scraped"])
    _log_add(
        f"Scraped {mono(_scraped_n)} of {len(_ss['filtered'])} kept pages over Tor · "
        f"{len(_ss['filtered']) - _scraped_n} unreachable · {_ms} ms."
    )
    _refresh_launch_meta(
        phase_label="05 SCRAPE",
        pages=f"{_scraped_n} / {len(_ss['filtered'])}",
        filtered=str(_kept),
        discarded=str(_discarded),
    )
    sources_slot.markdown(
        sources_html(_ss["filtered"], f"{len(_ss['filtered'])} OF {_kept} KEPT · {len(_ss['results'])} HITS"),
        unsafe_allow_html=True,
    )

    # Stage 6 - Summarize (streaming)
    _phase_start(5)
    _refresh_header("RUNNING", elapsed=time.monotonic() - run_started)
    _ss["streamed_summary"] = ""
    _log_add("Synthesising findings · report streams below →")
    _refresh_report_meta("STREAMING")

    def ui_emit(chunk: str):
        _ss["streamed_summary"] += chunk
        summary_slot.markdown(_ss["streamed_summary"])

    with status_slot.container():
        with st.spinner("GENERATING SUMMARY…"):
            _t0 = time.monotonic()
            try:
                stream_handler = BufferedStreamingHandler(ui_callback=ui_emit)
                llm.callbacks = [stream_handler]
                _ = generate_summary(
                    llm, query, _ss["scraped"],
                    preset=selected_preset, custom_instructions=custom_instructions,
                )
            except Exception as e:
                _ss["phase"][5]["state"] = "halt"
                _refresh_strip("halted")
                _refresh_header("HALTED", elapsed=time.monotonic() - run_started)
                _render_pipeline_error("generate the summary", e)
            _ms = round((time.monotonic() - _t0) * 1000)
    _phase_done(5, _ms)

    # Save investigation
    _fname = save_investigation(
        query=query,
        refined_query=_ss["refined"],
        model=model,
        preset_label=selected_preset_label,
        sources=_ss["filtered"],
        summary=_ss["streamed_summary"],
    )

    _total_ms = round((time.monotonic() - run_started) * 1000)
    _log_add(f"Investigation saved as {mono(_fname)}.")
    _refresh_strip("complete", total_ms=_total_ms)
    _refresh_header("COMPLETE", elapsed=time.monotonic() - run_started)
    _refresh_report_meta(f"COMPLETE · {len(_ss['streamed_summary']):,} CHARS")
    _refresh_launch_meta(
        phase_label="06 / 06 COMPLETE",
        pages=f"{_scraped_n} / {len(_ss['filtered'])}",
        filtered=str(_kept),
        discarded=str(_discarded),
    )

    with notes_slot.container():
        with st.expander("Run notes"):
            st.markdown(f"**Refined query:** `{_ss['refined']}`")
            st.markdown(f"**Model:** `{model}` &nbsp;&nbsp; **Domain:** {selected_preset_label}")
            st.markdown(
                f"**Results found:** {len(_ss['results'])} &nbsp;&nbsp; "
                f"**Filtered to:** {len(_ss['filtered'])} &nbsp;&nbsp; "
                f"**Scraped:** {len(_ss['scraped'])}"
            )

    inv_record = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "refined_query": _ss["refined"],
        "model": model,
        "preset": selected_preset_label,
        "sources": _ss["filtered"],
        "summary": _ss["streamed_summary"],
        "source_file": _fname,
    }
    with dl_slot.container():
        try:
            st.markdown(report_downloads_html(inv_record, with_markdown=True), unsafe_allow_html=True)
        except Exception as e:
            _log_add(f"Report export unavailable · {esc(str(e))}")
            now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            b64 = base64.b64encode(_ss["streamed_summary"].encode()).decode()
            st.markdown(
                f'<a class="dl" href="data:file/markdown;base64,{b64}" download="summary_{now}.md">Summary · Markdown</a>',
                unsafe_allow_html=True,
            )
    footer_slot.markdown(
        footer_html(
            "Investigation saved to disk · report downloads as PDF / JSON / Markdown",
            f"{esc(_fname)} · {esc(fmt_ms(_total_ms))} · PHASE 06 / 06",
        ),
        unsafe_allow_html=True,
    )
