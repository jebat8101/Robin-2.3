# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

OSINT investigators, threat-intelligence analysts, security researchers, and students who research dark-web sources as part of lawful investigations, reporting, or coursework. The README frames the tool as "educational and lawful investigative purposes"; the redesign brief confirmed this audience. They work at a single workstation in long, focused sessions (often outside business hours), and need to turn a rough lead — one or two keywords — into a defensible, source-backed picture without manually searching sixteen engines themselves.

## Product Purpose

Robin is an AI-powered dark-web OSINT tool: it refines an investigator's query with an LLM, searches 16 dark-web search engines over Tor, filters the raw hits, scrapes the survivors, and streams a synthesized investigation summary — then saves the whole investigation to disk for reporting. Success means an investigator reaches credible, source-linked findings in one sitting, with a record to back them.

## Positioning

The refine → search → filter → scrape → summarize pipeline is provider-agnostic: the identical investigation runs on a cloud model or a fully local one (Ollama / llama.cpp), and the summary never replaces its sources — findings stay linked to the pages they came from. A generic search UI could not truthfully copy the local-model path or the saved-investigation record.

## Operating Context

Requires a running Tor daemon (SOCKS5 at 127.0.0.1:9050) for all searching and scraping. LLM access comes from `.env` keys (OpenAI, Anthropic, Google, OpenRouter) or local servers (Ollama, llama.cpp, base URLs configurable). Runs via Docker (recommended upstream) or `streamlit run ui.py` on Python 3.10+; in this workspace it runs headless on 127.0.0.1:8501 from the project venv. Investigations persist as JSON in `investigations/` and reload from the sidebar. It is a local, single-user tool — no auth, no multi-user layer, no telemetry of its own.

## Capabilities and Constraints

Functionality the redesign must preserve exactly (user-confirmed invariant — behavior identical): the six-stage pipeline (load LLM → refine query → search → filter → scrape → stream summary); model selection gated by which provider keys are present, plus auto-detected local Ollama/llama.cpp models; research-domain presets (threat intel, ransomware, personal identity, corporate espionage) with optional custom instructions; the scraping-threads / max-results-to-filter / max-pages-to-scrape sliders; LLM, Tor, and per-engine health checks with latency; save / load / clear of past investigations; Markdown download of the summary. Framework constraint: Streamlit — structure and styling are limited to Streamlit theming, injected CSS, and the layout constructs Streamlit offers. The UI must not require new services, keys, or network calls of its own.

## Brand Commitments

The name "Robin", the tagline concept ("AI-Powered Dark Web OSINT Tool"), and the existing logo asset `.github/assets/robin_logo.png` (purple wizard-hat mark) are kept as-is (user-confirmed). Author credit to Apurv Singh Gautam stays visible. The incumbent black Streamlit look is explicitly NOT a commitment — replacing it is the point of this work (user chose redesign in the "Ops console" lane).

## Evidence on Hand

`.github/assets/` carries the logo (`robin_logo.png`, `logo.png`), a demo GIF, and screenshots; `README.md` documents architecture, Docker usage, and the workflow diagram. A real saved investigation exists at `investigations/investigation_20260503_224415.json` (keys: `timestamp`, `query`, `refined_query`, `model`, `preset`, `sources[]`, `summary`) and is legitimate design material — its query was "malaysia". No testimonials, benchmarks, user counts, or press exist, and none may be invented.

## Product Principles

1. **Operational awareness first.** Before a query runs, the analyst must see the tool's real state: Tor reachability, which providers are configured, which model is selected, engine health. (User-confirmed success criterion for this redesign.)
2. **Evidence over spectacle.** Sources are first-class objects; the summary is presented as a findings report, never as an oracle.
3. **Honest about the wire.** A query and scraped page content travel to third-party LLM providers unless a local model is chosen; the UI should make that fact and the current model choice visible rather than hiding it. [inferred design commitment — not yet product-confirmed]
4. **Calm in long sessions.** Dark, low-fatigue surfaces, restrained motion, no gamification, no engagement mechanics. Investigators sit in this tool for hours.
5. **A record, not a search box.** Every run is saveable, reloadable, and exportable; the archive of investigations is part of the tool's value.

## Accessibility & Inclusion

No formal standard was established. Working baseline: WCAG-AA-minded contrast on the dark theme, status never carried by color or emoji alone (every state gets a text label), full keyboard operability through Streamlit's native controls. [inferred — defaulting to maximum user protection]
