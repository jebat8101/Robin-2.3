#!/usr/bin/env python3
"""
Robin CLI — run the dark web OSINT pipeline from the command line.
Usage: python cli.py "your search query" [options]
"""


import click
import subprocess
import json
from yaspin import yaspin
from datetime import datetime
from scrape import scrape_multiple
from search import get_search_results
from llm import get_llm, refine_query, filter_results, generate_summary
from llm_utils import get_model_choices

MODEL_CHOICES = get_model_choices()


@click.group()
@click.version_option()
def robin():
    """Robin: AI-Powered Dark Web OSINT Tool."""
    pass


@robin.command()
@click.option(
    "--model",
    "-m",
    default="gpt-5-mini",
    show_default=True,
    type=click.Choice(MODEL_CHOICES),
    help="Select LLM model to use (e.g., gpt4o, claude sonnet 3.5, ollama models)",
)
@click.option("--query", "-q", required=True, type=str, help="Dark web search query")
@click.option(
    "--threads",
    "-t",
    default=5,
    show_default=True,
    type=int,
    help="Number of threads to use for scraping (Default: 5)",
)
@click.option(
    "--output",
    "-o",
    type=str,
    help="Filename to save the final intelligence summary. If not provided, a filename based on the current date and time is used.",
)
def cli(model, query, threads, output):
    """Run Robin in CLI mode.\n
    Example commands:\n
    - robin -m gpt4o -q "ransomware payments" -t 12\n
    - robin --model claude-3-5-sonnet-latest --query "sensitive credentials exposure" --threads 8 --output filename\n
    - robin -m llama3.1 -q "zero days"\n
    """
    llm = get_llm(model)

    # Show spinner while processing the query
    with yaspin(text="Processing...", color="cyan") as sp:
        refined_query = refine_query(llm, query)

        search_results = get_search_results(
            refined_query.replace(" ", "+"), max_workers=threads
        )

        search_filtered = filter_results(llm, refined_query, search_results)

        scraped_results = scrape_multiple(search_filtered, max_workers=threads)
        sp.ok("✔")

    # Generate the intelligence summary.
    summary = generate_summary(llm, query, scraped_results)

    # Save or print the summary
    if not output:
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"summary_{now}.md"
    else:
        filename = output + ".md"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(summary)
        click.echo(f"\n\n[OUTPUT] Final intelligence summary saved to {filename}")


@robin.command()
@click.option(
    "--model",
    "-m",
    default="gpt-5-mini",
    show_default=True,
    type=click.Choice(MODEL_CHOICES),
    help="Select LLM model to refine the query before searching",
)
@click.option(
    "--query",
    "-q",
    required=True,
    type=str,
    help="Dark web search query (will be refined by LLM before searching)",
)
@click.option(
    "--threads",
    "-t",
    default=5,
    show_default=True,
    type=int,
    help="Number of threads to use for scraping (Default: 5)",
)
@click.option(
    "--output",
    "-o",
    type=str,
    help="Filename to save raw search + scraped data as JSON. "
    "If not provided, a filename based on the current date and time is used.",
)
def export_json(model, query, threads, output):
    """
    Run the search + scraper backend pipeline and save raw data to a JSON file.

    This uses an LLM ONLY to refine/enrich your query, then:
      1) searches the dark web with the refined query
      2) scrapes the found URLs
      3) saves the combined data to a JSON file
    """
    with yaspin(text="Loading LLM and refining query...", color="cyan") as sp:
        # Load LLM and refine the user query first
        llm = get_llm(model)
        refined_query = refine_query(llm, query)

        sp.text = "Searching dark web with refined query..."
        search_results = get_search_results(
            refined_query.replace(" ", "+"),
            max_workers=threads,
        )

        sp.text = f"Scraping {len(search_results)} results..."
        scraped_results = scrape_multiple(search_results, max_workers=threads)
        sp.ok("✔")

    # Use a single timestamp for this scan so all entries are comparable
    scan_time = datetime.utcnow()
    scan_time_iso = scan_time.isoformat() + "Z"

    # Decide output filename (based on provided name or scan time)
    if not output:
        filename = f"backend_data_{scan_time.strftime('%Y-%m-%d_%H-%M-%S')}.json"
    else:
        filename = output if output.lower().endswith(".json") else output + ".json"

    # Load previous last_seen_detail per link if file already exists
    old_last_seen = {}
    try:
        with open(filename, "r", encoding="utf-8") as f:
            prev_data = json.load(f)
        if isinstance(prev_data, list):
            for entry in prev_data:
                link = entry.get("link")
                ts = entry.get("last_seen_detail")
                if link and ts:
                    old_last_seen[link] = ts
    except FileNotFoundError:
        pass
    except Exception:
        # Ignore parse errors; start fresh
        pass

    def _parse_iso_z(ts: str):
        if not ts:
            return None
        if isinstance(ts, str) and ts.endswith("Z"):
            ts_clean = ts[:-1]
        else:
            ts_clean = ts
        try:
            return datetime.fromisoformat(ts_clean)
        except Exception:
            return None

    def _format_delta(delta):
        seconds = int(delta.total_seconds())
        if seconds < 60:
            return "1 second ago" if seconds == 1 else f"{seconds} seconds ago"
        minutes = seconds // 60
        if minutes < 60:
            return "1 minute ago" if minutes == 1 else f"{minutes} minutes ago"
        hours = minutes // 60
        if hours < 24:
            return "1 hour ago" if hours == 1 else f"{hours} hours ago"
        days = hours // 24
        if days < 30:
            return "1 day ago" if days == 1 else f"{days} days ago"
        months = days // 30
        if months < 12:
            return "1 month ago" if months == 1 else f"{months} months ago"
        years = months // 12
        return "1 year ago" if years == 1 else f"{years} years ago"

    # Combine search metadata with scraped content and availability info
    combined = []
    total_results = len(search_results)
    active_count = 0
    seen_before_inactive = 0
    never_seen_active = 0

    for idx, item in enumerate(search_results, start=1):
        link = item.get("link")
        content = scraped_results.get(link, "")

        # Heuristic for availability:
        # - On successful scrape, scrape_single returns "title - full_text"
        # - On failure/non-200, it returns just "title"
        # So if content == title, we treat the link as inactive/unreachable.
        title = item.get("title")
        is_active = bool(content) and content != title

        if is_active:
            active_count += 1
            # Seen as active in this scan
            last_seen_detail = scan_time_iso
            last_seen = "just now"
        else:
            # Not reachable now; fall back to last active timestamp (if we have one)
            prev_ts = old_last_seen.get(link)
            prev_dt = _parse_iso_z(prev_ts) if prev_ts else None
            if prev_dt:
                delta = scan_time - prev_dt
                last_seen_detail = prev_ts
                last_seen = _format_delta(delta)
                seen_before_inactive += 1
            else:
                last_seen_detail = None
                last_seen = "never seen active"
                never_seen_active += 1

        # Final dark-web–focused schema for indexing (no Twitter fields)
        entry = {
            "keyword": refined_query,
            "scan_time": scan_time_iso,
            "id": idx,
            "title": title,
            "url": link,
            "content": content,
            "is_active": is_active,
            "last_seen": last_seen,  # human-readable, e.g. "3 minutes ago"
            "last_seen_detail": last_seen_detail,  # ISO timestamp or null
        }

        combined.append(entry)

    # Write compact JSON (no pretty-print / indent), e.g.:
    # [{"address": "...", "description": "...", "last_seen": "..."}]
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False)

    # Console log summary so the user can see what happened
    click.echo("\n[LOG] Export completed.")
    click.echo(f"[LOG]   Output file          : {filename}")
    click.echo(f"[LOG]   Raw query            : {query}")
    click.echo(f"[LOG]   Refined query        : {refined_query}")
    click.echo(f"[LOG]   Total results        : {total_results}")
    click.echo(f"[LOG]   Currently active     : {active_count}")
    click.echo(f"[LOG]   Inactive (seen before): {seen_before_inactive}")
    click.echo(f"[LOG]   Never seen active    : {never_seen_active}")

    click.echo(f"\n[OUTPUT] Backend data (search + scraped) saved to {filename}")


@robin.command()
@click.option(
    "--ui-port",
    default=8501,
    show_default=True,
    type=int,
    help="Port for the Streamlit UI",
)
@click.option(
    "--ui-host",
    default="localhost",
    show_default=True,
    type=str,
    help="Host for the Streamlit UI",
)
def ui(ui_port, ui_host):
    """Run Robin in Web UI mode."""
    import sys, os

    # Use streamlit's internet CLI entrypoint
    from streamlit.web import cli as stcli

    # When PyInstaller one-file, data files livei n _MEIPASS
    if getattr(sys, "frozen", False):
        base = sys._MEIPASS
    else:
        base = os.path.dirname(__file__)

    ui_script = os.path.join(base, "ui.py")
    # Build sys.argv
    sys.argv = [
        "streamlit",
        "run",
        ui_script,
        f"--server.port={ui_port}",
        f"--server.address={ui_host}",
        "--global.developmentMode=false",
    ]
    # This will never return until streamlit exits
    sys.exit(stcli.main())


if __name__ == "__main__":
    robin()
