#!/usr/bin/env python3
"""
Robin CLI — run the dark web OSINT pipeline from the command line.
Usage: python cli.py "your search query" [options]
"""

import argparse
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path

from scrape import scrape_multiple
from search import get_search_results
from llm_utils import get_model_choices
from llm import get_llm, refine_query, filter_results, generate_summary, PRESET_PROMPTS


PRESET_CHOICES = list(PRESET_PROMPTS.keys())


def _log(msg: str) -> None:
    """Print progress to stderr so stdout can be piped to a file."""
    print(msg, file=sys.stderr, flush=True)


def _safe_filename(name: str, max_len: int = 60) -> str:
    """Make a string safe for use as a filename."""
    s = re.sub(r"[^\w\s\-.]", "", name).strip()
    s = re.sub(r"[-\s]+", "_", s)
    return s[:max_len] or "page"


def _write_output_folder(
    out_dir: Path,
    query: str,
    refined: str,
    results: list,
    filtered: list,
    scraped: dict,
    summary: str,
) -> None:
    """Write all pipeline results into the output folder (JSON, TXT, CSV for each)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Refined query: json, txt
    (out_dir / "refined_query.json").write_text(
        json.dumps({"refined_query": refined, "original_query": query}, indent=2), encoding="utf-8"
    )
    (out_dir / "refined_query.txt").write_text(refined, encoding="utf-8")

    # Summary: json, txt (summary.md kept for readability)
    (out_dir / "summary.json").write_text(
        json.dumps({"summary": summary, "query": query, "refined_query": refined}, indent=2), encoding="utf-8"
    )
    (out_dir / "summary.md").write_text(summary, encoding="utf-8")
    (out_dir / "summary.txt").write_text(summary, encoding="utf-8")

    def _write_result_set(name: str, data: list) -> None:
        keys = ("link", "title")
        (out_dir / f"{name}.json").write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        with (out_dir / f"{name}.txt").open("w", encoding="utf-8") as f:
            for r in data:
                f.write(f"{r.get('link', '')}\t{r.get('title', '')}\n")
        with (out_dir / f"{name}.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(data)

    _write_result_set("search_results", results)
    _write_result_set("filtered_results", filtered)

    # Scraped: per-page txt, combined txt/json/csv
    scraped_dir = out_dir / "scraped"
    scraped_dir.mkdir(exist_ok=True)
    scraped_list = []
    for url, content in scraped.items():
        safe = _safe_filename(url.replace(".onion", "").replace("http://", "").replace("https://", ""))
        (scraped_dir / f"{safe}.txt").write_text(content, encoding="utf-8")
        scraped_list.append({"url": url, "content": content})

    (out_dir / "scraped_combined.json").write_text(
        json.dumps({"pages": scraped_list}, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (out_dir / "scraped_combined.txt").open("w", encoding="utf-8") as f:
        for url, content in scraped.items():
            f.write(f"\n{'='*60}\n{url}\n{'='*60}\n\n{content}\n")
    with (out_dir / "scraped_combined.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["url", "content"])
        w.writeheader()
        w.writerows(scraped_list)


def _run_pipeline(
    query: str,
    model: str,
    threads: int,
    preset: str,
    custom_instructions: str,
    output_path: str | None,
) -> int:
    try:
        _log("Loading LLM...")
        llm = get_llm(model)

        _log("Refining query...")
        refined = refine_query(llm, query)
        _log(f"Refined query: {refined}")

        _log("Searching dark web...")
        results = get_search_results(refined.replace(" ", "+"), max_workers=threads)
        _log(f"Search results: {len(results)}")

        _log("Filtering results...")
        filtered = filter_results(llm, refined, results)
        _log(f"Filtered results: {len(filtered)}")

        _log("Scraping content...")
        scraped = scrape_multiple(filtered, max_workers=threads)

        _log("Generating summary...")
        summary = generate_summary(
            llm, query, scraped, preset=preset, custom_instructions=custom_instructions or ""
        )

        # Always write all results to a folder
        if output_path:
            out_dir = Path(output_path)
        else:
            stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            out_dir = Path(f"robin_output_{stamp}")
        out_dir = Path(out_dir)
        _write_output_folder(out_dir, query, refined, results, filtered, scraped, summary)
        _log(f"All results saved to folder: {out_dir.resolve()}")
        if not output_path:
            print(summary, flush=True)

        _log("Pipeline completed successfully.")
        return 0

    except Exception as e:
        _log(f"Error: {e}")
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Robin CLI — AI-powered dark web OSINT. Run a search and get an LLM summary.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py "ransomware LockBit"
  python cli.py "data breach 2024" --model gpt4o --threads 8
  python cli.py "bitcoin mixer" --preset threat_intel -o ./results/run1
  python cli.py "leaked credentials" --instructions "Focus on .gov domains"
        """,
    )
    parser.add_argument(
        "query",
        type=str,
        nargs="?",
        default=None,
        help="Dark web search query (or use -q/--query)",
    )
    parser.add_argument(
        "-q", "--query",
        type=str,
        dest="query_opt",
        default=None,
        help="Dark web search query (alternative to positional)",
    )
    model_choices = get_model_choices()
    default_model = (
        next((m for m in model_choices if m.lower() == "gpt4o"), model_choices[0] if model_choices else "gpt4o")
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=default_model,
        help=f"LLM model (default: {default_model}). Available: %(default)s and others from config.",
    )
    parser.add_argument(
        "--threads", "-t",
        type=int,
        default=4,
        metavar="N",
        help="Scraping threads (default: 4)",
    )
    parser.add_argument(
        "--preset", "-p",
        type=str,
        choices=PRESET_CHOICES,
        default="threat_intel",
        help="Research domain / preset prompt (default: threat_intel)",
    )
    parser.add_argument(
        "--instructions", "-i",
        type=str,
        default="",
        help="Optional custom instructions for the summary",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        metavar="DIR",
        help="Output folder for all results (default: robin_output_YYYY-MM-DD_HH-MM-SS in cwd)",
    )

    args = parser.parse_args()

    query = args.query_opt or args.query
    if not query or not query.strip():
        parser.error("Provide a search query (positional or -q/--query).")

    if not model_choices:
        _log("No LLM models available. Set API keys or local providers in .env and try again.")
        return 1

    if args.model not in model_choices and args.model.lower() not in [m.lower() for m in model_choices]:
        _log(f"Unknown model: {args.model}. Available: {', '.join(model_choices)}")
        return 1

    # Resolve case-insensitive model name
    resolved_model = next((m for m in model_choices if m.lower() == args.model.lower()), args.model)

    return _run_pipeline(
        query=query.strip(),
        model=resolved_model,
        threads=args.threads,
        preset=args.preset,
        custom_instructions=args.instructions,
        output_path=args.output,
    )


if __name__ == "__main__":
    sys.exit(main())
