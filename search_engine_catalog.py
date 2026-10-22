"""
Dark web search engines for Robin (ONLINE entries from deepdarkCTI).

Upstream list: search_engines.cti.md
https://github.com/fastfire/deepdarkCTI/blob/main/search_engines.md
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse

CTI_SOURCE = "https://github.com/fastfire/deepdarkCTI/blob/main/search_engines.md"
_CTI_FILE = Path(__file__).resolve().parent / "search_engines.cti.md"
_ROW_RE = re.compile(r"\|\[([^\]]+)\]\(([^)]+)\)\|(ONLINE|OFFLINE)\|")

_HOST_SEARCH_PATH: dict[str, str] = {
    "juhanurmihxlp77nkq76byazcldy2hlmovfu2epvl5ankdibsot4csyd.onion": "/search/?q={query}",
    "3bbad7fauom4d6sgppalyqddsqbf5u5p56b5k5uk2zxsy3d6ey2jobad.onion": "/search?q={query}",
    "iy3544gmoeclh5de6gez2256v6pjh4omhpqdh2wpeeppjtvqmjhkfwad.onion": "/torgle/?query={query}",
    "amnesia7u5odx5xbwtpnqk3edybgud5bmiagu75bnqx2crntw5kry7ad.onion": "/search?query={query}",
    "findtorroveq5wdnipkaojfpqulxnkhblymc7aramjzajcvpptd4rjqd.onion": "/search?q={query}",
    "3fzh7yuupdfyjhwt3ugzqqof6ulbcl27ecev33knxe3u7goi3vfn2qqd.onion": "/oss/index.php?search={query}",
    "searchgf7gdtauh7bhnbyed4ivxqmuoat3nm6zfrg3ymkq6mtnpye3ad.onion": "/search?q={query}",
    "tor66sewebgixwhcqfnp5inzp5x5uohhdy3kvtnyfxc2e5mxiuh34iid.onion": "/search?q={query}",
    "wbr4bzzxbeidc6dwcqgwr3b6jl7ewtykooddsc5ztev3t3otnl45khyd.onion": "/evo/search.php?q={query}",
    "bobby64o755x3gsuznts6hf6agxqjcz5bop6hs7ejorekbm7omes34ad.onion": "/index.php?q={query}",
    "uzowkytjk4da724giztttfly4rugfnbqkexecotfp5wjc2uhpykrpryd.onion": "/search/?q={query}",
    "xcprh4cjas33jnxgs3zhakof6mctilfxigwjcsevdfap7vtyj57lmjad.onion": "/tgs/?q={query}",
    "torchqsxkllrj2eqaitp5xvcgfeg3g5dr3hr2wnuvnj76bbxkxfiwxqd.onion": "/search?query={query}",
    "z5vawdol25vrmorm4yydmohsd4u6rdoj2sylvoi3e3nqvxkvpqul7bqd.onion": "/search?q={query}",
    "metagerv65pwclop2rsfzg4jwowpavpwd6grhhlvdgsswvo6ii4akgyd.onion": "/meta/search?q={query}",
    "haystak5njsmn2hqkewecpaxetahtwhsbsa64jom2k22z5afxhnpxfid.onion": "/?q={query}",
    "tordexu73joywapk2txdr54jed4imqledpcvcuf75qsas2gwdgksvnyd.onion": "/search?query={query}",
    "search7tdrcvri22rieiwgi5g46qnwsesvnubqav2xakhezv4hjzkkad.onion": "/search?q={query}",
    "searchpxsd4vdpf35uk4ycgxolp732zhs7zr4qgftt6qvmgpo6mukbyd.onion": "/search?q={query}",
    "on62jjkocppf3alrznspngqt4v7emcyxcxz4r5cq5pwnajyshr2u4uqd.onion": "/onionbag/search?q={query}",
}


def _host_from_url(url: str) -> str:
    return (urlparse(url.strip()).hostname or "").lower()


def _search_url_template(onion_url: str) -> str:
    host = _host_from_url(onion_url)
    if not host:
        raise ValueError(f"invalid onion url: {onion_url}")
    if host in _HOST_SEARCH_PATH:
        return f"http://{host}{_HOST_SEARCH_PATH[host]}"

    path = (urlparse(onion_url.strip()).path or "/").rstrip("/")
    if path.endswith("/search.php"):
        return f"http://{host}{path}?q={{query}}"
    if path.endswith("/index.php"):
        return f"http://{host}{path}?q={{query}}"
    if path.endswith("/search"):
        return f"http://{host}{path}?q={{query}}"
    if path.endswith("/search/"):
        return f"http://{host}{path}?q={{query}}"
    if path.endswith("/tgs"):
        return f"http://{host}{path}/?q={{query}}"
    if path.endswith("/tgs/"):
        return f"http://{host}{path}?q={{query}}"
    if path.endswith("/onionbag"):
        return f"http://{host}{path}/search?q={{query}}"
    if path.endswith("/onionbag/"):
        return f"http://{host}{path}search?q={{query}}"
    if path.endswith("/torgle"):
        return f"http://{host}{path}/?query={{query}}"
    if path.endswith("/torgle/"):
        return f"http://{host}{path}?query={{query}}"
    if path.endswith("/oss"):
        return f"http://{host}{path}/index.php?search={{query}}"
    if path.endswith("/oss/"):
        return f"http://{host}{path}index.php?search={{query}}"
    return f"http://{host}/search?q={{query}}"


def _parse_cti_markdown(text: str) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    seen_hosts: set[str] = set()
    for name, url, status in _ROW_RE.findall(text):
        if status != "ONLINE":
            continue
        if url.startswith("https://"):
            continue
        host = _host_from_url(url)
        if not host or host in seen_hosts:
            continue
        seen_hosts.add(host)
        rows.append((name.strip(), url.strip()))
    return rows


def build_search_engines() -> list[dict[str, str]]:
    text = _CTI_FILE.read_text(encoding="utf-8")
    return [
        {"name": name, "url": _search_url_template(url)}
        for name, url in _parse_cti_markdown(text)
    ]


SEARCH_ENGINES: list[dict[str, str]] = build_search_engines()
