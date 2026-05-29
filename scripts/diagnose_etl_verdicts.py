"""
Diagnose ETL verdict distribution by running extract() locally.

Aggregates articles by domain, reports per-domain count and verdict breakdown,
and flags domains classified as "unknown" - those are the candidates for
inclusion in _INSTITUTIONAL_DOMAINS or _FACTCHECKER_DOMAINS.

Fast: does NOT embed nor touch Qdrant. ~30-90s depending on feed latency.

Usage: python -m scripts.diagnose_etl_verdicts
"""
from __future__ import annotations

from collections import Counter, defaultdict
from urllib.parse import urlparse

from services.etl.app.pipeline import extract, _domain_class


def _normalize(url: str) -> str:
    try:
        netloc = urlparse(url).netloc.lower()
    except Exception:
        return "<unparseable>"
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return netloc or "<empty>"


def main() -> None:
    print("Running extract() - this may take 30-90s while RSS feeds are fetched...")
    articles = extract()
    total = len(articles)
    print(f"\nTotal articles extracted: {total}\n")

    by_domain: dict[str, Counter] = defaultdict(Counter)
    overall = Counter()

    for art in articles:
        netloc = _normalize(art.get("url", ""))
        verdict = art.get("verdict", "UNVERIFIED")
        by_domain[netloc][verdict] += 1
        overall[verdict] += 1

    # Per-domain table, sorted by total articles descending
    header = f"{'Domain':<40} {'class':<14} {'REAL':>6} {'FAKE':>6} {'UNV':>6} {'tot':>6}"
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    unknown_domains: list[tuple[str, int]] = []
    for domain in sorted(by_domain, key=lambda d: -sum(by_domain[d].values())):
        c = by_domain[domain]
        dtotal = sum(c.values())
        cls = _domain_class(f"https://{domain}/")
        print(f"{domain:<40} {cls:<14} {c['REAL']:>6} {c['FAKE']:>6} {c['UNVERIFIED']:>6} {dtotal:>6}")
        if cls == "unknown" and dtotal > 0:
            unknown_domains.append((domain, dtotal))

    print("-" * len(header))
    print(
        f"{'TOTAL':<40} {'':<14} "
        f"{overall['REAL']:>6} {overall['FAKE']:>6} {overall['UNVERIFIED']:>6} {total:>6}"
    )

    print()
    if total > 0:
        real_pct = 100 * overall['REAL'] / total
        fake_pct = 100 * overall['FAKE'] / total
        unv_pct = 100 * overall['UNVERIFIED'] / total
        print(f"REAL:       {overall['REAL']:>4} ({real_pct:5.1f}%)")
        print(f"FAKE:       {overall['FAKE']:>4} ({fake_pct:5.1f}%)")
        print(f"UNVERIFIED: {overall['UNVERIFIED']:>4} ({unv_pct:5.1f}%)")

        print()
        print("Success criteria:")
        crit_unv = unv_pct < 50
        crit_total = total >= 200
        print(f"  UNVERIFIED < 50%: {'PASS' if crit_unv else 'FAIL'} ({unv_pct:.1f}%)")
        print(f"  total >= 200:     {'PASS' if crit_total else 'FAIL'} ({total})")
        print()
        if crit_unv and crit_total:
            print("✓ ALL CRITERIA MET")
        else:
            print("✗ Not yet - top unknown-class domains to triage:")
            for d, n in unknown_domains[:15]:
                print(f"    {d:<40} ({n} articles)")

    # Detect dead feeds: any RSS feed URL that didn't contribute any article
    from services.etl.app.pipeline import RSS_FEEDS
    contributing_domains = {d for d in by_domain}
    silent_feeds: list[str] = []
    for feed_url in RSS_FEEDS:
        feed_netloc = _normalize(feed_url)
        if feed_netloc not in contributing_domains:
            silent_feeds.append(feed_url)
    if silent_feeds:
        print()
        print(f"Silent feeds (zero articles contributed - likely dead or malformed):")
        for f in silent_feeds:
            print(f"    {f}")


if __name__ == "__main__":
    main()
