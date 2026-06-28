"""Download and extract text from SEC EDGAR 10-K / 10-Q filings."""

from __future__ import annotations

import re
import time
import warnings
from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from sec_edgar_downloader import Downloader

from app.config import CORPUS_DIR, CORPUS_COMPANIES, FILING_TYPES, settings

# SEC fair-access policy: max 10 requests/sec; we stay conservative.
_REQUEST_DELAY_SEC = 0.12


warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


def _clean_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _extract_pages(text: str) -> list[tuple[int, str]]:
    """Split on common SEC page-break markers; fall back to single page."""
    markers = list(re.finditer(r"(?:Page\s+(\d+)|<!--\s*Page\s*(\d+)\s*-->)", text, re.I))
    if not markers:
        return [(1, text)]

    pages: list[tuple[int, str]] = []
    for i, match in enumerate(markers):
        page_num = int(match.group(1) or match.group(2))
        start = match.end()
        end = markers[i + 1].start() if i + 1 < len(markers) else len(text)
        page_text = text[start:end].strip()
        if page_text:
            pages.append((page_num, page_text))
    if not pages:
        return [(1, text)]
    return pages


def _section_hint(text: str) -> str:
    """Best-effort Item/Part detection for metadata."""
    head = text[:800]
    for pattern in (
        r"(Item\s+\d+[A-Z]?\.?\s*[^\n]{0,80})",
        r"(PART\s+[IVXLC]+\s*[^\n]{0,60})",
        r"(MANAGEMENT['\u2019]?S DISCUSSION[^\n]{0,40})",
        r"(Risk Factors)",
    ):
        m = re.search(pattern, head, re.I)
        if m:
            return m.group(1).strip()[:120]
    return ""


def download_filings(
    tickers: list[str] | None = None,
    limit_per_type: int = 1,
    download_dir: Path | None = None,
) -> list[Path]:
    """
    Download recent 10-K and 10-Q filings for each ticker.
    Returns paths to extracted plain-text files.
    """
    download_dir = download_dir or CORPUS_DIR
    download_dir.mkdir(parents=True, exist_ok=True)
    tickers = tickers or list(CORPUS_COMPANIES.keys())

    ua = settings.sec_user_agent
    email = ua.split()[-1] if "@" in ua else "contact@example.com"
    company = ua.replace(email, "").strip() or "FinRAGEval"
    dl = Downloader(company, email, str(download_dir / "raw"))

    extracted: list[Path] = []

    for ticker in tickers:
        if ticker not in CORPUS_COMPANIES:
            continue
        for filing_type in FILING_TYPES:
            for attempt in range(3):
                try:
                    dl.get(filing_type, ticker, limit=limit_per_type, download_details=True)
                    break
                except Exception as exc:
                    wait = _REQUEST_DELAY_SEC * (2 ** attempt)
                    print(f"[sec_fetcher] {ticker} {filing_type} attempt {attempt+1} failed: {exc}")
                    time.sleep(wait)
            time.sleep(_REQUEST_DELAY_SEC)

    raw_root = download_dir / "raw" / "sec-edgar-filings"
    if not raw_root.exists():
        return extracted

    for ticker_dir in raw_root.iterdir():
        if not ticker_dir.is_dir():
            continue
        ticker = ticker_dir.name.upper()
        for filing_type_dir in ticker_dir.iterdir():
            if filing_type_dir.name not in FILING_TYPES:
                continue
            for filing_dir in filing_type_dir.iterdir():
                if not filing_dir.is_dir():
                    continue
                primary = _select_primary_document(filing_dir)
                if not primary:
                    print(f"[sec_fetcher] no primary doc for {ticker}/{filing_type_dir.name}/{filing_dir.name}")
                    continue
                try:
                    html = primary.read_text(encoding="utf-8", errors="ignore")
                    text = _clean_text(html)
                    if len(text) < 500:
                        print(f"[sec_fetcher] skipped short doc {primary.name} ({len(text)} chars)")
                        continue
                    filing_date = filing_dir.name
                    out_name = f"{ticker}_{filing_type_dir.name}_{filing_date}.txt"
                    out_path = download_dir / out_name
                    if out_path.exists():
                        continue
                    out_path.write_text(text, encoding="utf-8")
                    extracted.append(out_path)
                    print(f"[sec_fetcher] extracted {out_name} from {primary.name} ({len(text):,} chars)")
                except Exception as exc:
                    print(f"[sec_fetcher] extract failed {primary}: {exc}")
    return extracted


def _is_excluded_filing(path: Path) -> bool:
    """Skip exhibits, XBRL, graphics, and stylesheet assets."""
    name = path.name.lower()
    if path.suffix.lower() == ".xml":
        return True
    if name.endswith(".xsd") or "xsl" in name or "exhibit" in name:
        return True
    if re.match(r"^r\d+\.htm", name):
        return True
    if re.match(r"^ex\d+", name):
        return True
    if "graphic" in name or name in {"summary.html", "report.css"}:
        return True
    return False


def _select_primary_document(filing_dir: Path) -> Optional[Path]:
    """
    Pick the main 10-K/10-Q HTML document — largest text body after cleaning.
    Avoids ingesting every exhibit/R-file separately (common SEC EDGAR pitfall).
    """
    candidates: list[tuple[int, Path]] = []
    for html_path in filing_dir.rglob("*"):
        if html_path.suffix.lower() not in {".htm", ".html"}:
            continue
        if _is_excluded_filing(html_path):
            continue
        try:
            raw = html_path.read_text(encoding="utf-8", errors="ignore")
            text = _clean_text(raw)
            if len(text) >= 500:
                candidates.append((len(text), html_path))
        except Exception:
            continue
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def load_corpus_text_files(corpus_dir: Path | None = None) -> list[dict]:
    """Load all .txt corpus files into document records."""
    corpus_dir = corpus_dir or CORPUS_DIR
    docs: list[dict] = []
    for path in sorted(corpus_dir.glob("*.txt")):
        name = path.stem
        parts = name.split("_")
        ticker = parts[0] if parts else "UNK"
        filing_type = parts[1] if len(parts) > 1 else "FILING"
        filing_date = parts[2] if len(parts) > 2 else "unknown"
        text = path.read_text(encoding="utf-8", errors="ignore")
        pages = _extract_pages(text)
        docs.append({
            "path": str(path),
            "ticker": ticker,
            "filing_type": filing_type,
            "filing_date": filing_date,
            "source_url": f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={CORPUS_COMPANIES.get(ticker, '')}",
            "pages": pages,
            "full_text": text,
        })
    return docs


def fetch_transcript_urls(ticker: str) -> list[str]:
    """Placeholder for optional earnings transcript ingestion via local files."""
    transcript_dir = CORPUS_DIR / "transcripts"
    if not transcript_dir.exists():
        return []
    return [str(p) for p in transcript_dir.glob(f"{ticker}_*.txt")]