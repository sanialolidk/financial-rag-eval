"""Download and extract text from SEC EDGAR 10-K / 10-Q filings."""

from __future__ import annotations

import re
import time
import warnings
from pathlib import Path

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
            try:
                dl.get(filing_type, ticker, limit=limit_per_type, download_details=True)
                time.sleep(_REQUEST_DELAY_SEC)
            except Exception as exc:
                print(f"[sec_fetcher] {ticker} {filing_type} download failed: {exc}")

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
                for html_path in filing_dir.rglob("*.htm*"):
                    try:
                        html = html_path.read_text(encoding="utf-8", errors="ignore")
                        text = _clean_text(html)
                        if len(text) < 500:
                            continue
                        filing_date = filing_dir.name
                        out_name = f"{ticker}_{filing_type_dir.name}_{filing_date}.txt"
                        out_path = download_dir / out_name
                        out_path.write_text(text, encoding="utf-8")
                        extracted.append(out_path)
                    except Exception as exc:
                        print(f"[sec_fetcher] extract failed {html_path}: {exc}")
    return extracted


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