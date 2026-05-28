"""Scraper for ai-2027.com content."""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from src.config import RAW_DIR, SOURCE_URL

logger = logging.getLogger(__name__)


def get_page_links(base_url: str) -> list[str]:
    """Discover all content pages from the site."""
    response = requests.get(base_url, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    links = set()
    parsed_base = urlparse(base_url)

    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        full_url = urljoin(base_url, href)
        parsed = urlparse(full_url)

        if parsed.netloc == parsed_base.netloc and not href.startswith("#"):
            clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            links.add(clean_url)

    links.add(base_url)
    return sorted(links)


def scrape_page(url: str) -> dict | None:
    """Scrape a single page and return structured document."""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.warning(f"Failed to fetch {url}: {e}")
        return None

    soup = BeautifulSoup(response.text, "html.parser")

    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""

    meta_author = soup.find("meta", attrs={"name": "author"})
    author = meta_author["content"] if meta_author else None

    meta_date = soup.find("meta", attrs={"name": "publication_date"}) or soup.find(
        "meta", attrs={"property": "article:published_time"}
    )
    published_date = meta_date["content"] if meta_date else None

    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    main_content = soup.find("main") or soup.find("article") or soup.find("body")
    if not main_content:
        return None

    content_html = str(main_content)
    content_text = main_content.get_text(separator="\n", strip=True)

    if len(content_text.strip()) < 50:
        return None

    return {
        "source_url": url,
        "title": title,
        "author": author,
        "published_date": published_date,
        "content_html": content_html,
        "content_text": content_text,
        "ingestion_timestamp": datetime.now(timezone.utc).isoformat(),
    }


def scrape_all(base_url: str = SOURCE_URL, delay: float = 1.0) -> list[dict]:
    """Scrape all pages from the source site."""
    logger.info(f"Discovering pages from {base_url}")
    links = get_page_links(base_url)
    logger.info(f"Found {len(links)} pages to scrape")

    documents = []
    for url in links:
        logger.info(f"Scraping: {url}")
        doc = scrape_page(url)
        if doc:
            documents.append(doc)
        time.sleep(delay)

    logger.info(f"Successfully scraped {len(documents)} documents")
    return documents


def save_raw_documents(documents: list[dict], output_dir: Path = RAW_DIR) -> None:
    """Save scraped documents to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, doc in enumerate(documents):
        filename = output_dir / f"doc_{i:04d}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)

    manifest = {
        "total_documents": len(documents),
        "source": documents[0]["source_url"] if documents else "",
        "scraped_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Saved {len(documents)} documents to {output_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    docs = scrape_all()
    save_raw_documents(docs)
