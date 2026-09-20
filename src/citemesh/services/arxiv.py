"""Bibliography and metadata reads for the seed-only arXiv fallback."""

from __future__ import annotations

import logging
import re
import time
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from urllib.parse import unquote, urlparse

import requests

from citemesh.core import Author, Paper
from citemesh.core.paper_ids import strip_arxiv_version

logger = logging.getLogger(__name__)
_ARXIV_TOKEN = r"(?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[a-z-]+)?/\d{7})(?:v\d+)?"
_ARXIV_TEXT = re.compile(r"arxiv\s*:\s*(" + _ARXIV_TOKEN + r")\b", re.I)
_ARXIV_LEGACY_TEXT = re.compile(r"\b([a-z-]+(?:\.[a-z-]+)?/\d{7}(?:v\d+)?)\b", re.I)
_ARXIV_URL_TEXT = re.compile(
    r"https?://(?:www\.)?arxiv\.org/(?:abs|pdf|html)/[^\s<>\"]+", re.I
)
_DOI = re.compile(r"\b10\.\d{4,9}/[-._;()/:a-z0-9]+", re.I)
_VOID_TAGS = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
}
_ATOM = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


def arxiv_identifier(value: str) -> str | None:
    """Recognize an explicit arXiv ID or URL, retaining its version.

    :param str value: Paper identifier, arXiv field, or arXiv URL.
    :return str | None: Valid suffix suitable for an arXiv request.
    """
    candidate = value.strip()
    if candidate.lower().startswith("arxiv:"):
        candidate = candidate.split(":", 1)[1].strip()
    elif "://" in candidate:
        parsed = urlparse(candidate)
        if (parsed.hostname or "").lower() not in {
            "arxiv.org",
            "www.arxiv.org",
            "export.arxiv.org",
        }:
            return None
        match = re.fullmatch(r"/(?:abs|pdf|html)/(.+)", parsed.path)
        if not match:
            return None
        candidate = unquote(match[1]).removesuffix(".pdf")
    return candidate if re.fullmatch(_ARXIV_TOKEN, candidate, re.I) else None


def _identifiers(text: str, links: list[str]) -> tuple[str, ...]:
    """Extract explicit identifiers from one bibliography entry.

    :param str text: Visible entry text.
    :param list[str] links: Entry hyperlink targets.
    :return tuple[str, ...]: Unique canonical aliases, arXiv before DOI.
    """
    arxiv_ids = [match[1] for match in _ARXIV_TEXT.finditer(text)]
    arxiv_ids.extend(match[1] for match in _ARXIV_LEGACY_TEXT.finditer(text))
    for match in _ARXIV_URL_TEXT.finditer(text):
        identifier = arxiv_identifier(match[0].rstrip(".,;:)]"))
        if identifier:
            arxiv_ids.append(identifier)
    for link in links:
        identifier = arxiv_identifier(link)
        if identifier:
            arxiv_ids.append(identifier)
    identifiers = [f"arxiv:{strip_arxiv_version(value).lower()}" for value in arxiv_ids]
    for source in [text, *(unquote(link) for link in links)]:
        for match in _DOI.finditer(source):
            doi = match[0].rstrip(".,;:")
            # Sentence punctuation is not part of a DOI; balanced suffixes can be.
            while doi.endswith(")") and doi.count(")") > doi.count("("):
                doi = doi[:-1]
            identifiers.append(doi.lower())
    return tuple(dict.fromkeys(identifiers))


class _BibliographyParser(HTMLParser):
    """Read LaTeXML bibliography entries without including article prose."""

    def __init__(self) -> None:
        """Initialize entry and element state."""
        super().__init__(convert_charrefs=True)
        self.entries: list[tuple[str, ...]] = []
        self.stack: list[str] = []
        self.bibliography_depth: int | None = None
        self.entry_depth: int | None = None
        self.text: list[str] = []
        self.links: list[str] = []

    def _finish_entry(self) -> None:
        """Store the active bibliography entry when its boundary is reached.

        :return None: Appends the active entry and clears its state.
        """
        if self.entry_depth is not None:
            self.entries.append(_identifiers("".join(self.text), self.links))
            self.entry_depth = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Track bibliography scope and capture entry links.

        :param str tag: Element name.
        :param list[tuple[str, str | None]] attrs: Element attributes.
        :return None: Updates parser state.
        """
        attributes = dict(attrs)
        classes = (attributes.get("class") or "").split()
        if tag not in _VOID_TAGS:
            self.stack.append(tag)
        if (
            "ltx_bibliography" in classes
            or attributes.get("role") == "doc-bibliography"
        ):
            self.bibliography_depth = len(self.stack)
        if self.bibliography_depth is not None and (
            "ltx_bibitem" in classes or attributes.get("role") == "doc-biblioentry"
        ):
            # LaTeXML occasionally leaves a <li> implicitly closed before the
            # next bibliography item begins.
            self._finish_entry()
            self.entry_depth = len(self.stack)
            self.text = []
            self.links = []
        if self.entry_depth is not None and tag == "a" and attributes.get("href"):
            self.links.append(attributes["href"] or "")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Handle self-closing elements without leaving stack entries.

        :param str tag: Element name.
        :param list[tuple[str, str | None]] attrs: Element attributes.
        :return None: Updates parser state.
        """
        self.handle_starttag(tag, attrs)
        if tag not in _VOID_TAGS:
            self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        """Finish bibliography entries at their containing element boundary.

        :param str tag: Closing element name.
        :return None: Updates parser state and extracted entries.
        """
        if tag not in self.stack:
            return
        depth = len(self.stack) - self.stack[::-1].index(tag)
        if self.entry_depth is not None and depth <= self.entry_depth:
            self._finish_entry()
        if self.bibliography_depth is not None and depth <= self.bibliography_depth:
            self.bibliography_depth = None
        del self.stack[depth - 1 :]

    def handle_data(self, data: str) -> None:
        """Collect visible entry text only.

        :param str data: Text node.
        :return None: Appends text inside bibliography entries.
        """
        if self.entry_depth is not None:
            self.text.append(data)


def extract_reference_identifiers(html: str) -> list[tuple[str, ...]]:
    """Extract aliases grouped by bibliography entry, in document order.

    :param str html: arXiv LaTeXML HTML.
    :return list[tuple[str, ...]]: Entries including those without explicit IDs.
    """
    parser = _BibliographyParser()
    parser.feed(html)
    parser.close()
    return parser.entries


class ArxivClient:
    """Small request-scoped client; no persistent HTML or metadata cache."""

    def __init__(self) -> None:
        """Initialize request spacing shared by HTML and metadata reads."""
        self._last_request: float | None = None

    def _get(
        self, url: str, *, params: dict[str, str | int] | None = None
    ) -> str | None:
        """Read an arXiv response with a timeout and request spacing.

        :param str url: arXiv endpoint.
        :param dict[str, str | int] | None params: Query parameters.
        :return str | None: Response text, or unavailable.
        """
        if self._last_request is not None:
            time.sleep(max(0.0, 3.0 - (time.monotonic() - self._last_request)))
        self._last_request = time.monotonic()
        try:
            response = requests.get(
                url,
                params=params,
                timeout=30,
                headers={
                    "User-Agent": "CiteMesh (https://github.com/pszemraj/CiteMesh)"
                },
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.debug("arXiv request unavailable for %s: %s", url, exc)
            return None
        return response.text

    def get_bibliography(self, identifier: str) -> list[tuple[str, ...]] | None:
        """Read the seed's bibliography, retaining the requested version.

        :param str identifier: Recognized arXiv suffix.
        :return list[tuple[str, ...]] | None: Entries, or unavailable HTML.
        """
        html = self._get(f"https://arxiv.org/html/{identifier}")
        return extract_reference_identifiers(html) if html is not None else None

    def get_papers(self, identifiers: list[str]) -> dict[str, Paper] | None:
        """Fetch exact arXiv metadata in one Atom request.

        :param list[str] identifiers: Canonical arxiv-prefixed identifiers.
        :return dict[str, Paper] | None: Available records, or unavailable metadata.
        """
        if not identifiers:
            return {}
        content = self._get(
            "https://export.arxiv.org/api/query",
            params={
                "id_list": ",".join(value.split(":", 1)[1] for value in identifiers),
                "max_results": len(identifiers),
            },
        )
        if content is None:
            return None
        try:
            feed = ET.fromstring(content)
        except ET.ParseError:
            logger.debug("arXiv metadata response was not valid Atom XML.")
            return None
        papers = {}
        for entry in feed.findall("a:entry", _ATOM):
            identifier = arxiv_identifier(entry.findtext("a:id", "", _ATOM))
            if not identifier:
                continue
            paper_id = f"arxiv:{strip_arxiv_version(identifier).lower()}"
            if paper_id not in identifiers:
                continue
            title = " ".join(entry.findtext("a:title", "", _ATOM).split())
            if not title:
                continue
            published = entry.findtext("a:published", "", _ATOM)
            year = int(published[:4]) if published[:4].isdigit() else None
            try:
                paper = Paper(
                    paper_id=paper_id,
                    title=title,
                    year=year,
                    authors=[
                        Author(
                            name=" ".join(author.findtext("a:name", "", _ATOM).split())
                        )
                        for author in entry.findall("a:author", _ATOM)
                    ],
                    abstract=" ".join(entry.findtext("a:summary", "", _ATOM).split()),
                    arxiv_id=strip_arxiv_version(identifier),
                    doi=entry.findtext("arxiv:doi", "", _ATOM).strip(),
                    venue=entry.findtext("arxiv:journal_ref", "", _ATOM).strip(),
                    categories=[
                        category.get("term", "")
                        for category in entry.findall("a:category", _ATOM)
                    ],
                )
            except ValueError as exc:
                logger.debug(
                    "Skipping unusable arXiv metadata for %s: %s", paper_id, exc
                )
                continue
            papers[paper_id] = paper
        return papers
