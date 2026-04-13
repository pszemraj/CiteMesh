"""Paper-identifier normalization helpers."""

from __future__ import annotations

import re
from typing import Optional
from urllib.parse import unquote, urlparse


def strip_arxiv_version(identifier: str) -> str:
    """Strip trailing arXiv version suffixes.

    :param str identifier: Raw arXiv identifier candidate.
    :return str: Identifier without trailing ``v<digits>`` suffix.
    """
    return re.sub(r"v\d+$", "", identifier.strip(), flags=re.IGNORECASE)


def extract_arxiv_identifier(raw_path: str) -> Optional[str]:
    """Extract an arXiv identifier from an arXiv-style URL path.

    :param str raw_path: URL path component such as ``/abs/2508.14040``.
    :return Optional[str]: Canonical arXiv identifier suffix, or ``None`` on failure.
    """
    segments = [segment for segment in raw_path.strip("/").split("/") if segment]
    if not segments:
        return None

    if segments[0] in {"abs", "pdf"}:
        candidate = "/".join(segments[1:])
    else:
        candidate = "/".join(segments)

    candidate = candidate.strip()
    if not candidate:
        return None

    if candidate.endswith(".pdf"):
        candidate = candidate[:-4]
    candidate = candidate.strip()
    candidate = re.sub(r"^(?:arxiv:)", "", candidate, flags=re.IGNORECASE)
    candidate = strip_arxiv_version(candidate)
    return candidate or None


def host_matches_domain(host: str, domain: str) -> bool:
    """Return whether a parsed host belongs to an expected domain.

    :param str host: Parsed hostname candidate.
    :param str domain: Expected domain suffix.
    :return bool: ``True`` when host is exactly ``domain`` or a valid subdomain.
    """
    normalized_host = host.lower().strip()
    normalized_domain = domain.lower().strip()
    return normalized_host == normalized_domain or normalized_host.endswith(
        f".{normalized_domain}"
    )


def external_ids_from_canonical_paper_id(paper_id: str) -> tuple[str, str]:
    """Infer arXiv/DOI identifiers from a canonicalized paper ID.

    :param str paper_id: Canonical paper identifier.
    :return tuple[str, str]: ``(arxiv_id, doi)`` inference tuple.
    """
    normalized = str(paper_id or "").strip()
    if not normalized:
        return "", ""

    lowered = normalized.lower()
    if lowered.startswith("arxiv:"):
        return strip_arxiv_version(normalized.split(":", 1)[1]), ""

    if lowered.startswith("doi:"):
        return "", normalized.split(":", 1)[1].strip()

    if re.match(r"^10\.\d{4,9}/\S+$", normalized):
        return "", normalized

    return "", ""


def normalize_paper_id(paper_id: str) -> str:
    """Normalize paper identifiers and DOI/arXiv URLs into canonical tokens.

    :param str paper_id: Raw user-provided identifier or URL.
    :return str: Canonical Semantic Scholar paper identifier string.
    :raises ValueError: If the identifier is invalid or empty after trimming.
    """
    if not isinstance(paper_id, str):
        raise ValueError(f"Invalid paper ID: {paper_id}")

    normalized = paper_id.strip()
    if not normalized:
        raise ValueError(f"Invalid paper ID: {paper_id}")

    lowered = normalized.lower()

    if lowered.startswith("doi:"):
        suffix = unquote(normalized.split(":", 1)[1]).strip()
        if not suffix:
            raise ValueError(f"Invalid paper ID: {paper_id}")
        return suffix

    if lowered.startswith("arxiv:"):
        suffix = unquote(normalized.split(":", 1)[1]).strip()
        suffix = strip_arxiv_version(suffix)
        if not suffix:
            raise ValueError(f"Invalid paper ID: {paper_id}")
        return f"arxiv:{suffix}"

    if lowered.startswith("http://") or lowered.startswith("https://"):
        parsed = urlparse(normalized)
        host = (parsed.hostname or "").lower()
        path = unquote(parsed.path)

        if host_matches_domain(host, "arxiv.org"):
            arxiv_id = extract_arxiv_identifier(path)
            if arxiv_id:
                return f"arxiv:{arxiv_id}"

        if host_matches_domain(host, "doi.org"):
            doi_id = path.strip("/")
            if doi_id:
                return doi_id

    if "://" not in lowered and "/" in lowered:
        parsed = urlparse(f"https://{normalized}")
        host = (parsed.hostname or "").lower()
        path = unquote(parsed.path)

        if host_matches_domain(host, "doi.org"):
            doi_id = path.strip("/")
            if doi_id:
                return doi_id

        if host_matches_domain(host, "arxiv.org"):
            arxiv_id = extract_arxiv_identifier(path)
            if arxiv_id:
                return f"arxiv:{arxiv_id}"

    return normalized
