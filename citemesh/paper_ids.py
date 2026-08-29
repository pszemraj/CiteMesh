"""Paper-identifier normalization helpers."""

from __future__ import annotations

import re
from typing import Any, List, Optional
from urllib.parse import unquote, urlparse


def strip_arxiv_version(identifier: str) -> str:
    """Strip trailing arXiv version suffixes.

    :param str identifier: Raw arXiv identifier candidate.
    :return str: Identifier without trailing ``v<digits>`` suffix.
    """
    return re.sub(r"v\d+$", "", identifier.strip(), flags=re.IGNORECASE)


_ARXIV_IDENTIFIER_PATTERN = re.compile(
    r"^(?:\d{4}\.\d{4,5}|[a-z\-]+(?:\.[a-z\-]+)?/\d{7})(?:v\d+)?$",
    re.IGNORECASE,
)


def recognize_arxiv_identifier(
    identifier: Any, *, allow_bare: bool = False
) -> Optional[str]:
    """Recognize and canonicalize an arXiv identifier candidate.

    ``normalize_paper_id`` deliberately leaves a bare arXiv-looking user input
    unchanged. Internal identity matching can opt into ``allow_bare`` so that
    dataset IDs such as ``2508.12345v2`` still match API ``arxiv:`` IDs.

    :param Any identifier: Raw identifier candidate.
    :param bool allow_bare: Whether to accept identifiers without ``arxiv:``.
    :return Optional[str]: Canonical ``arxiv:<suffix>`` token, or ``None``.
    """
    if not isinstance(identifier, str):
        return None

    normalized = identifier.strip()
    if not normalized:
        return None

    lowered = normalized.lower()
    explicitly_prefixed = lowered.startswith("arxiv:")
    if explicitly_prefixed:
        candidate = unquote(normalized.split(":", 1)[1]).strip()
    elif allow_bare:
        candidate = normalized
    else:
        return None

    if not candidate:
        return None
    if not explicitly_prefixed and not _ARXIV_IDENTIFIER_PATTERN.fullmatch(candidate):
        return None
    return f"arxiv:{strip_arxiv_version(candidate)}"


def paper_identifier_aliases(
    *, paper_id: Any = "", arxiv_id: Any = "", doi: Any = ""
) -> List[str]:
    """Build stable identifier aliases for one paper payload.

    This expands identifiers supplied by separate API fields as well as the
    primary paper ID. It intentionally recognizes bare arXiv IDs here without
    changing the stricter public ``normalize_paper_id`` contract.

    :param Any paper_id: Primary Semantic Scholar, DOI, arXiv, or URL ID.
    :param Any arxiv_id: Optional arXiv suffix field.
    :param Any doi: Optional DOI suffix field.
    :return List[str]: Sorted normalized and raw identifier aliases.
    """
    aliases: set[str] = set()
    raw_candidates = (paper_id, arxiv_id, doi)

    for candidate in raw_candidates:
        if not isinstance(candidate, str):
            continue
        normalized = candidate.strip()
        if not normalized:
            continue
        aliases.add(normalized)
        try:
            canonical_identifier = normalize_paper_id(normalized)
            aliases.add(canonical_identifier)
        except ValueError:
            continue
        arxiv_alias = recognize_arxiv_identifier(canonical_identifier)
        if arxiv_alias:
            aliases.add(arxiv_alias)
            aliases.add(arxiv_alias.split(":", 1)[1])

    for candidate in (paper_id, arxiv_id):
        arxiv_alias = recognize_arxiv_identifier(candidate, allow_bare=True)
        if arxiv_alias:
            aliases.add(arxiv_alias)
            aliases.add(arxiv_alias.split(":", 1)[1])

    return sorted(aliases)


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
    prefixed_candidate = (
        candidate if candidate.lower().startswith("arxiv:") else f"arxiv:{candidate}"
    )
    recognized = recognize_arxiv_identifier(prefixed_candidate)
    return recognized.split(":", 1)[1] if recognized else None


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


def _normalize_hosted_identifier(candidate: str) -> Optional[str]:
    """Normalize DOI/arXiv identifiers embedded in hosted URL-like inputs.

    :param str candidate: URL-like identifier candidate.
    :return Optional[str]: Canonical ID when the host/path matches a known source.
    """
    parsed = urlparse(candidate)
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

    return None


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
        arxiv_identifier = recognize_arxiv_identifier(normalized)
        if not arxiv_identifier:
            raise ValueError(f"Invalid paper ID: {paper_id}")
        return arxiv_identifier

    if lowered.startswith("http://") or lowered.startswith("https://"):
        hosted_identifier = _normalize_hosted_identifier(normalized)
        if hosted_identifier:
            return hosted_identifier

    if "://" not in lowered and "/" in lowered:
        hosted_identifier = _normalize_hosted_identifier(f"https://{normalized}")
        if hosted_identifier:
            return hosted_identifier

    return normalized
