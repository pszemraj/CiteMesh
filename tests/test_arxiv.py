"""Network-free contracts for the seed-only arXiv fallback service."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import requests

from citemesh.services import arxiv as arxiv_module
from citemesh.services.arxiv import (
    ArxivClient,
    arxiv_identifier,
    extract_reference_identifiers,
)


class _Response:
    """Minimal text response used to keep arXiv tests offline."""

    def __init__(self, text: str = "", status_code: int = 200) -> None:
        """Store text and a response status.

        :param str text: Response body returned by the fake HTTP request.
        :param int status_code: HTTP status returned by the fake request.
        :return None: Initializes the response fixture.
        """
        self.text = text
        self.status_code = status_code

    def raise_for_status(self) -> None:
        """Raise the same request error used by Requests for HTTP failures.

        :raises requests.HTTPError: When the configured status is unsuccessful.
        :return None: Returns for successful responses.
        """
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)


_BIBLIOGRAPHY_HTML = """
<article>
  <p>Article prose must ignore arXiv:1111.11111 and 10.5555/prose.</p>
  <section id="bib" class="ltx_bibliography">
    <ol class="ltx_biblist">
      <li id="bib.one" class="ltx_bibitem">
        <span>[1] arXiv:2608.27147v2; DOI 10.1000/Visible].</span>
        <a href="https://arxiv.org/abs/2608.27147v2"><span>split</span> link</a>
        <a href="https://doi.org/10.1000/Visible?source=arxiv#reference">DOI</a>
      </li>
      <li id="bib.two" class="ltx_bibitem">
        <a href="https://arxiv.org/pdf/hep-th/9901001v3.pdf"><em>legacy</em></a>
        <span>doi:10.2000/Nested).</span>
      </li>
      <li id="bib.three" class="ltx_bibitem">Publication without identifiers.</li>
    </ol>
  </section>
  <p>Trailing prose must ignore arXiv:2222.22222.</p>
</article>
"""


def test_extract_bibliography_keeps_entry_order_and_ignores_article_prose() -> None:
    """LaTeXML bibliography entries retain grouped explicit identifier aliases.

    :return None: Checks document scoping, nesting, canonicalization, and order.
    """
    assert extract_reference_identifiers(_BIBLIOGRAPHY_HTML) == [
        ("arxiv:2608.27147", "10.1000/visible"),
        ("arxiv:hep-th/9901001", "10.2000/nested"),
        (),
    ]


def test_extract_bibliography_accepts_semantic_bibliography_roles() -> None:
    """A role-based LaTeXML bibliography is accepted without class selectors.

    :return None: Checks the alternate element attributes emitted by arXiv HTML.
    """
    html = """
    <main>
      <section role="doc-bibliography">
        <div role="doc-biblioentry">
          <a href="https://arxiv.org/html/2401.01234v4">nested <span>label</span></a>
        </div>
      </section>
    </main>
    """
    assert extract_reference_identifiers(html) == [("arxiv:2401.01234",)]


def test_extract_bibliography_recognizes_plain_arxiv_urls_and_punctuated_dois() -> None:
    """Visible URLs and DOI punctuation do not require a hyperlink to resolve.

    :return None: Checks explicit identifier extraction from visible entry text.
    """
    html = """
    <section class="ltx_bibliography">
      <li class="ltx_bibitem">
        https://arxiv.org/html/2608.27147v5). DOI 10.3141/Plain].
      </li>
    </section>
    """
    assert extract_reference_identifiers(html) == [
        ("arxiv:2608.27147", "10.3141/plain")
    ]


def test_extract_bibliography_recognizes_bare_legacy_ids_and_implicit_items() -> None:
    """Bare old-style IDs survive the omitted closing tag emitted by LaTeXML.

    :return None: Checks legacy text extraction and implicit list-item closure.
    """
    html = """
    <section class="ltx_bibliography">
      <ol class="ltx_biblist">
        <li class="ltx_bibitem">hep-th/9709013, An early paper.
        <li class="ltx_bibitem">arXiv:2608.27147v2, A later paper.</li>
      </ol>
    </section>
    """
    assert extract_reference_identifiers(html) == [
        ("arxiv:hep-th/9709013",),
        ("arxiv:2608.27147",),
    ]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("arXiv:2608.27147v3", "2608.27147v3"),
        ("https://arxiv.org/html/2608.27147v3", "2608.27147v3"),
        ("https://arxiv.org/pdf/2608.27147v3.pdf?download=1", "2608.27147v3"),
        ("https://arxiv.org/abs/hep-th/9901001v2", "hep-th/9901001v2"),
        ("https://example.com/abs/2608.27147", None),
        ("arxiv:invalid", None),
    ],
)
def test_arxiv_identifier_accepts_versioned_seed_identifiers_and_urls(
    value: str, expected: str | None
) -> None:
    """Seed input parsing preserves versions for the arXiv HTML request.

    :param str value: User or metadata identifier supplied to the parser.
    :param str | None expected: Expected request suffix or rejection.
    :return None: Checks version-aware identifier parsing.
    """
    assert arxiv_identifier(value) == expected


def test_get_bibliography_uses_the_supplied_version_and_skips_missing_bibliographies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The HTML fallback requests its explicit version and accepts no bibliography.

    :param pytest.MonkeyPatch monkeypatch: Fixture that replaces Requests.
    :return None: Checks exact URL construction and the empty-bibliography result.
    """
    get = MagicMock(
        return_value=_Response("<article><p>No bibliography.</p></article>")
    )
    monkeypatch.setattr(arxiv_module.requests, "get", get)

    assert ArxivClient().get_bibliography("2608.27147v3") == []
    assert get.call_args.args == ("https://arxiv.org/html/2608.27147v3",)
    assert get.call_args.kwargs["timeout"] == 30


@pytest.mark.parametrize(
    "outcome",
    [_Response(status_code=404), requests.Timeout("arXiv unavailable")],
)
def test_get_bibliography_returns_unavailable_for_http_and_transport_failures(
    monkeypatch: pytest.MonkeyPatch, outcome: _Response | requests.Timeout
) -> None:
    """Missing HTML and transport failures leave the fallback unavailable.

    :param pytest.MonkeyPatch monkeypatch: Fixture that replaces Requests.
    :param _Response | requests.Timeout outcome: Simulated request outcome.
    :return None: Checks that optional recovery failure is non-fatal.
    """
    get = MagicMock()
    if isinstance(outcome, Exception):
        get.side_effect = outcome
    else:
        get.return_value = outcome
    monkeypatch.setattr(arxiv_module.requests, "get", get)

    assert ArxivClient().get_bibliography("2608.27147v1") is None


def test_arxiv_client_spaces_requests_by_the_documented_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Consecutive arXiv requests wait only for the unelapsed interval.

    :param pytest.MonkeyPatch monkeypatch: Fixture controlling requests and time.
    :return None: Checks three-second request pacing.
    """
    monotonic = MagicMock(side_effect=[0.0, 1.25, 3.0])
    sleep = MagicMock()
    monkeypatch.setattr(arxiv_module.time, "monotonic", monotonic)
    monkeypatch.setattr(arxiv_module.time, "sleep", sleep)
    monkeypatch.setattr(
        arxiv_module.requests, "get", MagicMock(return_value=_Response())
    )

    client = ArxivClient()
    assert client._get("https://arxiv.org/html/first") == ""
    assert client._get("https://arxiv.org/html/second") == ""

    sleep.assert_called_once_with(1.75)


def test_get_papers_maps_atom_metadata_and_ignores_unrequested_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Atom metadata becomes CiteMesh papers keyed by canonical arXiv identity.

    :param pytest.MonkeyPatch monkeypatch: Fixture that replaces Requests.
    :return None: Checks title, authors, venue, categories, and version removal.
    """
    atom = """
    <feed xmlns="http://www.w3.org/2005/Atom"
          xmlns:arxiv="http://arxiv.org/schemas/atom">
      <entry>
        <id>http://arxiv.org/abs/2608.27147v2</id>
        <title>  A\n  normalized title </title>
        <published>2026-08-27T00:00:00Z</published>
        <summary> An\n abstract. </summary>
        <author><name> Ada\n Example </name></author>
        <author><name> Bob Example </name></author>
        <arxiv:doi>10.1000/example</arxiv:doi>
        <arxiv:journal_ref>Example Journal</arxiv:journal_ref>
        <category term="cs.LG" />
        <category term="stat.ML" />
      </entry>
      <entry>
        <id>http://arxiv.org/abs/9999.99999v1</id>
        <title>Unrequested record</title>
      </entry>
    </feed>
    """
    get = MagicMock(return_value=_Response(atom))
    monkeypatch.setattr(arxiv_module.requests, "get", get)

    papers = ArxivClient().get_papers(["arxiv:2608.27147"])

    assert list(papers) == ["arxiv:2608.27147"]
    paper = papers["arxiv:2608.27147"]
    assert paper.title == "A normalized title"
    assert paper.year == 2026
    assert paper.abstract == "An abstract."
    assert [author.name for author in paper.authors] == ["Ada Example", "Bob Example"]
    assert paper.arxiv_id == "2608.27147"
    assert paper.doi == "10.1000/example"
    assert paper.venue == "Example Journal"
    assert paper.categories == ["cs.LG", "stat.ML"]
    assert get.call_args.kwargs["params"] == {
        "id_list": "2608.27147",
        "max_results": 1,
    }


@pytest.mark.parametrize(
    "content",
    ["not atom", '<feed xmlns="http://www.w3.org/2005/Atom"><entry>'],
)
def test_get_papers_rejects_bad_atom_responses_without_requesting_again(
    monkeypatch: pytest.MonkeyPatch, content: str
) -> None:
    """Malformed metadata responses preserve provider unavailability.

    :param pytest.MonkeyPatch monkeypatch: Fixture that replaces Requests.
    :param str content: Malformed Atom response body.
    :return None: Checks parse failure handling.
    """
    get = MagicMock(return_value=_Response(content))
    monkeypatch.setattr(arxiv_module.requests, "get", get)

    assert ArxivClient().get_papers(["arxiv:2608.27147"]) is None
    get.assert_called_once()


def test_get_papers_returns_empty_without_making_a_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No unresolved arXiv identifiers need no metadata lookup.

    :param pytest.MonkeyPatch monkeypatch: Fixture that replaces Requests.
    :return None: Checks the zero-work metadata fast path.
    """
    get = MagicMock()
    monkeypatch.setattr(arxiv_module.requests, "get", get)

    assert ArxivClient().get_papers([]) == {}
    get.assert_not_called()
