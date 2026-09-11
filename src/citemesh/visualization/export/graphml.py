"""GraphML attribute-key, value, and XML-safety normalization."""

from __future__ import annotations

import json
import re

from .keys import slug_key

GRAPHML_LAYOUT_METADATA_KEY = "citemesh_graphml_determinism"
GRAPHML_LAYOUT_VERSION_KEY = "citemesh_graphml_writer_version"

# tab/newline/CR, lone surrogates, and the two non-characters U+FFFE/U+FFFF.
_XML_INVALID_CHARS_RE = re.compile(
    "[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff" + chr(0xFFFE) + chr(0xFFFF) + "]"
)


def _xml_safe_graph_value(value: object) -> object:
    """Normalize nullable attributes and XML-invalid text bound for GraphML.

    Upstream titles/abstracts occasionally carry stray control bytes;
    ``nx.write_graphml`` passes them through and produces a file no XML
    parser will accept.

    :param object value: Raw attribute value.
    :return object: Empty text for nulls, otherwise an XML-safe attribute value.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return _XML_INVALID_CHARS_RE.sub("", value)
    return value


def _graphml_metadata_key(raw_key: object) -> str:
    """Normalize metadata key for GraphML graph-level attributes.

    :param object raw_key: Source metadata key.
    :return str: Sanitized GraphML-safe key.
    """
    return slug_key(
        raw_key,
        prefix="citemesh_meta_",
        fallback="metadata",
        keep_underscores=True,
    )


def _graphml_metadata_value(raw_value: object) -> str:
    """Normalize metadata value for GraphML graph-level attributes.

    :param object raw_value: Source metadata value.
    :return str: Scalar/serialized value for GraphML export.
    """
    if isinstance(raw_value, (str, int, float, bool)) or raw_value is None:
        return str(raw_value)
    return json.dumps(raw_value, sort_keys=True)
