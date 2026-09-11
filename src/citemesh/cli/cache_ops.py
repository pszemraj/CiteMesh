"""Cache inspection and destructive cache-clearing helpers.

Owns the cache scan report, the interactive confirmation prompts guarding
destructive operations, and the clear/rebuild routines that remove cached
embedding artifacts while preserving the user config.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from filelock import Timeout
from rich.text import Text

from citemesh._runtime import stdin_isatty
from citemesh.core.user_config import (
    USER_CONFIG_FILENAME,
    ConfigFileError,
    config_lock,
)
from citemesh.data import format_bytes, get_cache_dir
from citemesh.data.cache import (
    CACHE_COORDINATION_DIRNAME,
    cache_operation_lock,
    legacy_macos_cache_root,
    path_exists,
)

from . import console
from .build_options import _normalized_cache_reason
from .console import logger
from .parser import _output_table

LARGE_CACHE_CLEAR_WARNING_BYTES = 1024 * 1024 * 1024


def _embedding_cache_directory_stats() -> tuple[Path, int, int]:
    """Return embedding cache directory path + file/size totals.

    :return tuple[Path, int, int]: ``(path, files, size_bytes)``.
    """
    embedding_cache_dir = (
        get_cache_dir("embeddings", create=False).expanduser().resolve()
    )
    if not path_exists(embedding_cache_dir):
        return embedding_cache_dir, 0, 0
    files, size_bytes = _scan_path_stats(embedding_cache_dir)
    return embedding_cache_dir, files, size_bytes


def _confirm_destructive_cache_action(
    *,
    root: Path,
    total_files: int,
    total_bytes: int,
    confirmed: bool,
    operation: str,
    reason: str | None,
) -> bool:
    """Apply the shared warning, non-TTY, rationale, and prompt workflow.

    :param Path root: Cache path affected by the operation.
    :param int total_files: Files present under ``root``.
    :param int total_bytes: Bytes present under ``root``.
    :param bool confirmed: Whether the operation was explicitly acknowledged.
    :param str operation: ``rebuild`` or ``clear`` action selector.
    :param Optional[str] reason: Optional operator rationale.
    :return bool: ``True`` when the destructive action may proceed.
    """
    if operation == "rebuild":
        confirmation_flag = "--overwrite-cache"
        operation_label = "embedding cache rebuild"
        reason_label = "Cache overwrite"
        non_interactive_error = (
            "Refusing --force-rebuild-cache in non-interactive mode without "
            "--overwrite-cache. Re-run with --overwrite-cache to proceed."
        )
        prompt = "Proceed with embedding cache overwrite?"
        large_cache_detail = "Clearing may require long rehydration."
        eof_action = "build"
    elif operation == "clear":
        confirmation_flag = "--yes"
        operation_label = reason_label = "Cache clear"
        non_interactive_error = (
            "Refusing to clear cache in non-interactive mode without --yes. "
            "Re-run with: citemesh cache clear --yes"
        )
        prompt = f"Delete CiteMesh cache directory '{root}'?"
        large_cache_detail = "Deletion is immediate and irreversible."
        eof_action = "cache clear"
    else:
        raise ValueError(f"Unsupported destructive cache operation: {operation}")

    total_size_label = format_bytes(total_bytes)
    large_cache = total_bytes >= LARGE_CACHE_CLEAR_WARNING_BYTES
    threshold_label = format_bytes(LARGE_CACHE_CLEAR_WARNING_BYTES)
    normalized_reason = _normalized_cache_reason(reason)
    # Namespace resolution happens after model load, so the rebuild snapshot
    # can only show whole-directory totals; say so rather than implying the
    # entire directory is deleted.
    scope_note = (
        "Snapshot covers the whole embedding cache directory; only the "
        "resolved model namespace payload will be cleared."
        if operation == "rebuild"
        else None
    )

    if confirmed:
        logger.warning(
            "%s acknowledged destructive %s (root=%s files=%d size=%s).",
            confirmation_flag,
            operation_label,
            root,
            total_files,
            total_size_label,
        )
    elif not stdin_isatty():
        logger.error(non_interactive_error)
        return False
    else:
        logger.warning("%s requires explicit confirmation.", operation_label)
        logger.warning(
            "Cache snapshot: root=%s files=%d size=%s.",
            root,
            total_files,
            total_size_label,
        )

    if scope_note:
        logger.warning("%s", scope_note)
    if large_cache:
        logger.warning(
            "Large cache warning: %s >= %s. %s",
            total_size_label,
            threshold_label,
            large_cache_detail,
        )
    if normalized_reason:
        logger.warning("%s rationale: %s", reason_label, normalized_reason)
    if confirmed:
        return True
    logger.warning(
        "Use %s to bypass this prompt in scripted/non-interactive workflows.",
        confirmation_flag,
    )
    # Text, not markup: the clear prompt interpolates a path that could
    # otherwise be parsed as console tags.
    styled_prompt = Text.assemble((prompt, "bold"), (" [y/N]: ", "dim"))
    try:
        response = console.log_console.input(styled_prompt).strip().lower()
    except EOFError:
        logger.error("No confirmation input received; %s aborted.", eof_action)
        return False
    return response in {"y", "yes"}


def _confirm_force_rebuild_cache(args: argparse.Namespace) -> bool:
    """Confirm destructive embedding namespace rebuild requested by CLI flags.

    :param argparse.Namespace args: Parsed build CLI arguments.
    :return bool: ``True`` when build may proceed.
    """
    if (
        str(args.command) != "build"
        or str(args.strategy) not in {"embedding", "hybrid"}
        or not bool(args.force_rebuild_cache)
    ):
        return True

    embedding_cache_dir, total_files, total_bytes = _embedding_cache_directory_stats()
    return _confirm_destructive_cache_action(
        root=embedding_cache_dir,
        total_files=total_files,
        total_bytes=total_bytes,
        confirmed=bool(args.overwrite_cache),
        operation="rebuild",
        reason=getattr(args, "cache_overwrite_reason", None),
    )


def _confirmed_cache_clear(
    cache_root: Path, assume_yes: bool, clear_reason: str | None
) -> bool:
    """Return whether cache directory deletion is confirmed.

    :param Path cache_root: Cache root directory targeted for deletion.
    :param bool assume_yes: Skip interactive prompt when ``True``.
    :param Optional[str] clear_reason: Optional operator rationale for cache clear.
    :return bool: ``True`` if cache deletion should proceed.
    """
    total_files, total_bytes = _scan_path_stats(cache_root)
    return _confirm_destructive_cache_action(
        root=cache_root,
        total_files=total_files,
        total_bytes=total_bytes,
        confirmed=assume_yes,
        operation="clear",
        reason=clear_reason,
    )


def _clear_cache_directory(*, assume_yes: bool, clear_reason: str | None) -> int:
    """Clear cached data while preserving configuration and its coordination lock.

    :param bool assume_yes: Whether to bypass interactive confirmation.
    :param Optional[str] clear_reason: Optional operator rationale for cache clear.
    :return int: Process exit code (``0`` success, ``1`` failure/cancelled).
    """
    raw_cache_root = get_cache_dir(create=False)
    cache_root = raw_cache_root.expanduser().resolve()

    if len(cache_root.parts) <= 1:
        logger.error("Refusing to clear unsafe cache path: %s", cache_root)
        return 1
    if cache_root == Path.home().expanduser().resolve():
        logger.error("Refusing to clear home directory path: %s", cache_root)
        return 1

    if not path_exists(cache_root):
        logger.info("Cache directory does not exist: %s", cache_root)
        return 0

    if not _confirmed_cache_clear(cache_root, assume_yes, clear_reason=clear_reason):
        logger.info("Cache clear aborted.")
        return 1

    config_path = cache_root / USER_CONFIG_FILENAME
    try:
        # The exclusive root lock prevents a live embedding operation from
        # losing its namespace lock inode while this removes cache payloads.
        with cache_operation_lock(cache_root, exclusive=True, blocking=False):
            # Hold the config lock so an in-flight atomic write cannot lose its
            # temporary file; preserve the lock's path for waiting writers.
            with config_lock(config_path) as lock_path:
                preserved_config = path_exists(config_path) or config_path.is_symlink()
                for child in sorted(cache_root.iterdir()):
                    if child in {
                        config_path,
                        lock_path,
                        cache_root / CACHE_COORDINATION_DIRNAME,
                    }:
                        continue
                    if child.is_dir() and not child.is_symlink():
                        shutil.rmtree(child)
                    else:
                        child.unlink()
    except Timeout:
        logger.error(
            "Cache directory is busy with an active cache operation: %s", cache_root
        )
        return 1
    except (OSError, ConfigFileError) as exc:
        logger.error("Failed to clear cache directory %s: %s", cache_root, exc)
        return 1

    suffix_parts = []
    if preserved_config:
        suffix_parts.append(f"preserved {USER_CONFIG_FILENAME}")
    normalized_reason = _normalized_cache_reason(clear_reason)
    if normalized_reason:
        suffix_parts.append(f"reason={normalized_reason}")
    if suffix_parts:
        logger.info(
            "✓ Cleared cache directory: %s (%s)", cache_root, ", ".join(suffix_parts)
        )
    else:
        logger.info("✓ Cleared cache directory: %s", cache_root)
    return 0


def _scan_path_stats(path: Path) -> tuple[int, int]:
    """Return file-count and total size stats for a path.

    :param Path path: Directory or file path to scan.
    :return tuple[int, int]: ``(file_count, size_bytes)`` totals.
    """
    if path.is_symlink() or path.is_file():
        try:
            return 1, path.lstat().st_size
        except OSError:
            return 1, 0

    file_count = 0
    size_bytes = 0
    for candidate in path.rglob("*"):
        if not candidate.is_symlink() and not candidate.is_file():
            continue
        file_count += 1
        try:
            size_bytes += candidate.lstat().st_size
        except OSError:
            continue
    return file_count, size_bytes


def _scan_cache_directory() -> int:
    """Scan the CiteMesh cache root and print a usage summary.

    :return int: Process exit code (``0`` success, ``1`` failure).
    """
    raw_cache_root = get_cache_dir(create=False)
    cache_root = raw_cache_root.expanduser().resolve()

    if not path_exists(cache_root):
        logger.info("Cache directory does not exist: %s", cache_root)
        _log_legacy_macos_cache_hint(cache_root)
        return 0
    if not cache_root.is_dir():
        logger.error("Cache path exists but is not a directory: %s", cache_root)
        return 1

    section_rows: list[tuple[str, int, int]] = []
    for child in sorted(cache_root.iterdir(), key=lambda item: item.name):
        files, size_bytes = _scan_path_stats(child)
        section_rows.append((child.name, files, size_bytes))

    total_files = sum(row[1] for row in section_rows)
    total_bytes = sum(row[2] for row in section_rows)

    console.output_console.print(
        Text.assemble(("Cache root: ", "bold"), str(cache_root))
    )
    table = _output_table("CiteMesh Cache Scan")
    table.add_column("Section", style="cyan")
    table.add_column("Files", justify="right")
    table.add_column("Size", justify="right")

    if section_rows:
        for name, files, size_bytes in section_rows:
            table.add_row(Text(name), f"{files:,}", format_bytes(size_bytes))
    else:
        table.add_row("(empty)", "0", "0 B")

    table.add_section()
    table.add_row("TOTAL", f"{total_files:,}", format_bytes(total_bytes), style="bold")
    console.output_console.print(table)
    _log_legacy_macos_cache_hint(cache_root)
    return 0


def _log_legacy_macos_cache_hint(cache_root: Path) -> None:
    """Point at the pre-unification macOS cache directory when it lingers.

    :param Path cache_root: Active resolved cache root.
    :return None: Emits an info-level migration hint when applicable.
    """
    legacy_root = legacy_macos_cache_root()
    if legacy_root is None:
        return
    try:
        if legacy_root.resolve() == cache_root:
            return
    except OSError:
        return
    logger.info(
        "Legacy macOS cache directory detected at %s. CiteMesh now uses %s; "
        "move or delete the old directory to reclaim space.",
        legacy_root,
        cache_root,
    )
