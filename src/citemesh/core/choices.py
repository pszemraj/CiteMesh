"""Dependency-free option vocabularies shared by config, CLI, and runtimes."""

STRATEGY_CHOICES: tuple[str, ...] = (
    "recommendation",
    "citation",
    "embedding",
    "hybrid",
)
THEME_CHOICES: tuple[str, ...] = ("light", "dark", "solarized", "auto")
DEVICE_CHOICES: tuple[str, ...] = ("auto", "cuda", "mps", "cpu")
MODEL_PROFILE_CHOICES: tuple[str, ...] = ("auto", "default", "embeddinggemma")
SEMANTIC_SOURCE_CHOICES: tuple[str, ...] = ("candidates", "arxiv-corpus")
STORAGE_PRECISION_CHOICES: tuple[str, ...] = ("int8", "float32")
SEARCH_MODE_CHOICES: tuple[str, ...] = ("auto", "local", "s2")
EXPORT_FORMATS: tuple[str, ...] = (
    "png",
    "html",
    "plotly",
    "dashboard",
    "json",
    "csv",
    "bibtex",
    "graphml",
)

EXPORT_CHOICES: tuple[str, ...] = (*EXPORT_FORMATS, "all")
