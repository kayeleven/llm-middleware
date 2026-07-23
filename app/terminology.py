"""
terminology.py — Wargame terminology enrichment (README Step 5).

Responsibilities:
- Load the acronym CSV ONCE at startup into an in-memory dict.
- Build ONE compiled regex from all acronyms for fast per-message matching.
- Given a piece of text (the latest user message, and optionally a freshly
  generated image description), find every acronym present using EXACT-CASE,
  WHOLE-WORD matching, and return a formatted system block of definitions.

MULTIPLE DEFINITIONS PER ACRONYM:
- An acronym may legitimately have SEVERAL meanings (e.g. "MOE" =
  "Measure of Effectiveness" AND "Main Operating Base" in different contexts).
- We preserve ALL definitions for each acronym and present every one to the
  model, so the LLM can use surrounding context to pick the correct meaning.
- Data model is therefore {acronym: [definition, definition, ...]}.

Design notes:
- PURE LOGIC after the one-time load. Verifiable by reading. No network I/O.
- Exact-case matching: "MoE" matches, "moe"/"MOE" do NOT (compiled WITHOUT
  re.IGNORECASE).
- Whole-word matching via lookaround boundaries (not \\b) for predictable
  behavior with acronyms containing digits or punctuation.
- Longest-match-first in the alternation so the fullest form wins.
"""

from __future__ import annotations

import csv
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------- #
# Module-level state, populated by load_acronyms() at startup.
# ---------------------------------------------------------------------- #
# Maps the exact-case acronym string -> LIST of definition strings.
# A list (not a single string) so multiple meanings are preserved and all are
# offered to the model for context-based disambiguation.
_definitions: dict[str, list[str]] = {}

# The single compiled regex matching any known acronym (exact case, whole word).
_pattern: Optional[re.Pattern[str]] = None

# Characters treated as "word characters" for boundary purposes.
_WORD_CHARS = r"A-Za-z0-9_"


def load_acronyms(csv_path: str) -> int:
    """
    Load the acronym CSV into memory and compile the matching regex.

    Called ONCE from the FastAPI lifespan startup hook (see main.py).

    CSV format (tolerant):
      - Column 1: the acronym (exact case as it should appear in text).
      - Column 2: the definition / expansion.
      - Any additional columns are ignored.
      - A header row is auto-detected and skipped.
      - MULTIPLE ROWS with the SAME acronym are ALLOWED and EXPECTED: each row
        contributes an additional meaning. All are retained.

    Returns:
      The number of DISTINCT acronyms loaded (not the number of definitions).
    """
    global _definitions, _pattern

    definitions: dict[str, list[str]] = {}

    try:
        with open(csv_path, mode="r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)
    except FileNotFoundError:
        logger.warning("Acronym CSV not found at %s; terminology disabled.", csv_path)
        _definitions = {}
        _pattern = None
        return 0
    except Exception as exc:  # noqa: BLE001 - log and degrade gracefully
        logger.error("Failed to read acronym CSV %s: %s", csv_path, exc)
        _definitions = {}
        _pattern = None
        return 0

    if not rows:
        logger.warning("Acronym CSV %s is empty; terminology disabled.", csv_path)
        _definitions = {}
        _pattern = None
        return 0

    start_index = 1 if _looks_like_header(rows[0]) else 0

    total_definitions = 0
    for row in rows[start_index:]:
        if len(row) < 2:
            continue
        acronym = row[0].strip()
        definition = row[1].strip()
        if not acronym or not definition:
            continue

        # Ensure a list exists for this acronym, then APPEND the new meaning.
        bucket = definitions.setdefault(acronym, [])

        # Dedupe EXACT-duplicate definitions (same acronym + identical text),
        # which would otherwise show the model the same line twice. Distinct
        # meanings are all kept.
        if definition not in bucket:
            bucket.append(definition)
            total_definitions += 1

    _definitions = definitions
    _pattern = _build_pattern(definitions.keys())

    # Log both counts so multi-definition CSVs are obvious at startup.
    logger.info(
        "Loaded %d acronyms (%d total definitions) for terminology enrichment.",
        len(_definitions),
        total_definitions,
    )
    return len(_definitions)


def _looks_like_header(row: list[str]) -> bool:
    """
    Heuristic: decide whether the first CSV row is a header to skip.
    Conservative: when unsure, do NOT skip, so no real data is lost.
    """
    if not row:
        return False
    first = row[0].strip().lower()
    header_labels = {"acronym", "acronyms", "term", "abbreviation", "key"}
    return first in header_labels


def _build_pattern(acronyms) -> Optional[re.Pattern[str]]:
    """
    Compile a single regex matching ANY known acronym, exact case, whole word.
    Longest-first alternation; lookaround boundaries; no re.IGNORECASE.
    """
    escaped = [re.escape(a) for a in acronyms if a]
    if not escaped:
        return None

    escaped.sort(key=len, reverse=True)  # longest-match-first

    alternation = "|".join(escaped)
    pattern_str = rf"(?<![{_WORD_CHARS}])(?:{alternation})(?![{_WORD_CHARS}])"

    return re.compile(pattern_str)  # exact case (no IGNORECASE)


def find_matches(text: str) -> dict[str, list[str]]:
    """
    Return an ordered dict {acronym: [definition, ...]} for every DISTINCT
    acronym found in `text`, using exact-case, whole-word matching.

    - Order preserved as first-seen in the text.
    - Each acronym appears once as a key; its value is the FULL list of ALL
      known definitions for that acronym, so the caller can present every
      candidate meaning to the model.
    - Returns an empty dict if terminology is disabled or nothing matches.
    """
    if _pattern is None or not text:
        return {}

    found: dict[str, list[str]] = {}
    for match in _pattern.finditer(text):
        acronym = match.group(0)
        if acronym not in found:
            defs = _definitions.get(acronym)
            if defs:  # non-empty list
                # Copy so the caller cannot mutate module state.
                found[acronym] = list(defs)
    return found


def build_terminology_block(*texts: str) -> Optional[str]:
    """
    Find all matching acronyms across ALL provided text sources and build a
    single system-block string that includes EVERY known definition for each
    matched acronym.

    Output format when an acronym has ONE definition:
        - MoE: Measure of Effectiveness

    Output format when an acronym has MULTIPLE definitions (the LLM disambiguates
    from context):
        - MOE (multiple possible meanings; choose by context):
            1. Measure of Effectiveness
            2. Ministry of Education

    Returns:
      - The formatted multi-line string, or
      - None if no acronyms matched (caller then skips adding a system block).

    The caller (pipeline.py) wraps this string in a
    {"role": "system", "content": <block>} message. Keeping that wrapping out of
    this module preserves its purity (text in, text out).
    """
    combined: dict[str, list[str]] = {}
    for text in texts:
        if not text:
            continue
        for acronym, defs in find_matches(text).items():
            if acronym not in combined:
                combined[acronym] = defs

    if not combined:
        return None

    lines = ["Relevant wargame terminology:"]
    for acronym, defs in combined.items():
        if len(defs) == 1:
            # Single, unambiguous meaning.
            lines.append(f"- {acronym}: {defs[0]}")
        else:
            # Multiple meanings: present all, and explicitly instruct the model
            # that it must choose the correct one from context.
            lines.append(
                f"- {acronym} (multiple possible meanings; choose by context):"
            )
            for i, definition in enumerate(defs, start=1):
                lines.append(f"    {i}. {definition}")
    return "\n".join(lines)


def acronym_count() -> int:
    """Number of DISTINCT acronyms currently loaded (not definition count)."""
    return len(_definitions)


def definition_count() -> int:
    """Total number of definitions across all acronyms. Useful for /health."""
    return sum(len(v) for v in _definitions.values())