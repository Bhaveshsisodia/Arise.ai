"""
metadata_filter.py - LLM-based metadata filter extraction.

Converts a natural language query into a MongoDB $vectorSearch filter dict.
"""

import json
import logging
import re
from typing import Optional

from pydantic import BaseModel, ConfigDict

from src.exception.custom_exception import DatabaseError, ValidationError
from src.exception.error_utils import raise_with_context


logger = logging.getLogger("arise.metadata_filter")

_filter_cache: dict = {}


def clear_filter_cache():
    """Call between sessions to reset cache."""
    _filter_cache.clear()


class QueryFilters(BaseModel):
    model_config = ConfigDict(extra="ignore")

    semantic_query: Optional[str] = None
    document_type: Optional[str] = None
    discom: Optional[str] = None
    filing_year: Optional[str] = None
    commission: Optional[str] = None
    section_num: Optional[str] = None
    main_section: Optional[str] = None
    section_heading: Optional[str] = None


_INDEXED_FILTER_FIELDS = [
    "document_type",
    "discom",
    "filing_year",
    "section_heading",
    "section_num",
    "main_section",
    "commission",
]


def build_dynamic_catalog(collection) -> dict:
    """
    Build a catalog of filterable metadata values from MongoDB.

    Only queries fields that are actually indexed for filtering.
    """
    catalog = {}
    try:
        for field in _INDEXED_FILTER_FIELDS:
            values = collection.distinct(f"metadata.{field}")
            if len(values) > 1:
                catalog[field] = sorted(values)
        return catalog
    except Exception as exc:
        raise_with_context(
            DatabaseError,
            exc,
            "Failed to build metadata catalog from MongoDB",
        )


def build_filter_prompt(user_query: str, metadata_catalog: dict) -> str:
    return f"""
You are an expert query understanding engine for a Power Sector Regulatory Assistant.

Available metadata values in the database:

{json.dumps(metadata_catalog, indent=2)}

Your task:

1. Understand the user's intent.
2. Extract ONLY metadata explicitly mentioned or strongly implied by the query.
3. Never guess metadata not present in the query.
4. If metadata is not present, do not include it.
5. Rewrite the question into a concise semantic search query.
6. Return ONLY valid JSON - no markdown, no explanation.

Field meanings:

- document_type:
  Type of regulatory document: petition, tariff_order, regulation, trueup_order etc.

- filing_year:
  MUST exactly match one of the values in the catalog above.
  Common formats: "FY24", "FY25", "FY 2024-25" etc.
  Never invent a value not in the catalog.

- discom:
  Utility name e.g. JUSNL. Only include if explicitly mentioned.

- commission:
  Regulatory body e.g. JSERC. Only include if explicitly mentioned.

- section_num:
  Section identifier e.g. "1.1", "2.1", "5.5".
  Only include if user explicitly references a section number.

- main_section:
  Top-level chapter e.g. "1. Introduction", "5. ARR and Tariff Proposal".
  Only include if user explicitly references a chapter.

- section_heading:
  Complete section title e.g. "1.4. Rationale for filing of Instant Petition".
  Only include if user asks about a specific named section.

Examples:

Query: What does section 2.1 say?
Output:
{{"semantic_query": "section 2.1 content", "section_num": "2.1"}}

Query: What is JUSNL?
Output:
{{"semantic_query": "JUSNL profile background"}}

Query: Show me the rationale for filing of instant petition
Output:
{{"semantic_query": "rationale filing instant petition", "section_heading": "1.4. Rationale for filing of Instant Petition"}}

Query: What ARR has JUSNL projected for FY 2025-26?
Output:
{{"semantic_query": "ARR projected FY 2025-26", "discom": "JUSNL"}}

Query: {user_query}
Output:"""


def parse_json_response(text: str) -> dict:
    if not text:
        raise ValidationError("Empty LLM response while extracting metadata filters")

    if isinstance(text, dict):
        return text

    if isinstance(text, (list, tuple)):
        raise ValidationError("LLM returned an unexpected non-object metadata payload")

    text = str(text).strip()
    if not text:
        raise ValidationError("Empty LLM response while extracting metadata filters")

    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValidationError("No JSON found in LLM metadata filter response")
    return json.loads(match.group())


def normalize_filter_payload(payload: dict) -> dict:
    """Coerce empty or partial LLM output into a valid QueryFilters payload."""
    if not isinstance(payload, dict):
        raise ValidationError("LLM returned a non-dict metadata filter payload")

    normalized = dict(payload)
    normalized.setdefault("semantic_query", None)
    return normalized


def build_mongo_filter(parsed: QueryFilters) -> dict:
    """Convert parsed QueryFilters into a MongoDB filter dict."""
    mongo_filter = {}

    if parsed.document_type:
        mongo_filter["metadata.document_type"] = parsed.document_type

    if parsed.discom:
        mongo_filter["metadata.discom"] = parsed.discom

    if parsed.filing_year:
        mongo_filter["metadata.filing_year"] = parsed.filing_year

    if parsed.commission:
        mongo_filter["metadata.commission"] = parsed.commission

    if parsed.section_num:
        mongo_filter["metadata.section_num"] = parsed.section_num

    if parsed.main_section:
        mongo_filter["metadata.main_section"] = parsed.main_section

    if parsed.section_heading:
        mongo_filter["metadata.section_heading"] = parsed.section_heading

    return mongo_filter


def get_query_filters(user_query: str, llm, collection) -> dict:
    """
    Extract a MongoDB filter dict from a natural language query.

    Empty dict means no metadata filter should be applied.
    """
    if user_query in _filter_cache:
        return _filter_cache[user_query]

    catalog = build_dynamic_catalog(collection)
    prompt = build_filter_prompt(user_query, catalog)

    try:
        response = llm.invoke(prompt)
        raw_response = getattr(response, "content", response)
        parsed_payload = normalize_filter_payload(parse_json_response(raw_response))
        if isinstance(parsed_payload, dict) and parsed_payload.get("error"):
            raise ValidationError(str(parsed_payload["error"]))

        parsed = QueryFilters(**parsed_payload)
        result = build_mongo_filter(parsed)
        if not result and not parsed.semantic_query:
            result = {}
    except Exception as exc:
        logger.warning("Filter extraction failed; using no filter: %s", exc)
        result = {}

    _filter_cache[user_query] = result
    return result
