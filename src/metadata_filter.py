"""
metadata_filter.py — LLM-based metadata filter extraction.

Converts a natural language query into a MongoDB $vectorSearch filter dict.

Key facts about your Atlas vector index:
  Fields declared as "type: filter" in your index:
    ✅ metadata.discom
    ✅ metadata.filing_year
    ✅ metadata.section_heading
    ✅ metadata.section_num
    ✅ metadata.main_section
    ✅ metadata.document_type
    ✅ metadata.commission
    ❌ metadata.cost_head  ← NOT in index, removed from filtering

  $vectorSearch only filters on fields declared in the index.
  Filtering on an undeclared field = 0 results silently.

Usage:
    from src.metadata_filter import get_query_filters
    mongo_filter = get_query_filters(query, llm, collection)
"""

import json
import re
from typing import Optional

from pydantic import BaseModel, ConfigDict


# ============================================================
# FILTER CACHE
# get_query_filters() makes one LLM call per query.
# In combined_retrieve(), 4 sub-retrievers call it for the
# same query → 4 identical LLM calls wasted.
# Cache by query string → call LLM only once per unique query.
# ============================================================

_filter_cache: dict = {}

def clear_filter_cache():
    """Call between sessions to reset cache."""
    _filter_cache.clear()


# ============================================================
# PYDANTIC MODEL
# ONLY fields that exist in your Atlas vector index.
# cost_head removed — not in index, causes silent 0 results.
# ============================================================

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


# ============================================================
# DYNAMIC CATALOG
# Reads distinct values live from MongoDB.
# Only includes fields that:
#   1. Are declared in your Atlas vector index (can be filtered)
#   2. Have more than 1 distinct value (otherwise filter is useless)
# ============================================================

# Fields that exist in your Atlas vector index as "type: filter"
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
    Builds a catalog of filterable metadata values from MongoDB.
    Only queries fields that are actually indexed for filtering.
    """
    catalog = {}
    for field in _INDEXED_FILTER_FIELDS:
        values = collection.distinct(f"metadata.{field}")
        # Skip fields with 0 or 1 value — filtering on them narrows nothing
        if len(values) > 1:
            catalog[field] = sorted(values)  # sorted for consistent prompt
    return catalog


# ============================================================
# PROMPT
# ============================================================

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
6. Return ONLY valid JSON — no markdown, no explanation.

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


# ============================================================
# JSON PARSER
# ============================================================

def parse_json_response(text: str) -> dict:
    if not text:
        raise ValueError("Empty LLM response")

    if isinstance(text, dict):
        return text

    if isinstance(text, (list, tuple)):
        raise ValueError("LLM returned an unexpected non-object payload")

    text = str(text).strip()
    if not text:
        raise ValueError("Empty LLM response")

    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found in LLM response:\n{text}")
    return json.loads(match.group())


def normalize_filter_payload(payload: dict) -> dict:
    """Coerce empty/partial LLM output into a valid QueryFilters payload."""
    if not isinstance(payload, dict):
        raise ValueError("LLM returned a non-dict filter payload")

    normalized = dict(payload)
    normalized.setdefault("semantic_query", None)
    return normalized


# ============================================================
# MONGO FILTER BUILDER
# Converts Pydantic model → MongoDB filter dict.
# Only includes fields that are non-None.
# ============================================================

def build_mongo_filter(parsed: QueryFilters) -> dict:
    """
    Converts parsed QueryFilters → MongoDB filter dict.

    All fields here are guaranteed to exist in your Atlas
    vector index as 'type: filter' — safe to use in $vectorSearch.
    """
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


# ============================================================
# PUBLIC FUNCTION
# ============================================================

def get_query_filters(user_query: str, llm, collection) -> dict:
    """
    Extracts MongoDB filter dict from a natural language query.

    Pipeline:
      1. Check cache — return immediately if query seen before
      2. Build dynamic catalog from MongoDB (indexed fields only)
      3. Ask LLM to extract structured filters
      4. Validate with Pydantic
      5. Convert to MongoDB filter dict
      6. Cache result for future calls

    Args:
        user_query: natural language question
        llm:        LangChain LLM instance
        collection: pymongo Collection

    Returns:
        dict — MongoDB filter ready for $vectorSearch or find()
        Empty dict {} means no filter — full collection search.
    """
    # Check cache first
    if user_query in _filter_cache:
        return _filter_cache[user_query]

    # Build catalog, extract filters, validate
    catalog = build_dynamic_catalog(collection)
    prompt = build_filter_prompt(user_query, catalog)
    response = llm.invoke(prompt)

    try:
        raw_response = getattr(response, "content", response)
        parsed_payload = normalize_filter_payload(parse_json_response(raw_response))
        if isinstance(parsed_payload, dict) and parsed_payload.get("error"):
            raise ValueError(parsed_payload["error"])

        parsed = QueryFilters(**parsed_payload)
        result = build_mongo_filter(parsed)
        if not result and not parsed.semantic_query:
            result = {}
    except Exception as e:
        print(f"⚠️  Filter extraction failed ({e}) — using no filter")
        result = {}

    # Cache and return
    _filter_cache[user_query] = result
    return result
