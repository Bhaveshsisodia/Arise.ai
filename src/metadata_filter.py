"""
metadata_filter.py — LLM-based metadata filter extraction.

Converts a natural language query into a MongoDB filter dict
by asking the LLM to extract structured metadata from the query
and validating it against the live catalog of values in the DB.

Usage:
    from src.filters import get_query_filters
    mongo_filter = get_query_filters(query, llm, collection)
"""

import json
import re
from typing import Optional

from pydantic import BaseModel


# ============================================================
# PYDANTIC MODEL
# Defines which metadata fields can be filtered.
# Add new fields here if your schema grows.
# ============================================================

class QueryFilters(BaseModel):
    semantic_query:  str
    cost_head:       Optional[str] = None
    document_type:   Optional[str] = None
    discom:          Optional[str] = None
    filing_year:     Optional[str] = None
    commission:      Optional[str] = None
    section_num:     Optional[str] = None
    main_section:    Optional[str] = None
    section_heading: Optional[str] = None


# ============================================================
# DYNAMIC CATALOG
# Reads distinct metadata values live from MongoDB.
# Only keeps fields that have >1 unique value (otherwise filtering
# on them doesn't narrow results at all).
# ============================================================

def build_dynamic_catalog(collection) -> dict:
    fields = [
        "document_type", "discom", "section_num",
        "main_section", "filing_year", "commission",
        "cost_head", "section_heading",
    ]
    catalog = {}
    for field in fields:
        values = collection.distinct(f"metadata.{field}")
        if len(values) > 1:
            catalog[field] = values
    return catalog


# ============================================================
# PROMPT BUILDER
# ============================================================

def build_filter_prompt(user_query: str, metadata_catalog: dict) -> str:
    return f"""
You are an expert query understanding engine for a Power Sector Regulatory Assistant.

Available metadata values:

{json.dumps(metadata_catalog, indent=2)}

Your task:

1. Understand the user's intent.
2. Extract ONLY metadata explicitly mentioned or strongly implied.
3. Never guess metadata.
4. If metadata is not present, do not include it.
5. Rewrite the question into a concise semantic search query.
6. Return ONLY valid JSON.
7. Do not add explanations.
8. Do not wrap output in markdown.

Metadata meanings:

- cost_head:
  Financial category such as employee_expense, depreciation, roe, interest, arr, capex.

- document_type:
  petition, tariff_order, regulation, trueup_order etc.

- filing_year:
  MUST exactly match one of the values in the available metadata above.
  Common formats: "FY24", "FY25", "FY26" or "FY 2024-25" etc.
  Always pick from the catalog values, never invent a value.

- discom:
  Utility name such as JUSNL.

- commission:
  Regulatory commission such as JSERC.

- section_num:
  Section identifier such as 1.1, 2.1, 5.5.

- main_section:
  Top-level chapter such as
  "1. Introduction",
  "2. Regulatory Framework",
  "5. ARR and Tariff Proposal".

- section_heading:
  Complete section title such as:
  "1.4. Rationale for filing of Instant Petition"
  "5.5. Operation and Maintenance Expenses"
  "3.8. Return on Equity"

Examples:

Question: What does section 2.1 say?
Output:
{{
  "semantic_query": "section 2.1",
  "section_num": "2.1"
}}

Question: What is JUSNL?
Output:
{{
  "semantic_query": "JUSNL profile background"
}}

Question: Show me the rationale for filing of instant petition
Output:
{{
  "semantic_query": "rationale for filing instant petition",
  "section_heading": "1.4. Rationale for filing of Instant Petition"
}}

{user_query}
"""


# ============================================================
# JSON PARSER
# ============================================================

def parse_json_response(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found in LLM response:\n{text}")
    return json.loads(match.group())


# ============================================================
# MONGO FILTER BUILDER
# Converts ParsedFilters → MongoDB filter dict.
# Skips fields whose value is "other" (not meaningful for filtering).
# ============================================================

def build_mongo_filter(parsed: QueryFilters) -> dict:
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
# MAIN PUBLIC FUNCTION
# ============================================================

def get_query_filters(user_query: str, llm, collection) -> dict:
    """
    Full pipeline:
      1. Build dynamic catalog from MongoDB
      2. Ask LLM to extract structured filters
      3. Validate with Pydantic
      4. Convert to MongoDB filter dict

    Returns:
        dict — MongoDB filter ready for $vectorSearch or find()
    """
    catalog  = build_dynamic_catalog(collection)
    prompt   = build_filter_prompt(user_query, catalog)
    response = llm.invoke(prompt)
    parsed   = QueryFilters(**parse_json_response(response.content))
    return build_mongo_filter(parsed)