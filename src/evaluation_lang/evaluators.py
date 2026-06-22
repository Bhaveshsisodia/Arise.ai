
from typing_extensions import Annotated, TypedDict
import json
import re


class CorrectnessGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    correct: Annotated[bool, ..., "True if the answer is correct, False otherwise."]

    # Grade prompt
# ============================================================
# Evaluation Instructions
# ============================================================

correctness_instructions = """
You are an expert evaluator for Indian power sector regulatory documents.

You will be given:

1. QUESTION
2. GROUND TRUTH ANSWER
3. STUDENT ANSWER

Evaluate whether the STUDENT ANSWER is factually correct relative to the GROUND TRUTH.

Rules:

- Focus ONLY on factual correctness.
- Ignore writing style, grammar, formatting, and verbosity.
- Additional information is acceptable if factually correct.
- Check carefully:
  - Numerical values
  - Financial years (FY)
  - ARR values
  - Tariff values
  - Employee expenses
  - A&G expenses
  - Regulatory references
  - Company/entity names
  - Petition details

Scoring Rubric:

5 = Completely correct

4 = Mostly correct with minor omissions

3 = Partially correct

2 = Mostly incorrect

1 = Completely incorrect

Return:
1. explanation
2. score

Think step-by-step before assigning a score.
"""





def _invoke_and_parse(llm, messages):
    """Invoke an LLM and return a parsed dict.

    If the LLM returns a dict-like object, return it directly. Otherwise
    attempt to extract a JSON object from the returned text and parse it.
    On failure return a dict with the raw text under the `raw` key.
    """
    resp = llm.invoke(messages)
    if isinstance(resp, dict):
        return resp

    if hasattr(resp, "content"):
        text = resp.content
    elif isinstance(resp, str):
        text = resp
    else:
        text = str(resp)

    m = re.search(r"(\{[\s\S]*\})", text)
    json_text = m.group(1) if m else text.strip()
    try:
        return json.loads(json_text)
    except Exception:
        return {"raw": text}


class CorrectnessEvaluator:

    def __init__(self, llm, model_name):
        self.model_name = model_name

        # Use raw llm and parse manually to avoid unsupported json_schema responses
        self.grader_llm = llm

    def __call__(self, inputs, outputs, reference_outputs):

        answers = f"""
QUESTION:
{inputs['question']}

GROUND TRUTH:
{reference_outputs['answer']}

STUDENT ANSWER:
{outputs['answer']}
"""

        grade = _invoke_and_parse(self.grader_llm, [
            {"role": "system", "content": correctness_instructions},
            {"role": "user", "content": answers},
        ])

        # Normalize grader output: prefer explicit fields, else try to infer
        if isinstance(grade, dict) and "correct" in grade:
            correct = bool(grade.get("correct"))
            explanation = grade.get("explanation", grade.get("raw", ""))
        else:
            raw = grade.get("raw", "") if isinstance(grade, dict) else str(grade)
            m = re.search(r"\b(true|false)\b", raw, re.I)
            if m:
                correct = m.group(1).lower() == "true"
            else:
                # default conservative: treat unknown as incorrect
                correct = False
            explanation = raw

        return {
            "key": f"correctness_{self.model_name}",
            "score": 1 if correct else 0,
            "comment": explanation,
        }



from typing_extensions import Annotated, TypedDict

# ============================================================
# Schema
# ============================================================

class RelevanceGrade(TypedDict):
    explanation: Annotated[
        str,
        ...,
        "Detailed reasoning for the assigned score"
    ]

    score: Annotated[
        int,
        ...,
        "Score between 1 and 5"
    ]


# ============================================================
# Instructions
# ============================================================

relevance_instructions = """
You are an expert evaluator for Question Answering systems.

You will be given:

1. QUESTION
2. STUDENT ANSWER

Your task is to evaluate how well the answer addresses the question.

Evaluation Rules:

- The answer should directly address the user's question.
- The answer should contain information relevant to the question.
- The answer should not be off-topic.
- The answer should not avoid answering the question.
- Completeness should be considered.
- Ignore grammar, formatting, and writing style.

Scoring Rubric:

5 = Fully answers the question directly and completely

4 = Mostly answers the question with minor missing details

3 = Partially answers the question

2 = Weakly related to the question

1 = Does not answer the question or is irrelevant

Return:
1. explanation
2. score

Think step-by-step before assigning a score.
"""


# ============================================================
# Evaluator
# ============================================================

class RelevanceEvaluator:

    def __init__(self, llm, model_name="judge"):

        self.model_name = model_name

        self.grader_llm = llm

    def __call__(self, inputs, outputs):

        prompt = f"""
QUESTION:
{inputs['question']}

STUDENT ANSWER:
{outputs['answer']}
"""

        grade = _invoke_and_parse(self.grader_llm, [
            {"role": "system", "content": relevance_instructions},
            {"role": "user", "content": prompt},
        ])

        # Normalize score and explanation
        raw = grade.get("raw", "") if isinstance(grade, dict) else str(grade)
        if isinstance(grade, dict) and "score" in grade:
            s = int(grade.get("score", 3))
        else:
            m = re.search(r"\b([1-5])\b", raw)
            s = int(m.group(1)) if m else 3
        explanation = grade.get("explanation", raw)

        return {
            "key": f"relevance_{self.model_name}",
            "score": s / 5.0,
            "comment": explanation,
        }

from typing_extensions import Annotated, TypedDict

# ============================================================
# Schema
# ============================================================

class GroundedGrade(TypedDict):
    explanation: Annotated[
        str,
        ...,
        "Detailed reasoning for the assigned score"
    ]

    score: Annotated[
        int,
        ...,
        "Score between 1 and 5"
    ]


# ============================================================
# Instructions
# ============================================================

grounded_instructions = """
You are an expert evaluator for Retrieval Augmented Generation (RAG) systems.

You will be given:

1. RETRIEVED CONTEXT
2. STUDENT ANSWER

Your task is to determine whether the answer is supported by the retrieved context.

Evaluation Rules:

- Every factual claim in the answer should be supported by the context.
- Do not use outside knowledge.
- If the answer contains information not found in the context, penalize it.
- If the answer contradicts the context, heavily penalize it.
- Minor rewording is acceptable.
- Focus on:
  - Numerical values
  - Financial years
  - ARR values
  - Employee expenses
  - Regulatory references
  - Entity names
  - Petition details

Scoring Rubric:

5 = Completely grounded in context

4 = Mostly grounded with minor unsupported details

3 = Partially grounded

2 = Significant unsupported information

1 = Hallucinated or contradicted by context

Return:
1. explanation
2. score

Think step-by-step before assigning a score.
"""


# ============================================================
# Evaluator
# ============================================================

class GroundednessEvaluator:

    def __init__(self, llm, model_name="judge"):

        self.model_name = model_name

        self.grader_llm = llm

    def __call__(self, inputs, outputs):

        if "documents" in outputs:
            context = "\n\n".join(
                getattr(doc, "page_content", str(doc))
                for doc in outputs["documents"]
            )
        else:
            context = outputs.get("context", "")

        prompt = f"""
RETRIEVED CONTEXT:
{context}

STUDENT ANSWER:
{outputs['answer']}
"""

        grade = _invoke_and_parse(self.grader_llm, [
            {"role": "system", "content": grounded_instructions},
            {"role": "user", "content": prompt},
        ])

        raw = grade.get("raw", "") if isinstance(grade, dict) else str(grade)
        if isinstance(grade, dict) and "score" in grade:
            s = int(grade.get("score", 3))
        else:
            m = re.search(r"\b([1-5])\b", raw)
            s = int(m.group(1)) if m else 3
        explanation = grade.get("explanation", raw)

        return {
            "key": f"groundedness_{self.model_name}",
            "score": s / 5.0,
            "comment": explanation,
        }


from typing_extensions import Annotated, TypedDict

# ============================================================
# Schema
# ============================================================

class RetrievalRelevanceGrade(TypedDict):
    explanation: Annotated[
        str,
        ...,
        "Detailed reasoning for the assigned score"
    ]

    score: Annotated[
        int,
        ...,
        "Score between 1 and 5"
    ]


# ============================================================
# Instructions
# ============================================================

retrieval_relevance_instructions = """
You are an expert evaluator for Retrieval Augmented Generation (RAG) systems.

You will be given:

1. QUESTION
2. RETRIEVED CONTEXT

Your task is to determine whether the retrieved context is relevant for answering the question.

Evaluation Rules:

- Focus only on retrieval quality.
- Determine whether the retrieved documents contain information useful for answering the question.
- Documents do not need to contain the exact answer.
- Documents may contain additional unrelated information.
- If the retrieved context contains keywords, concepts, entities, financial years, regulations, or semantic meaning related to the question, consider it relevant.
- Penalize retrieval only when most of the context is unrelated.

For regulatory tariff petitions pay special attention to:
- Utility names (JUSNL, JBVNL, etc.)
- ARR values
- Employee expenses
- O&M expenses
- Depreciation
- RoE
- Capital expenditure
- Tariff proposals
- Financial years
- Regulatory references

Scoring Rubric:

5 = Highly relevant; contains the information needed to answer

4 = Mostly relevant with minor irrelevant content

3 = Partially relevant

2 = Weakly relevant

1 = Completely irrelevant

Return:
1. explanation
2. score

Think step-by-step before assigning a score.
"""


# ============================================================
# Evaluator
# ============================================================

class RetrievalRelevanceEvaluator:

    def __init__(self, llm, model_name="judge"):

        self.model_name = model_name

        self.grader_llm = llm

    def __call__(self, inputs, outputs):

        context = "\n\n".join(
            getattr(doc, "page_content", str(doc))
            for doc in outputs["documents"]
        )

        prompt = f"""
QUESTION:
{inputs['question']}

RETRIEVED CONTEXT:
{context}
"""

        grade = _invoke_and_parse(self.grader_llm, [
            {"role": "system", "content": retrieval_relevance_instructions},
            {"role": "user", "content": prompt},
        ])

        raw = grade.get("raw", "") if isinstance(grade, dict) else str(grade)
        if isinstance(grade, dict) and "score" in grade:
            s = int(grade.get("score", 3))
        else:
            m = re.search(r"\b([1-5])\b", raw)
            s = int(m.group(1)) if m else 3
        explanation = grade.get("explanation", raw)

        return {
            "key": f"retrieval_relevance_{self.model_name}",
            "score": s / 5.0,
            "comment": explanation,
        }



