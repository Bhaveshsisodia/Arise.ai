
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
You are evaluating a Retrieval Augmented Generation (RAG) system.

You will receive:

QUESTION
GROUND TRUTH ANSWER
STUDENT ANSWER

Your task is to determine whether the STUDENT ANSWER correctly answers the QUESTION.

Rules:

1. The STUDENT ANSWER does NOT need to exactly match the wording of the GROUND TRUTH.

2. Paraphrases, equivalent wording, synonyms, abbreviations, and alternate descriptions should be treated as correct.

3. Additional information is acceptable if it is factually correct and does not contradict the ground truth.

4. Minor differences in wording should NOT affect correctness.

5. Only mark incorrect if:
   - the answer is factually wrong
   - the answer contradicts the ground truth
   - the answer fails to answer the question

Output:
- explanation
- correct (true/false)
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
relevance_instructions = """You are a teacher grading a quiz. You will be given a QUESTION and a STUDENT ANSWER. Here is the grade criteria to follow:
(1) Ensure the STUDENT ANSWER is concise and relevant to the QUESTION
(2) Ensure the STUDENT ANSWER helps to answer the QUESTION

Relevance:
A relevance value of True means that the student's answer meets all of the criteria.
A relevance value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""



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

grounded_instructions = """You are a teacher grading a quiz. You will be given FACTS and a STUDENT ANSWER. Here is the grade criteria to follow:
(1) Ensure the STUDENT ANSWER is grounded in the FACTS. (2) Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the FACTS.

Grounded:
A grounded value of True means that the student's answer meets all of the criteria.
A grounded value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""


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

retrieval_relevance_instructions = """You are a teacher grading a quiz. You will be given a QUESTION and a set of FACTS provided by the student. Here is the grade criteria to follow:
(1) You goal is to identify FACTS that are completely unrelated to the QUESTION
(2) If the facts contain ANY keywords or semantic meaning related to the question, consider them relevant
(3) It is OK if the facts have SOME information that is unrelated to the question as long as (2) is met

Relevance:
A relevance value of True means that the FACTS contain ANY keywords or semantic meaning related to the QUESTION and are therefore relevant.
A relevance value of False means that the FACTS are completely unrelated to the QUESTION.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""


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



