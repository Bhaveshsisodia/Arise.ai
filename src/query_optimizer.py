
def hyde_rewrite(query, llm):
    """
    Step 1: Generate a hypothetical answer.
    We only use this for SEARCHING — not as the final answer.
    """
    prompt = f"""You are an expert in Indian power sector regulation.

Write a detailed, factual-sounding passage that would answer this question.
Use domain-specific vocabulary: ₹ Cr, FY, O&M, ARR, JSERC, petitioner etc.
Write 3-4 sentences. Even if unsure, write something plausible.
Do NOT say "I don't know."

Question: {query}
Answer:"""

    response = llm.invoke(prompt)
    return response.content.strip()



def multi_query_rewrite(query, llm, num_variants=4):
    """
    Step 1: Generate N rephrasings of the same question.
    """
    prompt = f"""Generate {num_variants} different search queries for this question.
Use different vocabulary, synonyms, and perspectives.
Think about how the answer might be written in a regulatory document.

Original: {query}

Output one query per line, no numbering:"""

    response = llm.invoke(prompt)
    variants = [
        line.strip()
        for line in response.content.strip().split("\n")
        if line.strip()
    ][:num_variants]

    all_queries = [query] + variants

    return "\n".join(all_queries)   # always include original


def stepback_rewrite(query, llm):
    """
    Step 1: Generate a broader, more abstract version of the question.
    """
    prompt = f"""Given this specific question, generate a broader abstract question
that covers the general topic. The abstract question retrieves foundational
context that helps answer the specific one.

Example:
  Specific: "What did JUSNL project for employee expenses in FY27?"
  Abstract: "What are O&M expense components and regulatory norms for power utilities?"

Now do this for:
  Specific: {query}
  Abstract:"""

    response = llm.invoke(prompt)
    return response.content.strip()



