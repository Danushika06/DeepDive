"""LLM-as-judge for citation faithfulness (course Module 2.4 / 4).

Returns a :class:`JudgeVerdict` of FIVE binary (0/1) criteria summed to 0-5 — NOT a raw
1-5 Likert score (which the course flags as unreliable). Fields are presented in a
fixed labeled order (CLAIM / QUOTE / SOURCE) and judged independently to blunt position
bias. ``faithful`` = score >= 4.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from deepdive.schemas.extraction import JudgeVerdict

JUDGE_SYSTEM = (
    "You are a strict citation auditor. Judge whether a CLAIM is faithfully supported by a "
    "QUOTE drawn from a SOURCE. Score each of five criteria independently as 0 or 1. Judge only "
    "on content; the order of fields carries no meaning. Be conservative: when unsure, score 0.\n"
    "Criteria:\n"
    "  quote_present       = 1 if the quote appears (verbatim or near-verbatim) in the source.\n"
    "  quote_supports_claim= 1 if the quote alone supports the claim.\n"
    "  no_overreach        = 1 if the claim asserts no more than the source warrants.\n"
    "  correct_attribution = 1 if the quote genuinely comes from this source.\n"
    "  no_fabrication      = 1 if nothing in the claim is invented relative to the source."
)


def judge_citation(llm, claim: str, quote: str, source_text: str) -> JudgeVerdict:
    structured = llm.with_structured_output(JudgeVerdict, include_raw=True)
    human = (
        f"CLAIM:\n{claim}\n\n"
        f"QUOTE:\n{quote}\n\n"
        f"SOURCE:\n{source_text}\n\n"
        "Return the five binary criteria and a one-sentence rationale."
    )
    result = structured.invoke([SystemMessage(JUDGE_SYSTEM), HumanMessage(human)])
    verdict: JudgeVerdict | None = result.get("parsed")
    return verdict or JudgeVerdict(
        quote_present=0,
        quote_supports_claim=0,
        no_overreach=0,
        correct_attribution=0,
        no_fabrication=0,
        rationale="judge produced no parseable verdict",
    )
