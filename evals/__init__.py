"""DeepDive evaluation harness (course Module 4: evals + error analysis).

Three metric families, each tied to a lecture:
  * retrieval   — precision/recall/F1/hit-rate of retrieved source domains vs a gold set
  * extraction  — schema-valid rate + answer accuracy on fixed source text (offline)
  * citation    — faithfulness via a substring baseline AND a 5-criterion LLM-as-judge,
                  plus judge-vs-human agreement (the "evaluate your evaluator" loop)

Run:  python -m evals.runner --suite all
"""
