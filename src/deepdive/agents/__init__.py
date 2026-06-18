"""The LangGraph multi-agent orchestration.

Custom ``StateGraph`` with hand-written nodes (planner / searcher / retriever /
extractor / critic / synthesizer) over one shared state. Each node maps to a course
design pattern — see ARCHITECTURE.md. ReAct lives inside the planner node
(reason -> emit query plan -> observe on replan), not a prebuilt agent wrapper.
"""
