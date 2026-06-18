"""DeepDive — Agentic Research Synthesis Assistant.

A LangGraph multi-agent pipeline that decomposes a research question, searches the
live web, builds a FAISS index (RAG), extracts strict-JSON findings, flags
cross-source conflicts, and synthesizes a citation-backed report.

Maps directly onto the four agentic design patterns from DeepLearning.AI's
"Building Agentic AI" course (Andrew Ng): Planning, Tool use, Reflection, Multi-agent
— plus the Module-4 eval discipline. See ARCHITECTURE.md for the file-by-file map.
"""

__version__ = "0.1.0"
