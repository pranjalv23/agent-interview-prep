"""
Codebase tools for the interview prep agent.

The GitHub fetch is handled by the shared MCP tool server (fetch_github_repo tool),
called via the agent's existing MCP connection in app.py.
This module only provides the agent-facing tool for reading the stored codebase.
"""
from __future__ import annotations

import logging

from langchain_core.tools import tool

from database.mongo import MongoDB

logger = logging.getLogger("agent_interview_prep.codebase_parser")


@tool
async def analyze_codebase(session_id: str) -> str:
    """Retrieve the codebase that was linked for this session and return a
    structured summary of its files and content for codebase interview questioning.

    Args:
        session_id: The current session ID (from context).

    Returns a formatted string with repo metadata, file tree, and key file contents.
    Call this ONCE at the start of a codebase interview session.
    """
    doc = await MongoDB.get_codebase(session_id)
    if not doc:
        return (
            "No codebase has been linked for this session. "
            "Ask the user to share a GitHub repository URL so you can fetch it."
        )

    parts = [
        f"Repository: {doc.get('owner', '?')}/{doc.get('repo_name', '?')}",
        f"URL: {doc.get('repo_url', '?')}",
        f"Language: {doc.get('language', 'unknown')}",
    ]
    if doc.get("description"):
        parts.append(f"Description: {doc['description']}")
    parts.append(f"Total files: {doc.get('total_files', '?')}")

    file_tree: list[str] = doc.get("file_tree", [])
    if file_tree:
        parts.append("\nFile tree:")
        parts.extend(f"  {p}" for p in file_tree[:80])

    key_files: list[dict[str, str]] = doc.get("key_files", [])
    if key_files:
        parts.append(f"\n\nKey files ({len(key_files)} fetched):\n")
        for kf in key_files:
            parts.append(f"--- {kf['path']} ---")
            parts.append(kf["content"])
            parts.append("")

    return "\n".join(parts)
