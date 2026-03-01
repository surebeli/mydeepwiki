#!/usr/bin/env python3
"""
CodeWiki MCP Server
Exposes 4 tools for interacting with the local CodeWiki (deepwiki-open) service.

Supports:
  - stdio transport (for Claude Desktop / MCP clients)
  - HTTP transport on 0.0.0.0:8002 (for remote/network access)

Usage:
  # stdio mode (default - for MCP client config)
  python mcp_server.py

  # HTTP mode
  python mcp_server.py --transport http --port 8002
"""

import asyncio
import json
import sys
import argparse
import logging
from typing import Optional, Any
import httpx
import websockets

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("codewiki-mcp")

BACKEND_URL = "http://localhost:8001"
BACKEND_WS  = "ws://localhost:8001"

# Default models per provider
PROVIDER_DEFAULTS = {
    "ollama":      "qwen3:8b",
    "dashscope":   "qwen-plus",
    "kimi-coding": "k2p5",
    "openai":      "gpt-4o",
    "openrouter":  "openai/gpt-4o",
    "google":      "gemini-2.5-flash",
    "bedrock":     "anthropic.claude-3-sonnet-20240229-v1:0",
    "azure":       "gpt-4o",
}

# ──────────────────────────────────────────────
# Helper: HTTP client
# ──────────────────────────────────────────────
async def _get(path: str, params: dict = None) -> Any:
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.get(f"{BACKEND_URL}{path}", params=params)
        r.raise_for_status()
        return r.json()

async def _post(path: str, payload: dict) -> Any:
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(f"{BACKEND_URL}{path}", json=payload)
        r.raise_for_status()
        return r.json()

# ──────────────────────────────────────────────
# Tool 1: list_wikis
# ──────────────────────────────────────────────
async def list_wikis() -> str:
    """
    List all indexed wikis / processed projects in the CodeWiki service.

    Returns a JSON array with fields: id, owner, repo, name, repo_type, language, submittedAt.
    """
    try:
        data = await _get("/api/processed_projects")
        if not data:
            return "No wikis indexed yet. Generate a wiki first by visiting http://localhost:3000"
        lines = []
        for p in data:
            lines.append(
                f"• {p.get('name', p.get('repo', '?'))} "
                f"[{p.get('repo_type','?')}] lang={p.get('language','en')}"
            )
        return "Indexed wikis:\n" + "\n".join(lines)
    except Exception as e:
        return f"Error listing wikis: {e}"

# ──────────────────────────────────────────────
# Tool 2: get_wiki_content
# ──────────────────────────────────────────────
async def get_wiki_content(
    owner: str,
    repo: str,
    repo_type: str = "local",
    language: str = "en",
) -> str:
    """
    Retrieve the cached wiki structure/content for a given repository.

    Args:
        owner:     Repository owner (use 'local' for local paths).
        repo:      Repository name or last path segment.
        repo_type: 'github' | 'gitlab' | 'bitbucket' | 'local'. Default: 'local'.
        language:  Language code, e.g. 'en', 'zh'. Default: 'en'.

    Returns:
        Markdown-formatted wiki structure or a message if not cached.
    """
    try:
        params = dict(owner=owner, repo=repo, repo_type=repo_type, language=language)
        data = await _get("/api/wiki_cache", params)
        if not data:
            return (
                f"No cached wiki found for {owner}/{repo} ({repo_type}). "
                "Open http://localhost:3000 and generate the wiki first."
            )
        pages = data.get("pages", [])
        if not pages:
            return "Wiki cached but contains no pages."

        lines = [f"# Wiki: {owner}/{repo}", ""]
        for page in pages:
            title = page.get("title", "Untitled")
            content = page.get("content", "")
            lines.append(f"## {title}")
            lines.append(content[:2000])   # trim very large pages
            lines.append("")
        return "\n".join(lines)
    except Exception as e:
        return f"Error fetching wiki: {e}"

# ──────────────────────────────────────────────
# Tool 3: ask_codebase
# ──────────────────────────────────────────────
async def ask_codebase(
    question: str,
    repo_url_or_path: str,
    repo_type: str = "local",
    provider: str = "ollama",
    model: str = "",
    language: str = "en",
) -> str:
    """
    Ask a natural-language question about an indexed codebase via RAG chat.

    Args:
        question:          Your question about the code.
        repo_url_or_path:  Local path (e.g. /home/user/myproject) or GitHub URL.
        repo_type:         'local' | 'github' | 'gitlab' | 'bitbucket'. Default: 'local'.
        provider:          LLM provider. Default: 'ollama'.
        model:             Model name. Empty string uses provider default.
        language:          Response language. Default: 'en'.

    Returns:
        The AI answer as a string.
    """
    # Resolve default model for the provider
    if not model:
        model = PROVIDER_DEFAULTS.get(provider, "qwen3:8b")

    payload = {
        "repo_url": repo_url_or_path,
        "messages": [{"role": "user", "content": question}],
        "type": repo_type,
        "provider": provider,
        "model": model,
        "language": language,
    }

    try:
        ws_url = f"{BACKEND_WS}/ws/chat/completions"
        chunks = []
        try:
            async with websockets.connect(ws_url, open_timeout=10) as ws:
                await ws.send(json.dumps(payload))
                async for message in ws:
                    data = json.loads(message)
                    if data.get("type") == "content":
                        chunks.append(data.get("content", ""))
                    elif data.get("type") == "done":
                        break
                    elif data.get("type") == "error":
                        return f"Backend error: {data.get('message','unknown error')}"
            return "".join(chunks) if chunks else "No response received."
        except Exception:
            # Fallback to HTTP streaming endpoint
            r_text = []
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(
                    f"{BACKEND_URL}/chat/completions/stream",
                    json=payload
                )
                r_text.append(resp.text[:4000])
            return "".join(r_text) or "No response."
    except Exception as e:
        return f"Error asking codebase: {e}"


# ──────────────────────────────────────────────
# Host sampling: retrieve RAG context then ask
# the host client's LLM via MCP sampling
# ──────────────────────────────────────────────
async def _retrieve_context(
    question: str,
    repo_url_or_path: str,
    repo_type: str = "local",
    language: str = "en",
) -> str:
    """Retrieve RAG context from the backend without LLM generation."""
    payload = {
        "repo_url": repo_url_or_path,
        "messages": [{"role": "user", "content": question}],
        "type": repo_type,
        "language": language,
    }
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{BACKEND_URL}/chat/retrieve_context",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("context", "")
    except Exception as e:
        logger.warning(f"Failed to retrieve context: {e}")
        return ""


async def ask_codebase_via_host(
    question: str,
    repo_url_or_path: str,
    repo_type: str,
    language: str,
    ctx,
) -> str:
    """Use host's LLM via MCP sampling to answer a codebase question."""
    # Step 1: Get RAG context from backend
    context = await _retrieve_context(question, repo_url_or_path, repo_type, language)

    if not context:
        prompt = (
            f"The user is asking about the codebase at '{repo_url_or_path}'.\n\n"
            f"Question: {question}\n\n"
            "Note: No RAG context was available. Please answer based on your general knowledge, "
            "or suggest the user try a different provider (e.g. ollama, dashscope, kimi-coding)."
        )
    else:
        prompt = (
            f"Based on the following code context retrieved from the repository "
            f"at '{repo_url_or_path}', answer the user's question.\n\n"
            f"<code_context>\n{context}\n</code_context>\n\n"
            f"Question: {question}"
        )

    # Step 2: Ask host LLM via MCP sampling
    try:
        result = await ctx.sample(
            messages=prompt,
            system_prompt=(
                "You are a helpful code assistant. Answer questions based on the "
                "provided code context accurately and concisely."
            ),
            max_tokens=4096,
        )
        return result.text
    except Exception as e:
        logger.error(f"MCP sampling failed: {e}")
        return (
            f"MCP sampling failed: {e}\n\n"
            "The host client may not support MCP sampling. "
            "Try using a different provider: ollama, dashscope, or kimi-coding."
        )


# ──────────────────────────────────────────────
# Tool 4: inspect_local_repo
# ──────────────────────────────────────────────
async def inspect_local_repo(path: str) -> str:
    """
    Get the file tree and README of a local repository path.

    Args:
        path: Absolute local path to the repository (e.g. /home/user/myproject).

    Returns:
        JSON with 'file_tree' and 'readme' fields.
    """
    try:
        data = await _get("/local_repo/structure", {"path": path})
        file_tree = data.get("file_tree", [])
        readme    = data.get("readme", "No README found.")

        tree_str = "\n".join(file_tree[:50]) if isinstance(file_tree, list) else str(file_tree)
        return (
            f"## File Tree ({path})\n```\n{tree_str}\n```\n\n"
            f"## README\n{readme[:2000]}"
        )
    except Exception as e:
        return f"Error inspecting repo: {e}"


# ──────────────────────────────────────────────
# MCP Server assembly
# ──────────────────────────────────────────────
def build_mcp_server():
    try:
        from fastmcp import FastMCP
        from fastmcp.server.context import Context
        mcp = FastMCP(
            name="codewiki",
            instructions=(
                "CodeWiki MCP — provides tools to query a locally running deepwiki-open "
                "service. Supports multiple LLM providers: ollama (local), dashscope "
                "(Alibaba Bailian), kimi-coding (Kimi K2.5), and host (use the host "
                "client's own LLM via MCP sampling). Useful for generating, listing, and "
                "querying documentation wikis for local code repositories."
            ),
        )

        @mcp.tool()
        async def list_wikis_tool() -> str:
            """List all wikis that have been generated and cached by the CodeWiki service."""
            return await list_wikis()

        @mcp.tool()
        async def get_wiki_content_tool(
            owner: str,
            repo: str,
            repo_type: str = "local",
            language: str = "en",
        ) -> str:
            """Retrieve the full cached wiki content for a repository."""
            return await get_wiki_content(owner, repo, repo_type, language)

        @mcp.tool()
        async def ask_codebase_tool(
            question: str,
            repo_url_or_path: str,
            repo_type: str = "local",
            provider: str = "ollama",
            model: str = "",
            ctx: Context = None,
        ) -> str:
            """Ask a question about a codebase using RAG over the indexed repository.

            Args:
                question:          Your question about the code.
                repo_url_or_path:  Local path or GitHub/GitLab/Bitbucket URL.
                repo_type:         'local' | 'github' | 'gitlab' | 'bitbucket'.
                provider:          LLM provider to use. Options:
                                   - ollama: Local Ollama (default, model: qwen3:8b)
                                   - dashscope: Alibaba DashScope (model: qwen-plus)
                                   - kimi-coding: Kimi Code Plan K2.5 (model: k2p5)
                                   - host: Use the host client's LLM via MCP sampling
                                   - openai, openrouter, google, bedrock, azure also supported
                model:             Model name. Leave empty for provider default.
            """
            if provider == "host":
                return await ask_codebase_via_host(
                    question, repo_url_or_path, repo_type, "en", ctx
                )
            return await ask_codebase(
                question, repo_url_or_path, repo_type,
                provider=provider, model=model,
            )

        @mcp.tool()
        async def inspect_local_repo_tool(path: str) -> str:
            """Get the file tree and README of a local repository path."""
            return await inspect_local_repo(path)

        return mcp

    except ImportError:
        logger.error("fastmcp not installed. Run: uv pip install fastmcp")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="CodeWiki MCP Server")
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8002)
    args = parser.parse_args()

    mcp = build_mcp_server()

    if args.transport == "http":
        logger.info(f"Starting CodeWiki MCP HTTP server on {args.host}:{args.port}")
        mcp.run(transport="sse", host=args.host, port=args.port)
    else:
        logger.info("Starting CodeWiki MCP stdio server")
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
