"""
novelWriter – AI Utilities
==========================

File History:
Created: 2026-05-08 [2.8b1] AI Utilities

This file is a part of novelWriter
Copyright (C) 2026 Veronica Berglyd Olsen and novelWriter contributors

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""  # noqa
from __future__ import annotations

import json
import logging
import re

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from urllib import error, request

from novelwriter import CONFIG
from novelwriter.common import appendIfSet, jsonEncode, safeExists, simplified
from novelwriter.constants import nwFiles
from novelwriter.error import logException

if TYPE_CHECKING:
    from novelwriter.core.project import NWProject

logger = logging.getLogger(__name__)

OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_OPENROUTER_MODEL = "openrouter/auto"
DEFAULT_CONTEXT_LIMIT = 12000
DEFAULT_RAG_MAX_DOCS = 6
DEFAULT_RAG_SNIPPET = 720

BUILTIN_GUARDRAILS = "\n".join([
    "Literary technique rules:",
    "- Preserve narrative perspective and tense unless explicitly asked to change them.",
    "- Favour concrete sensory detail over abstract explanation.",
    "- Use dialogue tags and action beats purposefully; avoid clutter.",
    "- Avoid clichés and stock phrasing; keep voice consistent.",
    "",
    "Prose quality and anti-AI rules:",
    "- Avoid generic filler, boilerplate, or apologetic language.",
    "- Do not mention being an AI, a model, or refer to system policies.",
    "- Avoid repetitive sentence starts; vary rhythm and structure.",
    "- Maintain natural phrasing suited to literary prose.",
    "",
    "Structural rules:",
    "- Preserve scene and chapter boundaries unless instructed otherwise.",
    "- Maintain continuity of names, timelines, and character motivations.",
    "- Keep changes local to the requested scope unless asked to refactor broadly.",
])

STOP_WORDS = {
    "about", "after", "again", "also", "and", "any", "are", "because", "been", "before",
    "being", "between", "both", "but", "by", "can", "could", "did", "does", "doing", "down",
    "each", "else", "for", "from", "had", "has", "have", "her", "here", "hers", "him", "his",
    "how", "into", "its", "just", "like", "more", "most", "not", "off", "only", "our", "out",
    "over", "she", "some", "that", "the", "their", "them", "then", "there", "these", "they",
    "this", "those", "too", "under", "was", "were", "what", "when", "where", "which", "who",
    "will", "with", "you", "your",
}


@dataclass(frozen=True, slots=True)
class AiSubAgent:
    """Core: Sub-agent descriptor."""

    name: str
    focus: str


SUB_AGENTS = (
    AiSubAgent(
        "Literary Technique",
        "Review literary technique and suggest improvements to craft, imagery, and pacing.",
    ),
    AiSubAgent(
        "Prose Quality",
        "Review prose quality for clarity, rhythm, and avoidance of AI-like phrasing.",
    ),
    AiSubAgent(
        "Structure",
        "Review structure for continuity, scene integrity, and narrative flow.",
    ),
)


def buildGuardrails(custom: str, includeBuiltIns: bool) -> str:
    """Build the combined guardrails prompt."""
    parts: list[str] = []
    if includeBuiltIns:
        appendIfSet(parts, BUILTIN_GUARDRAILS)
    appendIfSet(parts, custom.strip())
    return "\n\n".join(parts)


def extractKeywords(text: str, limit: int = 8) -> list[str]:
    """Extract keywords for retrieval and compression."""
    words = re.findall(r"[A-Za-z][A-Za-z']{2,}", text.lower())
    counts = Counter(word for word in words if word not in STOP_WORDS)
    return [word for word, _ in counts.most_common(limit)]


def scoreChunk(chunk: str, keywords: list[str]) -> int:
    """Score a text chunk based on keyword density."""
    chunkLower = chunk.lower()
    return sum(chunkLower.count(word) for word in keywords)


def compressContext(text: str, keywords: list[str], maxChars: int) -> str:
    """Compress text by selecting the most relevant chunks."""
    if maxChars <= 0 or len(text) <= maxChars:
        return text

    chunks = [part.strip() for part in re.split(r"\n{2,}", text) if part.strip()]
    if len(chunks) <= 1:
        chunks = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]

    if not chunks:
        return text[:maxChars]

    scores = [(idx, scoreChunk(chunk, keywords), len(chunk)) for idx, chunk in enumerate(chunks)]
    scores.sort(key=lambda item: (item[1], item[2]), reverse=True)

    chosen: list[int] = []
    total = 0
    for idx, _, length in scores:
        if total + length + 2 > maxChars and chosen:
            continue
        chosen.append(idx)
        total += length + 2
        if total >= maxChars:
            break

    chosen.sort()
    condensed = "\n\n".join(chunks[idx] for idx in chosen)
    if len(condensed) > maxChars:
        condensed = condensed[:maxChars].rstrip()
    return condensed


def buildSnippet(text: str, keywords: list[str], maxChars: int) -> str:
    """Extract a snippet around the first matching keyword."""
    if not text or not keywords or maxChars <= 0:
        return ""

    lower = text.lower()
    matchIndex = -1
    for word in keywords:
        idx = lower.find(word)
        if idx != -1:
            matchIndex = idx
            break

    if matchIndex == -1:
        return ""

    pad = max(maxChars // 3, 120)
    start = max(0, matchIndex - pad)
    end = min(len(text), start + maxChars)
    snippet = text[start:end].strip()
    return snippet


class AiMemory:
    """Core: AI memory store for a project."""

    __slots__ = ("_items", "_project")

    def __init__(self, project: NWProject) -> None:
        self._project = project
        self._items: list[str] = []

    @property
    def items(self) -> list[str]:
        """Return a copy of the memory items."""
        return self._items.copy()

    def setItems(self, items: list[str]) -> None:
        """Replace memory items."""
        self._items = [self._cleanItem(item) for item in items if self._cleanItem(item)]

    def addItem(self, text: str) -> bool:
        """Add a memory item if it is unique."""
        item = self._cleanItem(text)
        if not item:
            return False
        lowered = {entry.lower() for entry in self._items}
        if item.lower() in lowered:
            return False
        self._items.append(item)
        return True

    def removeItem(self, text: str) -> bool:
        """Remove a memory item by value."""
        for idx, item in enumerate(self._items):
            if item == text:
                del self._items[idx]
                return True
        return False

    def load(self) -> bool:
        """Load memory items from disk."""
        memFile = self._project.storage.getMetaFile(nwFiles.AI_MEMORY_FILE)
        if not isinstance(memFile, Path):
            return False

        if not safeExists(memFile):
            self._items = []
            return True

        try:
            with open(memFile, mode="r", encoding="utf-8") as inFile:
                data = json.load(inFile)
            items = data.get("novelWriter.aiMemory", {}).get("items", [])
        except Exception:
            logger.error("Failed to load AI memory")
            logException()
            return False

        if isinstance(items, list):
            self.setItems([str(item) for item in items])
        else:
            self._items = []

        return True

    def save(self) -> bool:
        """Save memory items to disk."""
        memFile = self._project.storage.getMetaFile(nwFiles.AI_MEMORY_FILE)
        if not isinstance(memFile, Path):
            return False

        try:
            with open(memFile, mode="w+", encoding="utf-8") as outFile:
                data = {
                    "novelWriter.aiMemory": {
                        "items": self._items,
                    },
                }
                outFile.write(jsonEncode(data, nmax=3))
        except Exception:
            logger.error("Failed to save AI memory")
            logException()
            return False

        return True

    @staticmethod
    def _cleanItem(text: str) -> str:
        """Normalise a memory item."""
        return simplified(text).strip()


class AiContextBuilder:
    """Core: Context and retrieval helper."""

    __slots__ = ("_project", "_ragMaxDocs", "_ragSnippet")

    def __init__(self, project: NWProject, ragMaxDocs: int, ragSnippet: int) -> None:
        self._project = project
        self._ragMaxDocs = max(ragMaxDocs, 1)
        self._ragSnippet = max(ragSnippet, 120)

    def buildRagContext(self, prompt: str) -> list[str]:
        """Collect retrieval context from the project."""
        keywords = extractKeywords(prompt)
        if not keywords:
            return []

        results: list[str] = []
        storage = self._project.storage
        for item in self._project.tree:
            if not item.isFileType() or item.isInactiveClass():
                continue
            text = storage.getDocumentText(item.itemHandle)
            snippet = buildSnippet(text, keywords, self._ragSnippet)
            if snippet:
                title = item.itemName or item.itemHandle
                results.append(f"{title}\n{snippet}")
            if len(results) >= self._ragMaxDocs:
                break

        return results


class OpenRouterClient:
    """Core: OpenRouter chat client."""

    __slots__ = ("_apiKey", "_model", "_timeout")

    def __init__(self, apiKey: str, model: str | None = None, timeout: int = 60) -> None:
        self._apiKey = apiKey
        self._model = model or DEFAULT_OPENROUTER_MODEL
        self._timeout = timeout

    def sendChat(self, messages: list[dict[str, str]], temperature: float = 0.7) -> str:
        """Send a chat completion request and return the content."""
        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
        }

        headers = {
            "Authorization": f"Bearer {self._apiKey}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://novelwriter.io",
            "X-Title": "novelWriter",
        }

        data = json.dumps(payload).encode("utf-8")
        req = request.Request(OPENROUTER_CHAT_URL, data=data, headers=headers, method="POST")
        try:
            with request.urlopen(req, timeout=self._timeout) as resp:
                raw = resp.read().decode("utf-8")
        except error.HTTPError as exc:
            raise RuntimeError(f"OpenRouter request failed: {exc.code} {exc.reason}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"OpenRouter connection failed: {exc.reason}") from exc

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError("OpenRouter response was not valid JSON") from exc

        if isinstance(data, dict) and data.get("error"):
            raise RuntimeError(str(data["error"]))

        if isinstance(data, dict):
            choices = data.get("choices", [])
            if choices and isinstance(choices[0], dict):
                message = choices[0].get("message", {})
                if isinstance(message, dict):
                    content = message.get("content", "")
                    if isinstance(content, str):
                        return content

        raise RuntimeError("OpenRouter response did not include content")


def buildSystemPrompt(
    customInstructions: str,
    guardrails: str,
    reasoning: bool,
) -> str:
    """Build the system prompt for the AI assistant."""
    parts = [
        "You are a writing assistant for novelWriter.",
        "Focus on producing novel-quality prose and useful editing feedback.",
    ]
    appendIfSet(parts, customInstructions.strip())
    appendIfSet(parts, guardrails.strip())
    if reasoning:
        parts.append(
            "Provide a short reasoning summary after the response, without revealing chain-of-thought."
        )
    return "\n\n".join(parts)


def buildUserPrompt(prompt: str, context: list[str]) -> str:
    """Build the user prompt with context sections."""
    sections: list[str] = []
    if context:
        sections.append("Context:\n" + "\n\n".join(context))
    sections.append("Request:\n" + prompt.strip())
    return "\n\n".join(sections)


def shouldEnableAi() -> bool:
    """Return True if AI usage is allowed by configuration."""
    return CONFIG.aiEnabled and bool(CONFIG.aiApiKey)
