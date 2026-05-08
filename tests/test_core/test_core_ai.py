"""
novelWriter – AI Utilities Tester
=================================

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

import pytest

from novelwriter.constants import nwFiles
from novelwriter.core.ai import (
    BUILTIN_GUARDRAILS, AiContextBuilder, AiMemory, buildGuardrails,
    buildSystemPrompt, buildUserPrompt, compressContext, extractKeywords
)
from novelwriter.core.project import NWProject

from tests.tools import C, buildTestProject


@pytest.mark.core
def testCoreAi_BuildPrompts() -> None:
    """Test prompt and guardrail composition."""
    guard = buildGuardrails("Use present tense.", includeBuiltIns=True)
    assert BUILTIN_GUARDRAILS in guard
    assert "Use present tense." in guard

    system = buildSystemPrompt("Be concise.", guard, reasoning=True)
    assert "Be concise." in system
    assert "reasoning summary" in system

    user = buildUserPrompt("Rewrite this scene.", ["Context A", "Context B"])
    assert "Context A" in user
    assert "Rewrite this scene." in user


@pytest.mark.core
def testCoreAi_CompressContext() -> None:
    """Test context compression utility."""
    text = (
        "First paragraph about weather and setting.\n\n"
        "Second paragraph about Elara and the lighthouse mystery.\n\n"
        "Third paragraph unrelated to prompt."
    )
    keywords = extractKeywords("Improve Elara scene in lighthouse chapter.")
    compressed = compressContext(text, keywords, 90)
    assert len(compressed) <= 90
    assert "Elara" in compressed or "lighthouse" in compressed


@pytest.mark.core
def testCoreAi_MemoryLoadSave(mockGUI, fncPath) -> None:
    """Test AI memory persistence."""
    project = NWProject()
    project.storage._runtimePath = fncPath
    project.storage._ready = True
    (fncPath / "meta").mkdir()

    memory = AiMemory(project)
    assert memory.addItem("Character motive: Elara fears the sea.") is True
    assert memory.addItem("Character motive: Elara fears the sea.") is False
    assert memory.save() is True
    assert (fncPath / "meta" / nwFiles.AI_MEMORY_FILE).exists() is True

    memory2 = AiMemory(project)
    assert memory2.load() is True
    assert memory2.items == ["Character motive: Elara fears the sea."]


@pytest.mark.core
def testCoreAi_BuildRagContext(mockGUI, fncPath, mockRnd) -> None:
    """Test RAG context retrieval from project documents."""
    project = NWProject()
    mockRnd.reset()
    buildTestProject(project, fncPath)

    project.storage.getDocument(C.hChapterDoc).writeDocument(
        "## New Chapter\n\nElara waits at the lighthouse door in the storm.\n"
    )

    rag = AiContextBuilder(project, ragMaxDocs=3, ragSnippet=180)
    context = rag.buildRagContext("Polish Elara in the lighthouse scene.")
    assert context
    assert any("lighthouse" in item.lower() for item in context)
