"""
novelWriter – GUI AI Assistant
==============================

File History:
Created: 2026-05-08 [2.8b1] GuiAiAssistant

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

import logging

from typing import TYPE_CHECKING

from PyQt6.QtCore import pyqtSlot
from PyQt6.QtWidgets import (
    QDialogButtonBox, QFrame, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QPlainTextEdit, QPushButton, QSpinBox, QTreeWidget,
    QTreeWidgetItem, QVBoxLayout
)

from novelwriter import CONFIG, SHARED
from novelwriter.core.ai import (
    BUILTIN_GUARDRAILS, DEFAULT_CONTEXT_LIMIT, DEFAULT_CONTEXT_MIN,
    DEFAULT_RAG_MAX_DOCS, DEFAULT_RAG_SNIPPET, SUB_AGENTS, AiContextBuilder,
    AiMemory, OpenRouterClient, buildGuardrails, buildSystemPrompt,
    buildUserPrompt, compressContext, extractKeywords
)
from novelwriter.enum import nwStandardButton
from novelwriter.extensions.configlayout import NColorLabel
from novelwriter.extensions.modified import NToolDialog
from novelwriter.extensions.switch import NSwitch
from novelwriter.types import QtRoleAccept, QtRoleApply, QtRoleDestruct

if TYPE_CHECKING:
    from PyQt6.QtGui import QCloseEvent

    from novelwriter.guimain import GuiMain

logger = logging.getLogger(__name__)


class GuiAiAssistant(NToolDialog):
    """GUI Tools: AI assistant and management."""

    def __init__(self, parent: GuiMain) -> None:
        super().__init__(parent=parent)

        logger.debug("Create: GuiAiAssistant")
        self.setObjectName("GuiAiAssistant")
        self.setWindowTitle(self.tr("AI Assistant"))
        self.setMinimumSize(900, 620)

        options = SHARED.project.options
        self.resize(
            options.getInt("GuiAiAssistant", "winWidth", 1000),
            options.getInt("GuiAiAssistant", "winHeight", 720),
        )

        self._memory = AiMemory(SHARED.project)
        self._memory.load()
        self._latestResponse = ""

        # Header
        self.titleLabel = NColorLabel(
            self.tr("AI Assistant"), self, color=SHARED.theme.helpText,
            scale=NColorLabel.HEADER_SCALE, indent=4,
        )

        # Management
        self.grpManagement = QGroupBox(self.tr("AI Management"), self)
        self.layManagement = QGridLayout()
        self.grpManagement.setLayout(self.layManagement)

        self.swtEnabled = NSwitch(self)
        self.swtGuardrails = NSwitch(self)
        self.swtContextCompressor = NSwitch(self)
        self.swtMemory = NSwitch(self)
        self.swtReasoning = NSwitch(self)
        self.swtRag = NSwitch(self)
        self.swtSubAgents = NSwitch(self)

        self.edtApiKey = QLineEdit(self)
        self.edtApiKey.setPlaceholderText("sk-or-...")
        self.edtApiKey.setEchoMode(QLineEdit.EchoMode.Password)

        self.edtModel = QLineEdit(self)
        self.edtModel.setPlaceholderText("openrouter/auto")

        self.spnContextLimit = QSpinBox(self)
        self.spnContextLimit.setRange(1000, 100000)
        self.spnContextLimit.setSingleStep(1000)

        self.spnRagDocs = QSpinBox(self)
        self.spnRagDocs.setRange(1, 20)

        self.layManagement.addWidget(QLabel(self.tr("Enable AI")), 0, 0)
        self.layManagement.addWidget(self.swtEnabled, 0, 1)
        self.layManagement.addWidget(QLabel(self.tr("OpenRouter API key")), 1, 0)
        self.layManagement.addWidget(self.edtApiKey, 1, 1)
        self.layManagement.addWidget(QLabel(self.tr("Model router")), 2, 0)
        self.layManagement.addWidget(self.edtModel, 2, 1)
        self.layManagement.addWidget(QLabel(self.tr("Guardrails")), 3, 0)
        self.layManagement.addWidget(self.swtGuardrails, 3, 1)
        self.layManagement.addWidget(QLabel(self.tr("Context compressor")), 4, 0)
        self.layManagement.addWidget(self.swtContextCompressor, 4, 1)
        self.layManagement.addWidget(QLabel(self.tr("Context limit")), 5, 0)
        self.layManagement.addWidget(self.spnContextLimit, 5, 1)
        self.layManagement.addWidget(QLabel(self.tr("Memory support")), 6, 0)
        self.layManagement.addWidget(self.swtMemory, 6, 1)
        self.layManagement.addWidget(QLabel(self.tr("Reasoning support")), 7, 0)
        self.layManagement.addWidget(self.swtReasoning, 7, 1)
        self.layManagement.addWidget(QLabel(self.tr("RAG support")), 8, 0)
        self.layManagement.addWidget(self.swtRag, 8, 1)
        self.layManagement.addWidget(QLabel(self.tr("RAG max documents")), 9, 0)
        self.layManagement.addWidget(self.spnRagDocs, 9, 1)
        self.layManagement.addWidget(QLabel(self.tr("Sub-agents support")), 10, 0)
        self.layManagement.addWidget(self.swtSubAgents, 10, 1)
        self.layManagement.setColumnStretch(1, 1)

        # Instructions and Guardrails
        self.grpPrompts = QGroupBox(self.tr("Instructions and Guardrails"), self)
        self.layPrompts = QVBoxLayout()
        self.grpPrompts.setLayout(self.layPrompts)

        self.lblInstructions = QLabel(self.tr("Custom instructions"), self)
        self.txtInstructions = QPlainTextEdit(self)
        self.txtInstructions.setPlaceholderText(self.tr("How should AI assist your writing?"))

        self.lblGuardrails = QLabel(self.tr("Custom guardrails"), self)
        self.txtGuardrails = QPlainTextEdit(self)
        self.txtGuardrails.setPlaceholderText(self.tr("Project-specific writing constraints."))

        self.layPrompts.addWidget(self.lblInstructions)
        self.layPrompts.addWidget(self.txtInstructions, 1)
        self.layPrompts.addWidget(self.lblGuardrails)
        self.layPrompts.addWidget(self.txtGuardrails, 1)

        # Assistant
        self.grpAssistant = QGroupBox(self.tr("Assistant"), self)
        self.layAssistant = QVBoxLayout()
        self.grpAssistant.setLayout(self.layAssistant)

        self.lblPrompt = QLabel(self.tr("Prompt"), self)
        self.txtPrompt = QPlainTextEdit(self)
        self.txtPrompt.setPlaceholderText(
            self.tr("Ask for edits, rewrites, brainstorming, or critique.")
        )

        self.lblResponse = QLabel(self.tr("Response"), self)
        self.txtResponse = QPlainTextEdit(self)
        self.txtResponse.setReadOnly(True)

        self.btnAsk = QPushButton(self.tr("Ask AI"), self)
        self.btnAsk.clicked.connect(self._askAi)
        self.btnInsertResponse = QPushButton(self.tr("Insert response in editor"), self)
        self.btnInsertResponse.clicked.connect(self._insertResponse)

        self.assistantBtns = QHBoxLayout()
        self.assistantBtns.addWidget(self.btnAsk)
        self.assistantBtns.addWidget(self.btnInsertResponse)
        self.assistantBtns.addStretch(1)

        self.layAssistant.addWidget(self.lblPrompt)
        self.layAssistant.addWidget(self.txtPrompt, 1)
        self.layAssistant.addLayout(self.assistantBtns)
        self.layAssistant.addWidget(self.lblResponse)
        self.layAssistant.addWidget(self.txtResponse, 2)

        # Memory and Sub-Agents
        self.grpMemory = QGroupBox(self.tr("Memory and Sub-Agents"), self)
        self.layMemory = QVBoxLayout()
        self.grpMemory.setLayout(self.layMemory)

        self.memList = QTreeWidget(self)
        self.memList.setHeaderHidden(True)
        self.memList.setRootIsDecorated(False)
        self.memList.setIndentation(0)

        self.btnAddSelection = QPushButton(self.tr("Add editor selection to memory"), self)
        self.btnAddSelection.clicked.connect(self._addSelectionToMemory)
        self.btnAddResponse = QPushButton(self.tr("Add response to memory"), self)
        self.btnAddResponse.clicked.connect(self._addResponseToMemory)
        self.btnRemoveMemory = QPushButton(self.tr("Remove selected memory item"), self)
        self.btnRemoveMemory.clicked.connect(self._removeMemoryItem)

        self.lblSubAgents = QLabel(self.tr("Built-in sub-agents"), self)
        self.subAgents = QPlainTextEdit(self)
        self.subAgents.setReadOnly(True)
        self.subAgents.setFrameStyle(QFrame.Shape.NoFrame)

        self.layMemory.addWidget(self.memList, 3)
        self.layMemory.addWidget(self.btnAddSelection)
        self.layMemory.addWidget(self.btnAddResponse)
        self.layMemory.addWidget(self.btnRemoveMemory)
        self.layMemory.addSpacing(8)
        self.layMemory.addWidget(self.lblSubAgents)
        self.layMemory.addWidget(self.subAgents, 2)

        # Buttons
        self.btnApply = SHARED.theme.getStandardButton(nwStandardButton.APPLY, self)
        self.btnSave = SHARED.theme.getStandardButton(nwStandardButton.SAVE, self)
        self.btnClose = SHARED.theme.getStandardButton(nwStandardButton.CLOSE, self)
        self.btnApply.clicked.connect(self._applySettings)
        self.btnSave.clicked.connect(self._saveAndClose)
        self.btnClose.clicked.connect(self.closeDialog)

        self.btnBox = QDialogButtonBox(self)
        self.btnBox.addButton(self.btnApply, QtRoleApply)
        self.btnBox.addButton(self.btnSave, QtRoleAccept)
        self.btnBox.addButton(self.btnClose, QtRoleDestruct)

        # Assemble
        self.mainSplit = QGridLayout()
        self.mainSplit.addWidget(self.grpManagement, 0, 0)
        self.mainSplit.addWidget(self.grpPrompts, 0, 1)
        self.mainSplit.addWidget(self.grpAssistant, 1, 0, 1, 2)
        self.mainSplit.addWidget(self.grpMemory, 0, 2, 2, 1)
        self.mainSplit.setColumnStretch(0, 1)
        self.mainSplit.setColumnStretch(1, 2)
        self.mainSplit.setColumnStretch(2, 1)
        self.mainSplit.setRowStretch(1, 1)

        self.topBox = QHBoxLayout()
        self.topBox.addWidget(self.titleLabel)
        self.topBox.addStretch(1)

        self.outerBox = QVBoxLayout()
        self.outerBox.addLayout(self.topBox)
        self.outerBox.addLayout(self.mainSplit, 1)
        self.outerBox.addWidget(self.btnBox)
        self.outerBox.setSpacing(10)

        self.setLayout(self.outerBox)

        self._setSubAgentsText()
        self.loadContent()

        logger.debug("Ready: GuiAiAssistant")

    ##
    #  Methods
    ##

    def loadContent(self) -> None:
        """Load values from config and memory."""
        self.swtEnabled.setChecked(CONFIG.aiEnabled)
        self.edtApiKey.setText(CONFIG.aiApiKey)
        self.edtModel.setText(CONFIG.aiModel)
        self.swtGuardrails.setChecked(CONFIG.aiGuardrailsEnabled)
        self.swtContextCompressor.setChecked(CONFIG.aiContextCompressor)
        self.spnContextLimit.setValue(CONFIG.aiContextLimit or DEFAULT_CONTEXT_LIMIT)
        self.swtMemory.setChecked(CONFIG.aiMemoryEnabled)
        self.swtReasoning.setChecked(CONFIG.aiReasoningEnabled)
        self.swtRag.setChecked(CONFIG.aiRagEnabled)
        self.spnRagDocs.setValue(CONFIG.aiRagMaxDocs or DEFAULT_RAG_MAX_DOCS)
        self.swtSubAgents.setChecked(CONFIG.aiSubAgentsEnabled)
        self.txtInstructions.setPlainText(CONFIG.aiCustomInstructions)
        self.txtGuardrails.setPlainText(CONFIG.aiGuardrailsCustom)
        self._refreshMemoryList()

    ##
    #  Events
    ##

    def closeEvent(self, event: QCloseEvent) -> None:
        """Capture close event and persist UI state."""
        self._saveSettings()
        event.accept()
        self.softDelete()

    ##
    #  Internal Functions
    ##

    def _saveSettings(self) -> None:
        """Save window and memory state."""
        logger.debug("Saving State: GuiAiAssistant")
        options = SHARED.project.options
        options.setValue("GuiAiAssistant", "winWidth", self.width())
        options.setValue("GuiAiAssistant", "winHeight", self.height())
        self._memory.save()

    def _applySettings(self, showInfo: bool = True) -> None:
        """Apply AI settings to user config."""
        CONFIG.aiEnabled = self.swtEnabled.isChecked()
        CONFIG.aiApiKey = self.edtApiKey.text().strip()
        CONFIG.aiModel = self.edtModel.text().strip() or "openrouter/auto"
        CONFIG.aiGuardrailsEnabled = self.swtGuardrails.isChecked()
        CONFIG.aiContextCompressor = self.swtContextCompressor.isChecked()
        CONFIG.aiContextLimit = self.spnContextLimit.value()
        CONFIG.aiMemoryEnabled = self.swtMemory.isChecked()
        CONFIG.aiReasoningEnabled = self.swtReasoning.isChecked()
        CONFIG.aiRagEnabled = self.swtRag.isChecked()
        CONFIG.aiRagMaxDocs = self.spnRagDocs.value()
        CONFIG.aiSubAgentsEnabled = self.swtSubAgents.isChecked()
        CONFIG.aiCustomInstructions = self.txtInstructions.toPlainText().strip()
        CONFIG.aiGuardrailsCustom = self.txtGuardrails.toPlainText().strip()
        CONFIG.saveConfig()
        self._memory.save()
        if showInfo:
            SHARED.info(self.tr("AI settings saved."))

    @pyqtSlot()
    def _saveAndClose(self) -> None:
        """Save settings and close the dialog."""
        self._applySettings(showInfo=True)
        self.close()

    @pyqtSlot()
    def _addSelectionToMemory(self) -> None:
        """Add selected editor text to memory."""
        text = SHARED.mainGui.docEditor.getSelectedText().strip()
        if not text:
            SHARED.warn(self.tr("No selected editor text to add to memory."))
            return
        if self._memory.addItem(text):
            self._memory.save()
            self._refreshMemoryList()

    @pyqtSlot()
    def _addResponseToMemory(self) -> None:
        """Add latest response to memory."""
        text = self._latestResponse.strip()
        if not text:
            SHARED.warn(self.tr("There is no AI response to add to memory."))
            return
        if self._memory.addItem(text):
            self._memory.save()
            self._refreshMemoryList()

    @pyqtSlot()
    def _removeMemoryItem(self) -> None:
        """Remove selected memory entry."""
        if not (item := self.memList.currentItem()):
            return
        value = item.text(0)
        if self._memory.removeItem(value):
            self._memory.save()
            self._refreshMemoryList()

    @pyqtSlot()
    def _askAi(self) -> None:
        """Run an AI request."""
        self._applySettings(showInfo=False)

        prompt = self.txtPrompt.toPlainText().strip()
        if not prompt:
            SHARED.warn(self.tr("Please enter a prompt."))
            return
        if not CONFIG.aiEnabled:
            SHARED.warn(self.tr("AI assistant is disabled."))
            return
        if not CONFIG.aiApiKey:
            SHARED.warn(self.tr("Please set an OpenRouter API key."))
            return

        guardrails = buildGuardrails(CONFIG.aiGuardrailsCustom, CONFIG.aiGuardrailsEnabled)
        if self.swtGuardrails.isChecked() and BUILTIN_GUARDRAILS not in guardrails:
            guardrails = buildGuardrails(CONFIG.aiGuardrailsCustom, True)
        systemPrompt = buildSystemPrompt(
            CONFIG.aiCustomInstructions,
            guardrails,
            CONFIG.aiReasoningEnabled,
        )

        contextParts: list[str] = []
        if CONFIG.aiMemoryEnabled and self._memory.items:
            contextParts.append(
                "Memory:\n" + "\n".join(f"- {item}" for item in self._memory.items)
            )

        selected = SHARED.mainGui.docEditor.getSelectedText().strip()
        if selected:
            contextParts.append("Editor selection:\n" + selected)

        if CONFIG.aiRagEnabled and SHARED.hasProject:
            rag = AiContextBuilder(
                SHARED.project, CONFIG.aiRagMaxDocs or DEFAULT_RAG_MAX_DOCS, DEFAULT_RAG_SNIPPET
            ).buildRagContext(prompt)
            if rag:
                contextParts.append("Project context:\n" + "\n\n".join(rag))

        if CONFIG.aiSubAgentsEnabled:
            subList = "\n".join(f"- {agent.name}: {agent.focus}" for agent in SUB_AGENTS)
            contextParts.append("Sub-agent focus:\n" + subList)

        fullContext = "\n\n".join(contextParts)
        if CONFIG.aiContextCompressor and fullContext:
            keys = extractKeywords(prompt)
            limit = max(CONFIG.aiContextLimit or DEFAULT_CONTEXT_LIMIT, DEFAULT_CONTEXT_MIN)
            fullContext = compressContext(fullContext, keys, limit)

        userPrompt = buildUserPrompt(prompt, [fullContext] if fullContext else [])

        try:
            client = OpenRouterClient(CONFIG.aiApiKey, CONFIG.aiModel)
            response = client.sendChat([
                {"role": "system", "content": systemPrompt},
                {"role": "user", "content": userPrompt},
            ])
        except RuntimeError as exc:
            SHARED.error(self.tr("AI request failed."), details=str(exc))
            return

        self._latestResponse = response.strip()
        self.txtResponse.setPlainText(self._latestResponse)

    @pyqtSlot()
    def _insertResponse(self) -> None:
        """Insert latest response into editor at cursor."""
        if not self._latestResponse.strip():
            SHARED.warn(self.tr("There is no AI response to insert."))
            return
        SHARED.mainGui.docEditor.insertText(self._latestResponse)

    def _refreshMemoryList(self) -> None:
        """Refresh memory list widget."""
        self.memList.clear()
        for item in self._memory.items:
            self.memList.addTopLevelItem(QTreeWidgetItem([item]))

    def _setSubAgentsText(self) -> None:
        """Set built-in sub-agent descriptions."""
        text = "\n".join(f"• {agent.name}: {agent.focus}" for agent in SUB_AGENTS)
        self.subAgents.setPlainText(text)
