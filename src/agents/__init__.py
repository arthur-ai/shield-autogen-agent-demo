"""
AI agents for orchestrating financial analysis and user interactions.

This package contains the agent implementations for:
- Orchestrator agent for coordinating tool usage and workflow
- User proxy agent for handling user interactions
- Additional specialized agents for financial analysis
"""

from src.agents.orchestrator import SoloOrchestratorAssistantAgent as OrchestratorAgent
from src.agents.user_agent import SlowUserProxyAgent

__all__ = [
    'OrchestratorAgent',
    'SlowUserProxyAgent'
] 