"""
AI agents for orchestrating financial analysis and user interactions.

This package contains the agent implementations for:
- Orchestrator agent for coordinating tool usage and workflow
- User proxy agent for handling user interactions
- Additional specialized agents for financial analysis
"""

# Choose one of the following:
from src.agents.orchestrator import SoloOrchestratorAssistantAgent as OrchestratorAgent
# from src.agents.orchestrator_validator import OrchestratorAssistantAgent as OrchestratorAgent

from src.agents.user_agent import SlowUserProxyAgent

__all__ = [
    'OrchestratorAgent',
    'SlowUserProxyAgent'
] 