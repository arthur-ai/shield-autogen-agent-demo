"""
Core workflow management module for the AI assistant system.
Handles orchestration of conversations between users and AI agents,
including message routing, state management, and agent coordination.

Key responsibilities:
- Agent initialization and registration
- Conversation state management
- Message flow orchestration
- Runtime environment configuration
- Shield service integration
"""

import json
import logging
from typing import Any, Dict, Optional

from autogen_core import DefaultTopicId, SingleThreadedAgentRuntime
from autogen_core.models import ChatCompletionClient

from .arthur_shield import load_shield_config
from .core import (
    NeedsUserInputHandler,
    TerminationHandler,
    MockPersistence,
    AssistantTextMessage,
    UserTextMessage
)
from .agents import SlowUserProxyAgent, OrchestratorAgent

logger = logging.getLogger(__name__)

class WorkflowManager:
    def __init__(self):
        self.state_persister = MockPersistence()

    async def trigger_agentic_workflow(
        self,
        config_file: Dict[str, Any],
        latest_user_input: Optional[str] = None
    ) -> None | str:
        """
        Primary orchestration function for the AI assistant system.
        
        System Initialization:
        - Sets up language model client with provided configuration
        - Initializes and registers all required agents
        - Establishes intervention handlers
        - Configures runtime environment
        
        State Management:
        - Loads existing conversation state if available
        - Manages persistence of system state
        - Handles conversation context
        
        Flow Control:
        - Processes user inputs
        - Manages conversation lifecycle
        - Handles termination conditions
        
        Args:
            model_config (Dict[str, Any]): Configuration parameters for the language model
                including model type, parameters, and runtime settings
            latest_user_input (Optional[str]): Most recent user input to process,
                or None for initial conversation start
            
        Returns:
            Optional[str]: Required user input prompt if interaction needed,
                None if conversation complete or terminated
        """
        logger.info("[workflow] Starting workflow function")
        logger.debug(f"[workflow] Latest user input: {latest_user_input}")
        
        shield_config = load_shield_config("config/shield_config.json")

        with open(config_file) as f:
            model_config = json.load(f)
            logger.debug(f"[workflow] Model config: {model_config}")
        model_client = ChatCompletionClient.load_component(model_config)
        logger.debug("[workflow] Initialized model client")
        
        initial_schedule_assistant_message = AssistantTextMessage(
            content="Hi! How can I help you?", source="User"
        )

        termination_handler = TerminationHandler()
        needs_user_input_handler = NeedsUserInputHandler()
        runtime = SingleThreadedAgentRuntime(
            intervention_handlers=[needs_user_input_handler, termination_handler]
        )
        logger.debug("[workflow] Runtime initialized with handlers")

        await SlowUserProxyAgent.register(
            runtime, 
            "User", 
            lambda: SlowUserProxyAgent("User", "I am a user")
        )
        
        await OrchestratorAgent.register(
            runtime,
            "Orchestrator",
            lambda: OrchestratorAgent(
                "Orchestrator",
                description="AI that helps you parse tasks",
                model_client=model_client,
                initial_message=initial_schedule_assistant_message,
                shield_config=shield_config,
            ),
        )

        runtime_initiation_message: UserTextMessage | AssistantTextMessage
        if latest_user_input is not None:
            runtime_initiation_message = UserTextMessage(
                content=latest_user_input, 
                source="User"
            )
        else:
            runtime_initiation_message = initial_schedule_assistant_message
        
        state = self.state_persister.load_content()

        if state:
            await runtime.load_state(state)
            
        await runtime.publish_message(
            runtime_initiation_message,
            DefaultTopicId("assistant_conversation"),
        )

        runtime.start()
        await runtime.stop_when(
            lambda: termination_handler.is_terminated or needs_user_input_handler.needs_user_input
        )

        user_input_needed = None
        if needs_user_input_handler.user_input_content is not None:
            user_input_needed = needs_user_input_handler.user_input_content
        elif termination_handler.is_terminated:
            print("Terminated - ", termination_handler.termination_msg)

        state_to_persist = await runtime.save_state()
        self.state_persister.save_content(state_to_persist)

        return user_input_needed 