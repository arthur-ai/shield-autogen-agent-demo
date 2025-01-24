"""
Main orchestration module for the AI assistant system.
This module implements the core orchestration logic for managing conversations
between users and AI agents, handling message routing, and coordinating
different AI tools and services.

Key Components:
- Message routing and handling
- Tool coordination and execution
- State management and persistence
- Error handling and recovery
- Shield service integration
- Conversation flow management

The system uses a modular architecture with:
- Multiple specialized AI tools for financial analysis
- Asynchronous message handling
- State persistence
- Input validation and safety checks
"""

import asyncio
from datetime import datetime
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional
import os
from dotenv import load_dotenv

from autogen_core import (
    DefaultTopicId,
    SingleThreadedAgentRuntime,
)
from autogen_core.models import (
    ChatCompletionClient,
    LLMMessage,
)
import logging

# For Orchestrator Only Workflow
from Orchestrator_Only_Workflow import AssistantTextMessage, MockPersistence, NeedsUserInputHandler, SlowUserProxyAgent, SoloOrchestratorAssistantAgent, TerminationHandler, UserTextMessage

# For Orchestrator and Validator Workflow
# from Orchestrator_and_Validator_Workflow import AssistantTextMessage, MockPersistence, NeedsUserInputHandler, SlowUserProxyAgent, SoloOrchestratorAssistantAgent, TerminationHandler, UserTextMessage

logger = logging.getLogger(__name__)
load_dotenv()  # Load environment variables from .env file

SHIELD_URL = os.getenv('SHIELD_URL')  # Add default fallback
SHIELD_API_KEY = os.getenv('SHIELD_API_KEY')

state_persister = MockPersistence()

async def main(model_config: Dict[str, Any], latest_user_input: Optional[str] = None) -> None | str:
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
        
    Raises:
        RuntimeError: If critical system components fail to initialize
        ConfigurationError: If model configuration is invalid
    """
    logger.info("[main] Starting main function")
    logger.debug(f"[main] Model config: {model_config}")
    logger.debug(f"[main] Latest user input: {latest_user_input}")
    
    logger.debug("[main] Initialized model client")
    global state_persister

    model_client = ChatCompletionClient.load_component(model_config)
    logger.debug("Initialized model client")
    
    initial_schedule_assistant_message = AssistantTextMessage(
        content="Hi! How can I help you?", source="User"
    )

    termination_handler = TerminationHandler()
    needs_user_input_handler = NeedsUserInputHandler()
    runtime = SingleThreadedAgentRuntime(intervention_handlers=[needs_user_input_handler, termination_handler])
    logger.debug("[main] Runtime initialized with handlers")

    await SlowUserProxyAgent.register(runtime, "User", lambda: SlowUserProxyAgent("User", "I am a user"))
    logger.debug("[main] Registering OrchestratorAgent")

    logger.debug("[main] Registering OrchestratorAgent")

    await SoloOrchestratorAssistantAgent.register(
        runtime,
        "Orchestrator",
        lambda: SoloOrchestratorAssistantAgent(
            "Orchestrator",
            description="AI that helps you parse tasks",
            model_client=model_client,
            initial_message=initial_schedule_assistant_message,
        ),
    )

    runtime_initiation_message: UserTextMessage | AssistantTextMessage
    if latest_user_input is not None:
        runtime_initiation_message = UserTextMessage(content=latest_user_input, source="User")
    else:
        runtime_initiation_message = initial_schedule_assistant_message
    state = state_persister.load_content()

    if state:
        await runtime.load_state(state)
    await runtime.publish_message(
        runtime_initiation_message,
        DefaultTopicId("assistant_conversation"),
    )

    runtime.start()
    await runtime.stop_when(lambda: termination_handler.is_terminated or needs_user_input_handler.needs_user_input)

    user_input_needed = None
    if needs_user_input_handler.user_input_content is not None:
        user_input_needed = needs_user_input_handler.user_input_content
    elif termination_handler.is_terminated:
        print("Terminated - ", termination_handler.termination_msg)

    state_to_persist = await runtime.save_state()
    state_persister.save_content(state_to_persist)

    return user_input_needed


async def ainput(prompt: str = "") -> str:
    """
    Asynchronous wrapper for collecting user input without blocking the event loop.
    
    Implementation:
    - Uses ThreadPoolExecutor to handle blocking input operation
    - Maintains event loop responsiveness during input wait
    - Preserves prompt display functionality
    
    Args:
        prompt (str): Optional text to display when requesting input
        
    Returns:
        str: User's input response
        
    Note: This function should be used instead of standard input() in async contexts
    to prevent blocking the event loop.
    """
    logger.debug(f"[ainput] Getting user input with prompt: {prompt}")
    with ThreadPoolExecutor(1, "AsyncInput") as executor:
        return await asyncio.get_event_loop().run_in_executor(executor, input, prompt)

def setup_logging() -> logging.Logger:
    """
    Configure logging with timestamp-based filename and standard format.
    
    Returns:
        logging.Logger: Configured logger instance for the module
    """
    # Set up logging with timestamp-based filename
    timestamp = datetime.now().strftime('%Y%m%d')
    log_filename = f'logs/assistant/assistant_log_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("[__main__] Starting assistant application")
    logger.info(f"[__main__] Logging to file: {log_filename}")
    
    return logger

if __name__ == "__main__":
    """
    Entry point for the AI assistant application.
    
    Initialization:
    - Configures logging with timestamp-based filenames
    - Loads model configuration from JSON
    - Establishes core system components
    
    Runtime Management:
    - Sets up the conversation loop
    - Handles user input collection
    - Manages application lifecycle
    
    Logging:

    
    Note: This module should be run directly to start the assistant application
    rather than being imported.
    """
    logger = setup_logging()

    # Load model configuration from JSON file
    with open("model_config.json") as f:
        model_config = json.load(f)

    def get_user_input(question_for_user: str):
        """
        Prompts for and collects user input with a formatted display.
        
        Args:
            question_for_user (str): Question to display to the user
            
        Returns:
            str: User's input response
        """
        logger.debug(f"[get_user_input] Displaying question to user: {question_for_user}")
        print("--------------------------QUESTION_FOR_USER--------------------------")
        print(question_for_user)
        print("---------------------------------------------------------------------")
        user_input = input("Enter your input: ")
        logger.debug(f"[get_user_input] Received user input: {user_input}")
        return user_input

    async def run_main(question_for_user: str | None = None):
        """
        Main conversation loop that manages the flow of the assistant.
        
        Args:
            question_for_user (Optional[str]): Question to ask the user if needed
            
        This function:
        - Handles user input collection
        - Manages the conversation state
        - Processes responses through the main orchestration logic
        """
        logger.info("[run_main] Starting conversation loop")
        if question_for_user:
            logger.debug(f"[run_main] Getting user input for question: {question_for_user}")
            user_input = get_user_input(question_for_user)
        else:
            logger.debug("[run_main] No initial question, starting fresh conversation")
            user_input = None
        
        user_input_needed = await main(model_config, user_input)
        if user_input_needed:
            logger.info("[run_main] Further user input needed, continuing conversation")
            await run_main(user_input_needed)

    # Start the async event loop
    asyncio.run(run_main())