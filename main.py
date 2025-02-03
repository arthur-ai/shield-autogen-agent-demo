import asyncio
import os
from dotenv import load_dotenv

from src.utils import get_user_input

from src.utils import get_logger, setup_logging

from src.workflow_manager import WorkflowManager

load_dotenv()  # Load environment variables from .env file

setup_logging()
logger = get_logger(__name__)

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
    # Load model configuration from JSON file
    config_file = os.getenv('MODEL_CONFIG_PATH')

    workflow_manager = WorkflowManager()

    async def conversation_loop(question_for_user: str | None = None):
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
            user_input = await get_user_input(question_for_user)
        else:
            logger.debug("[run_main] No initial question, starting fresh conversation")
            user_input = None
        
        user_input_needed = await workflow_manager.trigger_agentic_workflow(
            config_file, 
            user_input
        )
        if user_input_needed:
            logger.info("[run_main] Further user input needed, continuing conversation")
            await conversation_loop(user_input_needed)

    # Start the async event loop
    asyncio.run(conversation_loop())
