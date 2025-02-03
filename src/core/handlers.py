from typing import Any
from autogen_core import DefaultInterventionHandler, MessageContext
from src.core.messages import (
    GetSlowUserMessage,
    TerminateMessage
)

from src.utils.logger import get_logger

logger = get_logger(__name__)

class NeedsUserInputHandler(DefaultInterventionHandler):
    """
    Intervention handler for managing user input requirements during conversation flow.
    
    Key Features:
    - Tracks pending questions awaiting user response
    - Maintains user input state
    - Provides status checks for input requirements
    - Manages input request messages
    
    Usage:
        This handler should be registered with the agent runtime to intercept
        and manage messages requiring user interaction.
    """
    
    def __init__(self):
        logger.debug("[NeedsUserInputHandler.init] Initializing handler")
        self.question_for_user: GetSlowUserMessage | None = None

    async def on_publish(self, message: Any, *, message_context: MessageContext) -> Any:
        """
        Intercepts published messages to track user input requests.
        
        Args:
            message: The message being published
            message_context: Context information for the message
            
        Returns:
            The original message unchanged
        """
        logger.debug(f"[NeedsUserInputHandler.on_publish] Processing message: {type(message)}")
        if isinstance(message, GetSlowUserMessage):
            logger.info("[NeedsUserInputHandler.on_publish] Received user input request")
            self.question_for_user = message
        return message

    @property
    def needs_user_input(self) -> bool:
        return self.question_for_user is not None

    @property
    def user_input_content(self) -> str | None:
        if self.question_for_user is None:
            return None
        return self.question_for_user.content


class TerminationHandler(DefaultInterventionHandler):
    """
    Intervention handler for managing system termination requests and shutdown processes.
    
    Key Responsibilities:
    - Tracks termination messages and reasons
    - Manages graceful shutdown sequences
    - Maintains termination state
    - Provides status checks for termination conditions
    
    Note: This handler ensures proper cleanup and state persistence
    before system shutdown.
    """
    
    def __init__(self):
        logger.debug("[TerminationHandler.init] Initializing handler")
        self.terminateMessage: TerminateMessage | None = None

    async def on_publish(self, message: Any, *, message_context: MessageContext) -> Any:
        """
        Intercepts published messages to track termination requests.
        
        Args:
            message: The message being published
            message_context: Context information for the message
            
        Returns:
            The original message unchanged
        """
        logger.debug(f"[TerminationHandler.on_publish] Processing message: {type(message)}")
        if isinstance(message, TerminateMessage):
            logger.info("[TerminationHandler.on_publish] Received termination request")
            self.terminateMessage = message
        return message

    @property
    def is_terminated(self) -> bool:
        return self.terminateMessage is not None

    @property
    def termination_msg(self) -> str | None:
        if self.terminateMessage is None:
            return None
        return self.terminateMessage.content