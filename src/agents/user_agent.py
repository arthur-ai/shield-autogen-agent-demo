from autogen_core import DefaultTopicId, MessageContext, RoutedAgent, message_handler, type_subscription
import logging
from autogen_core.model_context import BufferedChatCompletionContext
from autogen_core.models import (
    AssistantMessage
)
from src.core.messages import (
    AssistantTextMessage,
    UserTextMessage,
    GetSlowUserMessage,
    TerminateMessage
)
from typing import Any, Mapping

logger = logging.getLogger(__name__)


@type_subscription("assistant_conversation")
class SlowUserProxyAgent(RoutedAgent):
    """
    Agent that handles user interactions and manages the conversation flow.
    
    Key Responsibilities:
    - Manages bidirectional user message flow
    - Buffers and maintains conversation context
    - Coordinates with other system agents
    - Handles state persistence and recovery
    - Validates user inputs
    
    Attributes:
        _model_context: Buffered context maintaining conversation history
        _name: Unique identifier for this agent instance
        
    Note: This agent implements "slow" interaction patterns, suitable for
    scenarios requiring thoughtful user responses rather than real-time chat.
    """
    def __init__(
        self,
        name: str,
        description: str,
    ) -> None:
        """
        Initialize the user proxy agent.
        
        Args:
            name (str): Identifier for this agent instance
            description (str): Human-readable description of agent's purpose
        """
        logger.info(f"[SlowUserProxyAgent.init] Initializing SlowUserProxyAgent: {name}")
        super().__init__(description)
        self._model_context = BufferedChatCompletionContext(buffer_size=5)
        self._name = name

    @message_handler
    async def handle_message(self, message: AssistantTextMessage, ctx: MessageContext) -> None:
        """
        Handles incoming assistant messages and manages user interaction flow.
        
        Args:
            message (AssistantTextMessage): The message from the assistant
            ctx (MessageContext): Context information for the message
            
        Returns:
            None
        """
        logger.info(f"[SlowUserProxyAgent.handle_message] Received assistant message from {message.source}")
        logger.debug(f"[SlowUserProxyAgent.handle_message] Message content: {message.content[:100]}...")
        
        logger.debug(f"[SlowUserProxyAgent.handle_message] Adding message to model context")
        await self._model_context.add_message(AssistantMessage(content=message.content, source=message.source))
        
        logger.info(f"[SlowUserProxyAgent.handle_message] Publishing GetSlowUserMessage")
        await self.publish_message(
            GetSlowUserMessage(content=message.content), topic_id=DefaultTopicId("assistant_conversation")
        )

    async def save_state(self) -> Mapping[str, Any]:
        """
        Persists the agent's current state and conversation context.
        
        Returns:
            Mapping[str, Any]: Dictionary containing serializable state data
        """
        state_to_save = {
            "memory": await self._model_context.save_state(),
        }
        return state_to_save

    async def load_state(self, state: Mapping[str, Any]) -> None:
        """
        Restores the agent's state from previously saved data.
        
        Args:
            state (Mapping[str, Any]): Previously saved state data
        """
        await self._model_context.load_state(state["memory"])

