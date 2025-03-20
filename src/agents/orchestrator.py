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

import json
from typing import Any, Mapping

import uuid

from autogen_core import (
    DefaultTopicId,
    FunctionCall,
    MessageContext,
    RoutedAgent,
    message_handler,
    type_subscription,
)
from autogen_core.model_context import BufferedChatCompletionContext
from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    SystemMessage,
    UserMessage,
    LLMMessage,
)
from autogen_core.tools import BaseTool

from src.agents.prompts import ORCHESTRATOR_SYSTEM_MESSAGE, format_resolution_text
from src.inference.inference import InferenceResult
from src.core.messages import (
    AssistantTextMessage,
    UserTextMessage
)

from src.tools.tools import (
    StockInfoTool,
    StockForecastTool, 
    SentimentAnalysisTool, 
    FinancialLiteracyTool, 
    PortfolioOptimizationTool, 
    OptionsPricingTool, 
    StockScreenerTool
)

from src.arthur_shield.helpers import get_shield_task, send_prompt_to_shield, send_response_to_shield
from src.utils.logger import get_logger

logger = get_logger("src.core")

@type_subscription("assistant_conversation")
class SoloOrchestratorAssistantAgent(RoutedAgent):
    """
    Main orchestrator agent that coordinates between different tools and services.
    
    Key Responsibilities:
    - Task decomposition and intelligent routing
    - Tool coordination and execution management
    - Response validation and quality assurance
    - Shield service integration for safety checks
    - Conversation flow and context management
    
    Features:
    - Handles multiple specialized financial tools
    - Implements safety checks through shield service
    - Maintains conversation context and state
    - Provides response validation and formatting
    
    Attributes:
        _model_context: Buffered context for maintaining conversation state
        _name: Unique identifier for this agent instance
        _model_client: LLM client for generating responses
        _system_message: System prompts defining agent behavior
    """
    def __init__(
        self,
        name: str,
        description: str,
        model_client: ChatCompletionClient,
        initial_message: AssistantTextMessage | None = None,
        shield_config: dict = None,
    ) -> None:
        """
        Initialize the orchestrator agent.
        
        Args:
            name (str): Identifier for this agent instance
            description (str): Human-readable description of agent's purpose
            model_client (ChatCompletionClient): LLM client for generating responses
            initial_message (Optional[AssistantTextMessage]): Starting message for conversation
        """
        logger.info(f"[SoloOrchestratorAssistantAgent.init] Initializing SoloOrchestratorAssistantAgent: {name}")
        super().__init__(description)
        self._model_context = BufferedChatCompletionContext(
            buffer_size=20,
            initial_messages=[UserMessage(content=initial_message.content, source=initial_message.source)]
            if initial_message
            else None,
        )
        self._name = name
        self._model_client = model_client
        self._system_message = [
            SystemMessage(
                content=ORCHESTRATOR_SYSTEM_MESSAGE
                )
        ]
        self._config = shield_config
        self._orchestrator_task = get_shield_task("agents", "OrchestratorAgent", self._config)
        logger.debug(f"[SoloOrchestratorAssistantAgent.init] SoloOrchestratorAssistantAgent initialized with system messages")

    @message_handler
    async def handle_message(self, message: UserTextMessage, ctx: MessageContext) -> None:
        """
        Processes user messages, coordinates with shield service, and manages tool interactions.
        
        Args:
            message (UserTextMessage): The incoming user message
            ctx (MessageContext): Message context information
            
        Returns:
            None
        """
        # Generate a unique conversation ID
        conversation_id = str(uuid.uuid4())

        
        messages = await self._model_context.get_messages()
        logger.debug(f"[SoloOrchestratorAssistantAgent.handle_message] Getting messages, {messages}")
        if len(messages) == 0:
            await self._model_context.add_message(SystemMessage(content=f"System:{self._system_message}"))
        
        logger.info(f"[SoloOrchestratorAssistantAgent.handle_message] Received user message from {message.source}")
        logger.debug(f"[SoloOrchestratorAssistantAgent.handle_message] Message content: {message.content[:100]}...")
        logger.debug(f"[SoloOrchestratorAssistantAgent.handle_message] Conversation ID: {conversation_id}")
        
        await self._model_context.add_message(UserMessage(content=f"User: {message.content}\n", source=message.source))
        result, tool_validation_message = await self.message_loop(message, ctx, self._system_message, 0, conversation_id)

        # Format and publish final response
        # Creates human-readable version of the answer
        logger.info("[SoloOrchestratorAssistantAgent] Publishing assistant response")
        resolution_text = format_resolution_text(message.content, result, tool_validation_message)
        logger.debug(f"[SoloOrchestratorAssistantAgent] Resolution text: {resolution_text}")
        
        # Final shield validation of formatted response
        shield_response = await send_prompt_to_shield(message.content, self._orchestrator_task, conversation_id)
        inference_result = InferenceResult(shield_response)
        logger.debug(f"[SoloOrchestratorAssistantAgent] Shield validation response: {inference_result.get_rule_details()}")
        logger.debug(f"[SoloOrchestratorAssistantAgent] Shield validation response: {inference_result.get_pass_fail_results()}")
        await self._model_context.add_message(SystemMessage(content=f"Shield validations: {inference_result.get_pass_fail_string()}", source=message.source))
        
        # Generate and validate final human-readable response
        resolution_message = SystemMessage(content=resolution_text)
        await self._model_context.add_message(resolution_message)
        final_resolution_response = await self._model_client.create(
            [resolution_message]
        )
        context = await self._model_context.get_messages()
        shield_message = await send_response_to_shield(final_resolution_response.content, self._orchestrator_task, inference_result.get_inference_id(), context)
        inference_result = InferenceResult(shield_message)
        logger.debug(f"[SoloOrchestratorAssistantAgent] Shield validation response: {inference_result.get_rule_details()}")
        logger.debug(f"[SoloOrchestratorAssistantAgent] Shield validation response: {inference_result.get_pass_fail_results()}")
        await self._model_context.add_message(SystemMessage(content=f"Shield validations: {inference_result.get_pass_fail_string()}", source=message.source))
        PII_status = inference_result.return_pii()
        logger.debug(f"[SoloOrchestratorAssistantAgent] PII status: {PII_status}")
        if not PII_status:
            final_resolution_response.content = f"The response contains sensitive information and cannot be shared"

        hallucination_status = inference_result.return_hallucination()
        logger.debug(f"[SoloOrchestratorAssistantAgent] Hallucination status: {hallucination_status}")
        if not hallucination_status:
            final_resolution_response.content = f"The answer is not safe to share."


        final_resolution_response = f"[Trace ID: {conversation_id}] {final_resolution_response.content}"
        # Publish final response to user
        logger.debug(f"[SoloOrchestratorAssistantAgent] Validation response: {final_resolution_response}")
        speech = AssistantTextMessage(content=final_resolution_response, source=self.metadata["type"])
        await self._model_context.add_message(AssistantMessage(content=f"System:{final_resolution_response}", source=self.metadata["type"]))
        await self.publish_message(speech, topic_id=DefaultTopicId("assistant_conversation"))

    async def message_loop(self, message: UserTextMessage, ctx: MessageContext, system_message: SystemMessage, loop_count: int, conversation_id: str) -> None:
        """
        Processes user messages through a validation and response generation loop.
        
        This method handles the core conversation flow, including:
        - Initial shield validation of user input
        - Tool selection and execution
        - Response validation and refinement
        - Safety checks and quality assurance
        
        Args:
            message (UserTextMessage): The user's input message to process
            ctx (MessageContext): Context information for the current message
            system_message (SystemMessage): System-level configuration and prompts
            loop_count (int): Number of refinement iterations attempted
            
        Returns:
            str: The final validated and processed response
            
        Flow:
            1. Initial shield validation of user input
            2. Context management and tool initialization
            3. Initial model response generation
            4. Tool execution and response aggregation
            5. Response validation and refinement
            6. Final safety checks and formatting
            
        Note:
            The method will attempt up to 3 refinement loops if validation fails,
            helping ensure high-quality, relevant responses.
        """
        
        # Initial shield validation of user input
        query = message.content
        
        # Add user message to conversation contexts
        logger.debug(f"[SoloOrchestratorAssistantAgent.message_loop] Adding query to model context")
        await self._model_context.add_message(UserMessage(content=f"{message.source}: {query}", source=message.source))

        # Initialize available tools for processing user requests
        # Sets up specialized financial analysis and information tools
        logger.info("[SoloOrchestratorAssistantAgent.message_loop] Initializing tools")
        tools = [OptionsPricingTool(), StockInfoTool(), SentimentAnalysisTool(), FinancialLiteracyTool(), PortfolioOptimizationTool()]
        
        # Get initial model response without tools
        # This helps understand the user's intent before tool selection
        logger.info("[SoloOrchestratorAssistantAgent.message_loop] Requesting initial model response")
        response = await self._model_client.create(
            system_message + (await self._model_context.get_messages()), tools=[]
        )
        await self._model_context.add_message(SystemMessage(content=f"System:{response.content}", source=self.metadata["type"]))
        logger.debug(f"[SoloOrchestratorAssistantAgent.message_loop] Initial model response: {response.content[:100]}...")
        
        # Get conversation context for shield validation
        # Provides full conversation history for contextual validation
        logger.debug("[SoloOrchestratorAssistantAgent.message_loop] Getting context for shield")
        context = await self._model_context.get_messages()
        
        
        # Get final model response with tools enabled
        # Allows model to use specialized tools for detailed analysis
        logger.info("[SoloOrchestratorAssistantAgent.message_loop] Requesting final model response with tools")
        response_with_tools = await self._model_client.create(
            system_message + (await self._model_context.get_messages()), tools=tools
        )

        
        # Process tool calls and get combined response
        # Executes necessary tool operations and aggregates results
        final_response, tool_responses = await self.loop_calls(response_with_tools.content, tools, ctx, query)
        if final_response == "":
            return response, ""
        
        logger.debug(f"[SoloOrchestratorAssistantAgent.message_loop] Final response: {final_response}")
        
        # Process individual tool responses
        # Validates each tool's output for safety and quality
        tool_validation_message = await self.validate_tool_responses(tool_responses, query, context, conversation_id)
        
        logger.info(f"[SoloOrchestratorAssistantAgent.message_loop] Returning final response {final_response}")
        return final_response, tool_validation_message
    
    async def validate_tool_responses(self, tool_responses: list[dict], message: str, context: list[LLMMessage], conversation_id: str) -> bool:
        """
        Validates responses from multiple tools through the shield service.

        This function processes each tool response through appropriate validation tasks
        based on the tool type. It ensures responses meet safety, quality, and 
        relevance standards before being presented to users.

        Args:
            tool_responses (list[dict]): List of dictionaries containing tool responses
                Each dict should have:
                - name: The tool's identifier
                - response: The tool's output data
            message (LLMMessage): The original LLM message that triggered the tool calls
            context (list[LLMMessage]): Current conversation context for validation

        Returns:
            list[dict]: List of validation results for each tool response
                Each result contains pass/fail status and validation details

        Note:
            Different validation tasks are used based on tool type:
            - Financial data tools use quantitative validation (553368bd-...)
            - Educational content uses content safety validation (915ba7d1-...)
        """
        tool_context = []
        
        for tool_response in tool_responses:
            logger.debug(f"[ToolValidation] Processing tool response: {tool_response}...")
            
            # Get shield task from configuration
            validation_task = get_shield_task("tools", tool_response["name"], self._config)
            
            # Validate tool response through shield service
            logger.debug(f"[ToolValidation] Processing tool response: {tool_response['response'][:100]}...")
            shield_response = await send_prompt_to_shield(message, validation_task, conversation_id)
            inference_result = InferenceResult(shield_response)
            shield_response = await send_response_to_shield(
                tool_response["response"], 
                validation_task,
                inference_result.get_inference_id(),
                context
            )
            
            if shield_response is not None:
                inference_result = InferenceResult(shield_response)
                logger.debug(f"[ToolValidation] Shield validation response: {inference_result.get_rule_details()}")
                logger.debug(f"[ToolValidation] Shield validation response: {inference_result.get_pass_fail_results()}")
                tool_context.append(inference_result.get_pass_fail_string())
        
        tool_validation_message = f"""
            {tool_context}
            """
        tool_system_message = SystemMessage(content=f"System: {tool_validation_message}")
        await self._model_context.add_message(tool_system_message)
        return tool_validation_message

    async def loop_calls(self, calls: list[FunctionCall], tools: list[BaseTool], ctx: MessageContext, query: str) -> None:
        """
        Executes a series of tool function calls and validates their responses.
        
        Args:
            calls (list[FunctionCall]): List of tool functions to execute
            tools (list[BaseTool]): Available tools for execution
            ctx (MessageContext): Current message context
            query (str): Original user query
            
        Returns:
            str: Concatenated and validated responses from all tool calls
            
        Raises:
            ValueError: If a required tool is not found
        """
        final_response = ""
        tool_responses = []
        if isinstance(calls, list) and all(isinstance(item, FunctionCall) for item in calls):
            logger.info(f"[SoloOrchestratorAssistantAgent.loop_calls] Processing {len(calls)} function calls")
            logger.debug(f"[SoloOrchestratorAssistantAgent.loop_calls] Response content: {calls}")
            for idx, call in enumerate(calls, 1):
                logger.info(f"[SoloOrchestratorAssistantAgent.loop_calls] Processing function call {idx}/{len(calls)}: {call.name}")
                tool = next((tool for tool in tools if tool.name == call.name), None)
                if tool is None:
                    logger.error(f"[SoloOrchestratorAssistantAgent] Tool not found: {call.name}")
                    raise ValueError(f"Tool not found: {call.name}")
                logger.debug(f"[SoloOrchestratorAssistantAgent] Running tool {call.name} with arguments: {call.arguments}")
                arguments = json.loads(call.arguments)
                output = await tool.run_json(arguments, ctx.cancellation_token)
                logger.debug(f"[SoloOrchestratorAssistantAgent] Tool {call.name} completed with result: {output}")
                final_response += f"{output.data}"
                tool_response = {
                    "name": call.name,
                    "response": output.data
                }
                tool_responses.append(tool_response)
                await self._model_context.add_message(SystemMessage(content=f"Tool {call.name} response: {output.data}", source=call.name))
        logger.debug(f"[SoloOrchestratorAssistantAgent] Final response: {final_response}")
        return final_response, tool_responses
    
    async def save_state(self) -> Mapping[str, Any]:
        return {
            "memory": await self._model_context.save_state(),
        }

    async def load_state(self, state: Mapping[str, Any]) -> None:
        await self._model_context.load_state(state["memory"])