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
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, List

import httpx
import os
from dotenv import load_dotenv
import uuid

from autogen_core import (
    DefaultInterventionHandler,
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
import logging

from agents import StockInfoTool, StockForecastTool, SentimentAnalysisTool, FinancialLiteracyTool, PortfolioOptimizationTool, OptionsPricingTool, StockScreenerTool, SavingsGoalTool

logger = logging.getLogger(__name__)
load_dotenv()  # Load environment variables from .env file

SHIELD_URL = os.getenv('SHIELD_URL')
SHIELD_API_KEY = os.getenv('SHIELD_API_KEY')

@dataclass
class TextMessage:
    """
    Base class for text-based messages in the system.
    
    Attributes:
        source (str): The origin/sender of the message (e.g., "user", "assistant", "system")
        content (str): The actual message content as plain text
        
    Note: This serves as the parent class for specialized message types like
    UserTextMessage and AssistantTextMessage.
    """
    source: str
    content: str


@dataclass
class GetSlowUserMessage:
    """
    Message class representing a request for user input.
    Used when the system needs to pause and wait for user interaction.
    
    Attributes:
        content (str): The message or prompt to show to the user
    """
    content: str


@dataclass
class TerminateMessage:
    """
    Message class indicating that the conversation should be terminated.
    Used for graceful shutdown or error conditions.
    
    Attributes:
        content (str): The reason or message for termination
    """
    content: str


@dataclass
class UserTextMessage(TextMessage):
    """
    Represents a message from the user in the conversation.
    Inherits from TextMessage and maintains the same structure.
    Used to differentiate user messages from other message types.
    """
    pass


@dataclass
class AssistantTextMessage(TextMessage):
    """
    Represents a message from the AI assistant in the conversation.
    Inherits from TextMessage and maintains the same structure.
    Used to differentiate assistant messages from other message types.
    """
    pass

from typing import List, Dict, Optional, Any


class RuleResult:
    def __init__(
        self,
        rule_id: str,
        name: str,
        rule_type: str,
        scope: str,
        result: str,
        latency_ms: int,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.id = rule_id
        self.name = name
        self.rule_type = rule_type
        self.scope = scope
        self.result = result
        self.latency_ms = latency_ms
        self.details = details

    @property
    def result_boolean(self) -> bool:
        """
        Returns True if the result is "Pass", otherwise False.
        """
        return self.result.lower() == "pass"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "rule_type": self.rule_type,
            "scope": self.scope,
            "result": self.result_boolean,  # Convert "Pass"/"Fail" to True/False
            "latency_ms": self.latency_ms,
            "details": self.details,
        }


class InferenceResult:
    def __init__(self, input_json: Dict[str, Any]):
        """
        Initializes an InferenceResult object from a JSON dictionary.
        
        :param input_json: The input JSON dictionary.
        """
        self.inference_id = input_json['inference_id']
        self.user_id = input_json['user_id']
        self.rule_results = [
            RuleResult(
                rule_id=rule['id'],
                name=rule['name'],
                rule_type=rule['rule_type'],
                scope=rule['scope'],
                result=rule['result'],
                latency_ms=rule['latency_ms'],
                details=rule.get('details')
            )
            for rule in input_json['rule_results']
        ]

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the InferenceResult object to a dictionary.
        
        :return: A dictionary representation of the object.
        """
        return {
            "inference_id": self.inference_id,
            "user_id": self.user_id,
            "rule_results": [rule.to_dict() for rule in self.rule_results],
        }

    def get_pass_fail_results(self) -> List[Dict[str, Any]]:
        """
        Returns a list of rules with their pass/fail status.
        """
        return [
            {"id": rule.id, "name": rule.name, "result": rule.result_boolean}
            for rule in self.rule_results
        ]

    def get_pass_fail_string(self) -> str:
        """
        Returns a formatted string of rule names and their pass/fail status.
        
        Example output:
            "Content Safety: PASS
             Input Validation: FAIL
             Response Quality: PASS"
        """
        return "\n".join(
            f"{rule.name}: {'PASS' if rule.result_boolean else 'FAIL'}"
            for rule in self.rule_results
        )

    def get_rule_details(self) -> List[Dict[str, Any]]:
        """
        Returns a list of all rules with their details.
        """
        return [
            {"id": rule.id, "name": rule.name, "details": rule.details}
            for rule in self.rule_results
            if rule.details is not None
        ]
    
    def get_inference_id(self) -> str:
        """
        Returns the inference ID.
        """
        return self.inference_id


class MockPersistence:
    """
    Simple in-memory persistence layer for storing and retrieving state.
    
    This class provides a basic implementation for development and testing.
    In production, this should be replaced with a proper database solution.
    
    Attributes:
        _content (Mapping[str, Any]): In-memory dictionary storing state data
        
    Note: This is not suitable for production use as data is lost when the
    process terminates.
    """
    def __init__(self):
        logger.debug("[MockPersistence.init] Initializing in-memory persistence")
        self._content: Mapping[str, Any] = {}

    def load_content(self) -> Mapping[str, Any]:
        """Retrieves stored content from memory"""
        logger.debug("[MockPersistence.load_content] Loading stored content")
        return self._content

    def save_content(self, content: Mapping[str, Any]) -> None:
        """Saves content to memory
        
        Args:
            content: Dictionary of data to persist
        """
        logger.debug("[MockPersistence.save_content] Saving content to memory")
        self._content = content


state_persister = MockPersistence()


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
        _system_message/b: System prompts defining agent behavior
    """
    def __init__(
        self,
        name: str,
        description: str,
        model_client: ChatCompletionClient,
        initial_message: AssistantTextMessage | None = None,
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
                content=f"""
                            I am an AI assistant, that helps you with parsing and understanding tasks into smaller tasks.
                            This system comes with eight helpful financial tools.
                            
                            Available Tools:
                            1. StockInfoTool:
                               - Gets real-time and historical stock data
                               - Parameters:
                                 * symbol (required): Stock ticker symbol

                            2. FinancialLiteracyTool:
                               - Provides educational content and explanations about financial concepts
                               - Parameters:
                                 * topic (required): Financial concept to explain

                            3. SentimentAnalysisTool:
                               - Analyzes market sentiment from news and social media
                               - Parameters:
                                 * symbol (required): Stock ticker symbol

                            4. PortfolioOptimizationTool:
                               - Optimizes investment portfolios using modern portfolio theory
                               - Parameters:
                                 * symbols (required): List of stock symbols
                                 * investment_amount (required): Total investment amount

                            5. StockForecastTool:
                               - Predicts future stock price movements using machine learning
                               - Parameters:
                                 * symbol (required): Stock ticker symbol
                                 * days (required): Number of days to forecast

                            6. OptionsPricingTool:
                               - Calculates option prices and Greeks using Black-Scholes model
                               - Parameters:
                                 * symbol (required): Stock ticker symbol
                                 * strike_price (required): Option strike price
                                 * expiration_date (required): Option expiration date
                                 * option_type (required): "call" or "put"

                            7. StockScreenerTool:
                               - Screens stocks based on specified criteria
                               - Parameters:
                                 * criteria (required): Dictionary of screening criteria
                                 * market (required): Market to screen (e.g., "NYSE", "NASDAQ")

                            8. SavingsGoalTool:
                               - Calculates savings plans and investment strategies for financial goals
                               - Parameters:
                                 * target_amount (required): Goal amount to save
                                 * timeframe (required): Time period in months
                                 * initial_investment (required): Starting investment amount
                                 * monthly_contribution (required): Monthly contribution amount

                            I need to always try to split the task into smaller tasks that are listed above.
                            Each task is going to be a single tool call.
                            I will always be providing a JSON list where each element contains:
                            - agent: the agent responsible for the task
                            - task: the task to be performed
                            - params: the parameters for the task
                            Along with that I will provide a function call to the tool that is responsible for the task.
                        """
                )
        ]
        self._orchestrator_task = "2a584599-4463-49ce-ade6-eb73a0dcefc6"
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
        
        logger.info(f"[SoloOrchestratorAssistantAgent.handle_message] Received user message from {message.source}")
        logger.debug(f"[SoloOrchestratorAssistantAgent.handle_message] Message content: {message.content[:100]}...")
        logger.debug(f"[SoloOrchestratorAssistantAgent.handle_message] Conversation ID: {conversation_id}")
        
        await self._model_context.add_message(UserMessage(content=message.content, source=message.source))
        result = await self.message_loop(message, ctx, self._system_message, 0, conversation_id)

        # Format and publish final response
        # Creates human-readable version of the answer
        logger.info("[SoloOrchestratorAssistantAgent] Publishing assistant response")
        resolution_text = f"""
                    The initial query was: {message.content}                    
                    the answer was: {result}
                    Can you make the answer more human readable don't add any non-essential text?
                """
        logger.debug(f"[SoloOrchestratorAssistantAgent] Resolution text: {resolution_text}")
        
        # Final shield validation of formatted response
        shield_response = await send_prompt_to_shield(resolution_text, self._orchestrator_task, conversation_id)
        inference_result = InferenceResult(shield_response)
        logger.debug(f"[SoloOrchestratorAssistantAgent] Shield validation response: {inference_result.get_rule_details()}")
        logger.debug(f"[SoloOrchestratorAssistantAgent] Shield validation response: {inference_result.get_pass_fail_results()}")
        await self._model_context.add_message(SystemMessage(content=inference_result.get_pass_fail_string(), source=message.source))
        
        # Generate and validate final human-readable response
        resolution_message = SystemMessage(content=resolution_text)
        final_resolution_response = await self._model_client.create(
            [resolution_message], tools=[]
        )
        context = await self._model_context.get_messages()
        shield_message = await send_response_to_shield(final_resolution_response.content, self._orchestrator_task, inference_result.get_inference_id(), context)
        if shield_message is not None:
            inference_result = InferenceResult(shield_message)
            logger.debug(f"[SoloOrchestratorAssistantAgent] Shield validation response: {inference_result.get_rule_details()}")
            logger.debug(f"[SoloOrchestratorAssistantAgent] Shield validation response: {inference_result.get_pass_fail_results()}")
            await self._model_context.add_message(SystemMessage(content=inference_result.get_pass_fail_string(), source=message.source))

        
        # Publish final response to user
        logger.debug(f"[SoloOrchestratorAssistantAgent] Validation response: {final_resolution_response.content}")
        speech = AssistantTextMessage(content=final_resolution_response.content, source=self.metadata["type"])
        await self._model_context.add_message(AssistantMessage(content=final_resolution_response.content, source=self.metadata["type"]))
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
        # Checks for safety, appropriateness, and basic input validation
        logger.info(f"[SoloOrchestratorAssistantAgent.message_loop] Sending message to shield")
        shield_response = await send_prompt_to_shield(query, self._orchestrator_task, conversation_id)
        

        # Process shield validation results
        # Creates inference result object to track validation status and details
        inference_result = InferenceResult(shield_response)
        logger.debug(f"[SoloOrchestratorAssistantAgent.message_loop] Shield validation response: {inference_result.get_rule_details()}")
        logger.debug(f"[SoloOrchestratorAssistantAgent.message_loop] Shield validation response: {inference_result.get_pass_fail_results()}")
        
        # Add user message to conversation contexts
        # Maintains history for both main conversation and validation flows
        logger.debug(f"[SoloOrchestratorAssistantAgent.message_loop] Adding query to model context")
        await self._model_context.add_message(UserMessage(content=query, source=message.source))

        # Initialize available tools for processing user requests
        # Sets up specialized financial analysis and information tools
        logger.info("[SoloOrchestratorAssistantAgent.message_loop] Initializing tools")
        tools = [StockInfoTool(), SentimentAnalysisTool(), FinancialLiteracyTool(), PortfolioOptimizationTool()]
        
        # Get initial model response without tools
        # This helps understand the user's intent before tool selection
        logger.info("[SoloOrchestratorAssistantAgent.message_loop] Requesting initial model response")
        response = await self._model_client.create(
            system_message + (await self._model_context.get_messages()), tools=[]
        )
        logger.debug(f"[SoloOrchestratorAssistantAgent.message_loop] Initial model response: {response.content[:100]}...")
        
        # Get conversation context for shield validation
        # Provides full conversation history for contextual validation
        logger.debug("[SoloOrchestratorAssistantAgent.message_loop] Getting context for shield")
        context = await self._model_context.get_messages()
        
        # Validate model response through shield service
        # Ensures response meets safety and quality standards
        logger.info("[SoloOrchestratorAssistantAgent.message_loop] Sending response to shield")
        shield_message = await send_response_to_shield(response.content, self._orchestrator_task, inference_result.get_inference_id(), context)
        if shield_message is not None:
            inference_result = InferenceResult(shield_message)
            logger.debug(f"[SoloOrchestratorAssistantAgent.message_loop] Shield validation response: {inference_result.get_rule_details()}")
            logger.debug(f"[SoloOrchestratorAssistantAgent.message_loop] Shield validation response: {inference_result.get_pass_fail_results()}")
            await self._model_context.add_message(SystemMessage(content=inference_result.get_pass_fail_string(), source=message.source))
        
        # Get final model response with tools enabled
        # Allows model to use specialized tools for detailed analysis
        logger.info("[SoloOrchestratorAssistantAgent.message_loop] Requesting final model response with tools")
        response_with_tools = await self._model_client.create(
            system_message + (await self._model_context.get_messages()), tools=tools
        )
        
        # Process tool calls and get combined response
        # Executes necessary tool operations and aggregates results
        final_response, tool_responses = await self.loop_calls(response_with_tools.content, tools, ctx, query)
        logger.debug(f"[SoloOrchestratorAssistantAgent.message_loop] Final response: {final_response}")
        
        # Process individual tool responses
        # Validates each tool's output for safety and quality
        await self.validate_tool_responses(tool_responses, query, context, conversation_id)
        
        logger.info(f"[SoloOrchestratorAssistantAgent.message_loop] Returning final response {final_response}")
        return final_response
    
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
            # Select appropriate validation task based on tool type
            validation_task = ""
            if tool_response["name"] == "fetch_stock_data":  # StockInfoTool
                validation_task = "553368bd-69d4-4fa0-bf5c-89222f79afd8"
            elif tool_response["name"] == "predict_stock_price":  # StockForecastTool
                validation_task = "553368bd-69d4-4fa0-bf5c-89222f79afd8"
            elif tool_response["name"] == "analyze_sentiment":  # SentimentAnalysisTool
                validation_task = "553368bd-69d4-4fa0-bf5c-89222f79afd8"
            elif tool_response["name"] == "explain_finance":  # FinancialLiteracyTool
                validation_task = "915ba7d1-f3a9-4190-b264-a07df4eb6bf7"
            elif tool_response["name"] == "optimize_portfolio":  # PortfolioOptimizationTool
                validation_task = "553368bd-69d4-4fa0-bf5c-89222f79afd8"
            elif tool_response["name"] == "savings_goal_planner":  # SavingsGoalTool
                validation_task = "553368bd-69d4-4fa0-bf5c-89222f79afd8"
            elif tool_response["name"] == "options_pricing_calculator":  # OptionsPricingTool
                validation_task = "553368bd-69d4-4fa0-bf5c-89222f79afd8"
            elif tool_response["name"] == "ai_powered_stock_screener":  # StockScreenerTool
                validation_task = "553368bd-69d4-4fa0-bf5c-89222f79afd8"
            # Validate tool response through shield service
            logger.debug(f"[ToolValidation] Processing tool response: {tool_response['response'][:100]}...")
            shield_response = await send_prompt_to_shield(message, validation_task, conversation_id)
            inference_result = InferenceResult(shield_response)
            shield_response = await send_response_to_shield(tool_response["response"], validation_task, inference_result.get_inference_id(), context)
            if shield_response is not None:
                inference_result = InferenceResult(shield_response)
                logger.debug(f"[ToolValidation] Shield validation response: {inference_result.get_rule_details()}")
                logger.debug(f"[ToolValidation] Shield validation response: {inference_result.get_pass_fail_results()}")
                tool_context.append(inference_result.get_pass_fail_string())
        
        tool_validation_message = f"""
                                    The tool responses were sent to the shield service and the results are: {tool_context}
                                    """
        tool_system_message = SystemMessage(content=tool_validation_message)
        await self._model_context.add_message(tool_system_message)
        return

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
                final_response += f"\n{output.data}"
                tool_response = {
                    "name": call.name,
                    "response": output.data
                }
                tool_responses.append(tool_response)
                await self._model_context.add_message(SystemMessage(content=output.data, source=call.name))
        logger.debug(f"[SoloOrchestratorAssistantAgent] Final response: {final_response}")
        return final_response, tool_responses
    
    async def save_state(self) -> Mapping[str, Any]:
        return {
            "memory": await self._model_context.save_state(),
        }

    async def load_state(self, state: Mapping[str, Any]) -> None:
        await self._model_context.load_state(state["memory"])

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

def get_headers():
    """
    Returns the standard headers needed for API requests.
    Includes authorization and content type headers.
    
    Returns:
        dict: Headers dictionary with auth token and content type
    """
    logger.debug("[get_headers] Generating API request headers")
    return {
        "Authorization": f"Bearer {SHIELD_API_KEY}",
        "Content-Type": "application/json"
    }

async def send_prompt_to_shield(message: str, task: str, conversation_id: str):
    """
    Sends a prompt to the shield service for validation and safety checking.
    
    The shield service performs:
    - Content safety validation
    - Prompt injection detection
    - Quality and relevance checks
    - Response formatting validation
    
    Args:
        message (str): The prompt text to validate
        task (str): Task identifier for the shield service
        
    Returns:
        dict: Validation response containing:
            - inference_id: Unique identifier for this validation
            - validation_results: Safety and quality check results
            - status: Success/failure indication
        
    Raises:
        HTTPError: If the shield service request fails
        ConnectionError: If unable to connect to the service
    """
    logger.info(f"[send_prompt_to_shield] Sending prompt for validation")
    url = f"{SHIELD_URL}/api/v2/tasks/{task}/validate_prompt"
    logger.debug(f"[send_prompt_to_shield] Request URL: {url}")
    logger.debug(f"[send_prompt_to_shield] Message content: {message[:100]}...")
    
    body = {
        "prompt": message,
        "conversation_id": conversation_id,
        "user_id": "1"
    }
    async with httpx.AsyncClient() as client:
        logger.debug("[send_prompt_to_shield] Sending POST request")
        response = await client.post(url, json=body, headers=get_headers())
        if response.status_code == 200:
            result = response.json()
            logger.info("[send_prompt_to_shield] Validation successful")
            logger.debug(f"[send_prompt_to_shield] Response: {result}")
        else:
            logger.error(f"[send_prompt_to_shield] Validation failed with status code {response.status_code}")
            logger.error(f"[send_prompt_to_shield] Error response: {response.text}")
            result = None
    return result

async def send_response_to_shield(response: str, task: str, inference_id: str, context: list[LLMMessage]):
    """
    Validates an AI-generated response through the shield service's safety and quality checks.
    
    The validation process includes:
    - Content safety and appropriateness verification
    - Response quality assessment
    - Context relevance checking
    - Format and structure validation
    
    Args:
        response (str): The AI response requiring validation
        task (str): Task identifier for context-specific validation rules
        inference_id (str): Unique ID linking to the original prompt validation
        context (list[LLMMessage]): Conversation history for contextual validation
        
    Returns:
        dict: Validation results containing:
            - safety_checks: Content safety assessment
            - quality_metrics: Response quality scores
            - validation_status: Overall pass/fail status
            
    Raises:
        HTTPError: If shield service validation fails
        ConnectionError: If shield service is unreachable
    """
    logger.info(f"[send_response_to_shield] Sending response for validation")
    url = f"{SHIELD_URL}/api/v2/tasks/{task}/validate_response/{inference_id}"
    logger.debug(f"[send_response_to_shield] Request URL: {url}")
    
    context_str = ",".join(obj.content for obj in context)
    body = {
        "response": response,
        "context": context_str,
    }
    async with httpx.AsyncClient() as client:
        logger.debug("[send_response_to_shield] Sending POST request")
        response = await client.post(url, json=body, headers=get_headers())
        if response.status_code == 200:
            result = response.json()
            logger.info("[send_response_to_shield] Validation successful")
            logger.debug(f"[send_response_to_shield] Response: {result}")
        else:
            logger.error(f"[send_response_to_shield] Validation failed with status code {response.status_code}")
            logger.error(f"[send_response_to_shield] Error response: {response.text}")
            result = None
    return result
