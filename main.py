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
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, List

import httpx
import requests
from sklearn.linear_model import LinearRegression
import yfinance as yf
import pandas as pd
import numpy as np


from autogen_core import (
    BaseAgent,
    CancellationToken,
    DefaultInterventionHandler,
    DefaultTopicId,
    FunctionCall,
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
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
from pydantic import BaseModel, Field
import logging

from agents import StockInfoTool, StockForecastTool, SentimentAnalysisTool, FinancialLiteracyTool, PortfolioOptimizationTool, OptionsPricingTool, StockScreenerTool, SavingsGoalTool

logger = logging.getLogger(__name__)


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


class ScheduleMeetingInput(BaseModel):
    """
    Input model for scheduling meeting requests.
    
    Attributes:
        recipient (str): Name of the person to meet with
        date (str): Desired meeting date
        time (str): Desired meeting time
    """
    recipient: str = Field(description="Name of recipient")
    date: str = Field(description="Date of meeting")
    time: str = Field(description="Time of meeting")


class ScheduleMeetingOutput(BaseModel):
    """
    Output model for scheduling meeting results.
    Currently serves as a placeholder for future meeting confirmation details.
    """
    pass


class ScheduleMeetingTool(BaseTool[ScheduleMeetingInput, ScheduleMeetingOutput]):
    """
    Tool for scheduling meetings with specified recipients.
    Handles the creation and confirmation of meeting requests.
    """
    def __init__(self):
        """
        Initialize the meeting scheduling tool with input/output models
        and tool metadata.
        """
        logger.debug("[ScheduleMeetingTool.init] Initializing meeting scheduling tool")
        super().__init__(
            ScheduleMeetingInput,
            ScheduleMeetingOutput,
            "schedule_meeting",
            "Schedule a meeting with a recipient at a specific date and time",
        )

    async def run(self, args: ScheduleMeetingInput, cancellation_token: CancellationToken) -> ScheduleMeetingOutput:
        """
        Execute the meeting scheduling operation.
        
        Args:
            args (ScheduleMeetingInput): Meeting details including recipient, date, and time
            cancellation_token (CancellationToken): Token for cancelling the operation
            
        Returns:
            ScheduleMeetingOutput: Confirmation of meeting scheduling
        """
        logger.info(f"[ScheduleMeetingTool.run] Scheduling meeting with {args.recipient} on {args.date} at {args.time}")
        print(f"Meeting scheduled with {args.recipient} on {args.date} at {args.time}")
        return ScheduleMeetingOutput()


# Routing configuration for different task types
ROUTING_RULES = {
    "stock_data": "stock_data",  # Routes stock data requests to stock data agent
    "stock_predictor": "stock_predictor",  # Routes prediction requests to predictor agent
}


@type_subscription("assistant_conversation")
class OrchestratorAssistantAgent(RoutedAgent):
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
        logger.info(f"[OrchestratorAssistantAgent.init] Initializing OrchestratorAssistantAgent: {name}")
        super().__init__(description)
        self._model_context = BufferedChatCompletionContext(
            buffer_size=20,
            initial_messages=[UserMessage(content=initial_message.content, source=initial_message.source)]
            if initial_message
            else None,
        )
        self._validator_context = BufferedChatCompletionContext(
            buffer_size=20,
            initial_messages=[UserMessage(content="I am here to help the user validate the input", source="ValidatorAssistantAgent")]
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
        self._system_validation_message = [
            SystemMessage(
                content=f"""
                            I am an AI assistant specialized in resource management and validation. My responsibilities include:
                            1. Validating incoming prompts for completeness and clarity
                            2. Ensuring responses meet quality standards and business requirements
                            3. Formatting responses in a clear, human-readable format with proper structure
                            4. Managing system resources efficiently
                            5. Maintaining consistency in communication style and format
                            
                            When processing requests, I will:
                            - Verify input data integrity
                            - Check for required parameters
                            - Ensure responses are properly formatted
                            - Apply consistent styling and formatting rules
                            - Flag any potential issues or inconsistencies
                            
                            All responses will be formatted with:
                            - Proper paragraph breaks
                            - Consistent formatting
                            - Well-structured lists where applicable
                            - Proper grammar and punctuation
                        """
                )
        ]
        self._orchestrator_task = "2a584599-4463-49ce-ade6-eb73a0dcefc6"
        self._validator_task = "c09f3b53-a558-44b0-8d38-f547720d0631"
        logger.debug(f"[OrchestratorAssistantAgent.init] OrchestratorAssistantAgent initialized with system messages")

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
        shield_task = "915ba7d1-f3a9-4190-b264-a07df4eb6bf7"
        logger.info(f"[OrchestratorAssistantAgent.handle_message] Received user message from {message.source}")
        logger.debug(f"[OrchestratorAssistantAgent.handle_message] Message content: {message.content[:100]}...")
        
        # Initial shield validation of user input
        # Checks for safety, appropriateness, and basic input validation
        logger.info(f"[OrchestratorAssistantAgent.handle_message] Sending message to shield")
        shield_response = await send_prompt_to_shield(message.content, self._orchestrator_task)
        
        # Process shield validation results
        # Creates inference result object to track validation status and details
        inference_result = InferenceResult(shield_response)
        logger.debug(f"[OrchestratorAssistantAgent] Shield validation response: {inference_result.get_rule_details()}")
        logger.debug(f"[OrchestratorAssistantAgent] Shield validation response: {inference_result.get_pass_fail_results()}")

        # Add user message to conversation contexts
        # Maintains history for both main conversation and validation flows
        logger.debug(f"[OrchestratorAssistantAgent.handle_message] Adding query to model context")
        await self._model_context.add_message(UserMessage(content=message.content, source=message.source))
        await self._validator_context.add_message(UserMessage(content=message.content, source=message.source))

        # Initialize available tools for processing user requests
        # Sets up specialized financial analysis and information tools
        logger.info("[OrchestratorAssistantAgent] Initializing tools")
        tools = [StockInfoTool(), SentimentAnalysisTool(), FinancialLiteracyTool(), PortfolioOptimizationTool()]
        
        # Get initial model response without tools
        # This helps understand the user's intent before tool selection
        logger.info("[OrchestratorAssistantAgent] Requesting initial model response")
        response = await self._model_client.create(
            self._system_message + (await self._model_context.get_messages()), tools=[]
        )
        logger.debug(f"[OrchestratorAssistantAgent] Initial model response: {response.content[:100]}...")
        
        # Get conversation context for shield validation
        # Provides full conversation history for contextual validation
        logger.debug("[OrchestratorAssistantAgent] Getting context for shield")
        context = await self._model_context.get_messages()
        
        # Validate model response through shield service
        # Ensures response meets safety and quality standards
        logger.info("[OrchestratorAssistantAgent] Sending response to shield")
        shield_message = await send_response_to_shield(response.content, self._orchestrator_task, inference_result.get_inference_id(), context)
        if shield_message is not None:
            inference_result = InferenceResult(shield_message)
            logger.debug(f"[OrchestratorAssistantAgent] Shield validation response: {inference_result.get_rule_details()}")
            logger.debug(f"[OrchestratorAssistantAgent] Shield validation response: {inference_result.get_pass_fail_results()}")
        
        # Get final model response with tools enabled
        # Allows model to use specialized tools for detailed analysis
        logger.info("[OrchestratorAssistantAgent] Requesting final model response with tools")
        response = await self._model_client.create(
            self._system_message + (await self._model_context.get_messages()), tools=tools
        )

        query = context[-1]
        
        # Process tool calls and get combined response
        # Executes necessary tool operations and aggregates results
        final_response, tool_responses = await self.loop_calls(response.content, tools, ctx, query, shield_task)
        logger.debug(f"[OrchestratorAssistantAgent] Final response: {final_response}")
        
        # Process individual tool responses
        # Validates each tool's output for safety and quality
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
            shield_response = await send_prompt_to_shield(message.content, validation_task)
            # if shield_response is not None:
            inference_result = InferenceResult(shield_response)
            shield_response = await send_response_to_shield(tool_response["response"], validation_task, inference_result.get_inference_id(), context)
            
            if shield_message is not None:
                inference_result = InferenceResult(shield_message)
                logger.debug(f"[ToolValidation] Shield validation response: {inference_result.get_rule_details()}")
                logger.debug(f"[ToolValidation] Shield validation response: {inference_result.get_pass_fail_results()}")
                tool_context.append(inference_result.get_pass_fail_results())

        # Validate final response using LLM
        # Ensures response quality and relevance to original query
        context = await self._model_context.get_messages()
        is_valid, validation_response = await self.LLM_validation(query, final_response, self._validator_task, context)
        
        # Loop until valid response is generated
        # Continues refining response if validation fails
        logger.debug(f"[OrchestratorAssistantAgent] Is valid: {is_valid}")
        while not is_valid:
            is_valid, validation_response = await self.LLM_validation(query, final_response, self._validator_task, context)
            logger.info("[OrchestratorAssistantAgent] Publishing termination message")
            termination_message = TerminateMessage(content=validation_response)
            await self.publish_message(termination_message, topic_id=DefaultTopicId("assistant_conversation"))

        # Format and publish final response
        # Creates human-readable version of the answer
        logger.info("[OrchestratorAssistantAgent] Publishing assistant response")
        resolution_text = f"""
                    The initial query was: {context[1]}                    
                    the answer was: {final_response}
                    Can you make the answer more human readable don't add any non-essential text?
                """
        logger.debug(f"[OrchestratorAssistantAgent] Resolution text: {resolution_text}")
        
        # Final shield validation of formatted response
        shield_response = await send_prompt_to_shield(resolution_text, self._orchestrator_task)
        inference_result = InferenceResult(shield_response)
        logger.debug(f"[OrchestratorAssistantAgent] Shield validation response: {inference_result.get_rule_details()}")
        logger.debug(f"[OrchestratorAssistantAgent] Shield validation response: {inference_result.get_pass_fail_results()}")
        
        # Generate and validate final human-readable response
        resolution_message = SystemMessage(content=resolution_text)
        final_resolution_response = await self._model_client.create(
            self._system_validation_message + [resolution_message], tools=[]
        )
        shield_message = await send_response_to_shield(final_resolution_response.content, self._orchestrator_task, inference_result.get_inference_id(), context)
        if shield_message is not None:
            inference_result = InferenceResult(shield_message)
            logger.debug(f"[OrchestratorAssistantAgent] Shield validation response: {inference_result.get_rule_details()}")
            logger.debug(f"[OrchestratorAssistantAgent] Shield validation response: {inference_result.get_pass_fail_results()}")
        
        # Publish final response to user
        logger.debug(f"[OrchestratorAssistantAgent] Validation response: {final_resolution_response.content}")
        speech = AssistantTextMessage(content=final_resolution_response.content, source=self.metadata["type"])
        await self._model_context.add_message(AssistantMessage(content=final_resolution_response.content, source=self.metadata["type"]))
        await self.publish_message(speech, topic_id=DefaultTopicId("assistant_conversation"))

    async def loop_calls(self, calls: list[FunctionCall], tools: list[BaseTool], ctx: MessageContext, query: str, shield_task: str) -> None:
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
            logger.info(f"[OrchestratorAssistantAgent.loop_calls] Processing {len(calls)} function calls")
            logger.debug(f"[OrchestratorAssistantAgent.loop_calls] Response content: {calls}")
            for idx, call in enumerate(calls, 1):
                logger.info(f"[OrchestratorAssistantAgent.loop_calls] Processing function call {idx}/{len(calls)}: {call.name}")
                tool = next((tool for tool in tools if tool.name == call.name), None)
                if tool is None:
                    logger.error(f"[OrchestratorAssistantAgent] Tool not found: {call.name}")
                    raise ValueError(f"Tool not found: {call.name}")
                logger.debug(f"[OrchestratorAssistantAgent] Running tool {call.name} with arguments: {call.arguments}")
                arguments = json.loads(call.arguments)
                output = await tool.run_json(arguments, ctx.cancellation_token)
                logger.debug(f"[OrchestratorAssistantAgent] Tool {call.name} completed with result: {output}")
                final_response += f"\n{output.data}"
                tool_response = {
                    "name": call.name,
                    "response": output.data
                }
                tool_responses.append(tool_response)
                await self._model_context.add_message(SystemMessage(content=output.data, source=call.name))
                await self._validator_context.add_message(SystemMessage(content=output.data, source=call.name))
        logger.debug(f"[OrchestratorAssistantAgent] Final response: {final_response}")
        return final_response, tool_responses
    
    async def handleValidation_message(self, message: UserTextMessage, ctx: MessageContext) -> None:
        return
    
    
    async def save_state(self) -> Mapping[str, Any]:
        return {
            "memory": await self._model_context.save_state(),
        }

    async def load_state(self, state: Mapping[str, Any]) -> None:
        await self._model_context.load_state(state["memory"])

    async def LLM_validation(self, query: str, output_data: Any, shield_task: str, context: list[LLMMessage]) -> bool:
        """
        Validates if an LLM response properly answers the original query.
        
        Args:
            query (str): The original user query
            output_data (Any): The response data to validate
            shield_task (str): Shield service task identifier
            context (list[LLMMessage]): Current conversation context
            
        Returns:
            bool: True if response is valid, False otherwise
        """
        
        logger.info(f"[OrchestratorAssistantAgent.LLM_validation] Checking if the response is valid")
        check_text = f"""
                        The initial query was: {query}                    
                        the answer was: {output_data}
                        Can you check if the answer is valid and answers the query?
                        Your answer needs to start with Yes or No?
                    """
        logger.debug(f"[OrchestratorAssistantAgent.LLM_validation] Checking text: {check_text}")
        shield_response = await send_prompt_to_shield(check_text, shield_task)
        inference_id = shield_response["inference_id"]

        checking_message = SystemMessage(content=check_text)
        validation_response = await self._model_client.create(
            self._system_validation_message + [checking_message], tools=[]
        )
        shield_message = await send_response_to_shield(validation_response.content, shield_task, inference_id, context)

        logger.debug(f"[OrchestratorAssistantAgent] Response validation: {validation_response.content}")
        logger.debug(f"[OrchestratorAssistantAgent] Shield message: {shield_message}")
        
        return ["Yes" in validation_response.content, validation_response.content]

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
        "Authorization": "Bearer 1WEAq1j-6yyO59MAFwticjF51v-6U2vhYf4jy8Fi0EM",
        "Content-Type": "application/json"
    }

async def send_prompt_to_shield(message: str, task: str):
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
    url = f"http://localhost:8000/api/v2/tasks/{task}/validate_prompt"
    logger.debug(f"[send_prompt_to_shield] Request URL: {url}")
    logger.debug(f"[send_prompt_to_shield] Message content: {message[:100]}...")
    
    body = {
        "prompt": message,
        "conversation_id": "string",
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
    url = f"http://localhost:8000/api/v2/tasks/{task}/validate_response/{inference_id}"
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

    # agents = {
    #     "parser": SchedulingAssistantAgent("Parser",
    #         description="AI that helps you parse tasks",
    #         model_client=model_client,
    #         initial_message=initial_schedule_assistant_message,),
    #     # "stock_data": StockDataAgent("Agent that gets stock data"),
    #     # "stock_predictor": StockPredictorAgent("Agent that predicts stock prices"),
    # }

    termination_handler = TerminationHandler()
    needs_user_input_handler = NeedsUserInputHandler()
    runtime = SingleThreadedAgentRuntime(intervention_handlers=[needs_user_input_handler, termination_handler])
    logger.debug("[main] Runtime initialized with handlers")

    await SlowUserProxyAgent.register(runtime, "User", lambda: SlowUserProxyAgent("User", "I am a user"))
    logger.debug("[main] Registering OrchestratorAgent")

    logger.debug("[main] Registering OrchestratorAgent")
    # await SchedulingAssistantAgent.register(
    #     runtime,
    #     "Parser",
    #     lambda: SchedulingAssistantAgent(
    #         name="Parser",
    #         description="AI for parsing tasks.",
    #         model_client=model_client,
    #     ),
    # )

    await OrchestratorAssistantAgent.register(
        runtime,
        "Orchestrator",
        lambda: OrchestratorAssistantAgent(
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
    - Creates dated log files
    - Configures logging levels and formats
    - Maintains operation history
    
    Note: This module should be run directly to start the assistant application
    rather than being imported.
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