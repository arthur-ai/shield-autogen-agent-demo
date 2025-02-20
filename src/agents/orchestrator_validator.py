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

from src.agents.prompts import ORCHESTRATOR_SYSTEM_MESSAGE, VALIDATOR_SYSTEM_MESSAGE, format_resolution_text
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
    StockScreenerTool, 
    SavingsGoalTool
)

from src.arthur_shield.helpers import get_shield_task, send_prompt_to_shield, send_response_to_shield
from src.utils.logger import get_logger

logger = get_logger("src.core")

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
                content=ORCHESTRATOR_SYSTEM_MESSAGE
                )
        ]
        self._system_validation_message = [
            SystemMessage(
                content=VALIDATOR_SYSTEM_MESSAGE
                )
        ]
        self._config = shield_config
        self._orchestrator_task = self.get_shield_task("agents", "OrchestratorAgent")
        self._validator_task = self.get_shield_task("agents", "ValidatorAgent")
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
        # Generate a unique conversation ID
        conversation_id = str(uuid.uuid4())
        
        messages = await self._model_context.get_messages()
        logger.debug(f"[SoloOrchestratorAssistantAgent.handle_message] Getting messages, {messages}")
        if len(messages) == 0:
            await self._model_context.add_message(SystemMessage(content=f"System:{self._system_message}"))
        
        logger.info(f"[OrchestratorAssistantAgent.handle_message] Received user message from {message.source}")
        logger.debug(f"[OrchestratorAssistantAgent.handle_message] Message content: {message.content[:100]}...")
        logger.debug(f"[OrchestratorAssistantAgent.handle_message] Conversation ID: {conversation_id}")
        
        await self._model_context.add_message(UserMessage(content=message.content, source=message.source))
        await self._validator_context.add_message(UserMessage(content=message.content, source=message.source))
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
            final_resolution_response.content = f"The answer is not safe to share."


        final_resolution_response = f"[Trace ID: {conversation_id}] {final_resolution_response.content}"
        # Publish final response to user
        logger.debug(f"[SoloOrchestratorAssistantAgent] Validation response: {final_resolution_response}")
        speech = AssistantTextMessage(content=final_resolution_response, source=self.metadata["type"])
        await self._model_context.add_message(AssistantMessage(content=f"System:{final_resolution_response}", source=self.metadata["type"]))
        await self.publish_message(speech, topic_id=DefaultTopicId("assistant_conversation"))

    async def message_loop(self, message: UserTextMessage, ctx: MessageContext, system_message: SystemMessage, loop_count: int, conversation_id: str, tool_validation_message: str = None) -> None:
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

        # Initialize available tools for processing user requests
        # Sets up specialized financial analysis and information tools
        logger.info("[OrchestratorAssistantAgent.message_loop] Initializing tools")
        tools = [OptionsPricingTool(), StockInfoTool(), SentimentAnalysisTool(), FinancialLiteracyTool(), PortfolioOptimizationTool()]
        
        # Get initial model response without tools
        # This helps understand the user's intent before tool selection
        logger.info("[OrchestratorAssistantAgent.message_loop] Requesting initial model response")
        response = await self._model_client.create(
            system_message + (await self._validator_context.get_messages()), tools=[]
        )
        logger.debug(f"[OrchestratorAssistantAgent.message_loop] Initial model response: {response.content[:100]}...")
        await self._model_context.add_message(SystemMessage(content=f"System:{response.content}", source=self.metadata["type"]))

        # Get conversation context for shield validation
        # Provides full conversation history for contextual validation
        logger.debug("[OrchestratorAssistantAgent.message_loop] Getting context for shield")
        context = await self._validator_context.get_messages()

        # Get final model response with tools enabled
        # Allows model to use specialized tools for detailed analysis
        logger.info("[OrchestratorAssistantAgent.message_loop] Requesting final model response with tools")
        response_with_tools = await self._model_client.create(
            system_message + (await self._validator_context.get_messages()), tools=tools
        )
        
        # Process tool calls and get combined response
        # Executes necessary tool operations and aggregates results
        final_response, tool_responses = await self.loop_calls(response_with_tools.content, tools, ctx, query)
        if final_response == "":
            return response, ""
        logger.debug(f"[OrchestratorAssistantAgent.message_loop] Final response: {final_response}")
        
        # Process individual tool responses
        # Validates each tool's output for safety and quality
        tool_validation = f"""
                                    {tool_validation_message}
                                    The tool responses were sent to the shield service (on run {loop_count}) and the results are: {await self.validate_tool_responses(tool_responses, query, context, conversation_id)}
                                    """
        
        # Validate final response using LLM
        # Ensures response quality and relevance to original query
        context = await self._validator_context.get_messages()
        is_valid, validation_response = await self.LLM_validation(query, final_response, self._validator_task, context, conversation_id)
        
        # Loop until valid response is generated
        # Continues refining response if validation fails
        logger.debug(f"[OrchestratorAssistantAgent.message_loop] Is valid: {is_valid}")
        if not is_valid and loop_count < 3:
            correction_message = SystemMessage(content=f"""
                    The initial query was: {context[1]}                    
                    the answer was: {final_response}
                    This answer was not valid, the error is: {validation_response}
                    Shield validation results are: {tool_validation}
                    Having seen the error and the mistakes, can you answer the query, {context[1]} again?
                """)
            await self._validator_context.add_message(SystemMessage(content=correction_message.content))
            return await self.message_loop(correction_message, ctx, system_message, loop_count + 1, conversation_id, tool_validation)
        
        logger.info(f"[OrchestratorAssistantAgent.message_loop] Returning final response {validation_response}")
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

    async def LLM_validation(self, query: str, output_data: Any, shield_task: str, context: list[LLMMessage], conversation_id: str) -> bool:
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
                        Can you check if the answer is factually correct and answers the query?
                        Check if it also passes the shield validations and guardrails.
                        Your answer needs to start with Yes or No.
                    """
        logger.debug(f"[OrchestratorAssistantAgent.LLM_validation] Checking text: {check_text}")
        shield_response = await send_prompt_to_shield(check_text, shield_task, conversation_id)
        inference_id = shield_response["inference_id"]

        checking_message = SystemMessage(content=check_text)
        await self._validator_context.add_message(checking_message)
        validation_response = await self._model_client.create(
            self._system_validation_message + (await self._validator_context.get_messages()), tools=[]
        )
        shield_message = await send_response_to_shield(validation_response.content, shield_task, inference_id, context)

        logger.debug(f"[OrchestratorAssistantAgent] Response validation: {validation_response.content}")
        logger.debug(f"[OrchestratorAssistantAgent] Shield message: {shield_message}")
        
        return ["Yes" in validation_response.content, validation_response.content]
    
    def get_shield_task(self, entity_type: str, entity_name: str) -> str:
        """
        Gets the shield task ID for a given tool or agent.
        
        Args:
            entity_type (str): Type of entity ("tools" or "agents.orchestrator_agents")
            entity_name (str): Name of the tool or agent
            config (dict, optional): Pre-loaded configuration. If None, loads from file
            
        Returns:
            str: Shield task ID for the entity
            
        Raises:
            KeyError: If the entity isn't found in the configuration
        """
        
        logger.debug(f"[get_shield_task] Getting shield task for {entity_type}.{entity_name}")
        
        try:
            if "." in entity_type:
                # Handle nested paths like "agents.orchestrator_agents"
                parts = entity_type.split(".")
                current = self._config
                for part in parts:
                    current = current[part]
                shield_task = current[entity_name]["shield_task"]
            else:
                shield_task = self._config[entity_type][entity_name]["shield_task"]
            
            if shield_task == "":
                logger.error(f"[get_shield_task] Shield task not found for {entity_type}.{entity_name}")
                logger.error(f"[get_shield_task] Returning default shield task: {self._orchestrator_task}")
                return self._orchestrator_task
            
            logger.debug(f"[get_shield_task] Found shield task: {shield_task}")
            return shield_task
        except KeyError:
            logger.error(f"[get_shield_task] Entity {entity_name} not found in {entity_type}")
            raise KeyError(f"No shield task found for {entity_type}.{entity_name}")
