import httpx
import os
from dotenv import load_dotenv
from pathlib import Path
import json
from autogen_core.models import LLMMessage

from src.utils.logger import get_logger

logger = get_logger(__name__)
load_dotenv()  # Load environment variables from .env file

SHIELD_URL = os.getenv('SHIELD_URL')
SHIELD_API_KEY = os.getenv('SHIELD_API_KEY')

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

def get_shield_task(entity_type: str, entity_name: str, config: dict) -> str:
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
            current = config
            for part in parts:
                current = current[part]
            shield_task = current[entity_name]["shield_task"]
        else:
            shield_task = config[entity_type][entity_name]["shield_task"]
        
        if shield_task == "":
            logger.error(f"[get_shield_task] Shield task not found for {entity_type}.{entity_name}")
            logger.error(f"[get_shield_task] Returning default shield task ['default']['shield_task']")
            return config['tools']['default']['shield_task']
        
        logger.debug(f"[get_shield_task] Found shield task: {shield_task}")
        return shield_task
    except KeyError:
        logger.error(f"[get_shield_task] Entity {entity_name} not found in {entity_type}")
        raise KeyError(f"No shield task found for {entity_type}.{entity_name}")
    

def load_shield_config(config_path: str = "shield_config.json") -> dict:
    """
    Loads shield configuration from a JSON file that maps tools and agents to their shield tasks.
    
    Args:
        config_path (str): Path to the shield configuration JSON file
        
    Returns:
        dict: Dictionary containing tool and agent configurations with their shield tasks
        
    Raises:
        FileNotFoundError: If the config file doesn't exist
        JSONDecodeError: If the config file isn't valid JSON
    """
    logger.info(f"[load_shield_config] Loading shield configuration from {config_path}")
    try:
        with open(Path(config_path)) as f:
            config = json.load(f)
            logger.debug(f"[load_shield_config] Loaded configuration: {config}")
            return config
    except FileNotFoundError:
        logger.error(f"[load_shield_config] Shield configuration file not found at {config_path}")
        raise
    except json.JSONDecodeError:
        logger.error(f"[load_shield_config] Invalid JSON in shield configuration file")
        raise