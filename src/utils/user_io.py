"""
User input/output utility functions for handling asynchronous console interactions.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)

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

async def get_user_input(question_for_user: str) -> str:
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
    user_input = await ainput("Enter your input: ")
    logger.debug(f"[get_user_input] Received user input: {user_input}")
    return user_input 