# Arthur Shield AutoGen Agent Demo

## Overview
This project demonstrates an AI assistant application using AutoGen with integrated safety and quality validation through Arthur Shield service. The system provides stock market analysis and predictions using various AI agents and tools, ensuring reliable and secure AI interactions.

## Features
- Real-time stock data fetching and analysis
- Stock price predictions using machine learning
- Safety and quality validation of AI responses through Shield service
- Asynchronous conversation management
- Comprehensive logging system
- Configurable AI model settings
- Multi-agent collaboration for enhanced analysis
- Content moderation and validation
- Error handling and recovery mechanisms

## Prerequisites
- Python 3.11+
- Required Python packages (see requirements.txt)
- Azure OpenAI API access
- Alpha Vantage API key for stock data
- Shield API credentials

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/arthur-ai/shield-autogen-agent-demo.git
   cd shield-autogen-agent-demo
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables:
   Create a `.env` file in the project root with:
   ```
   AZURE_OPENAI_API_KEY=your_azure_openai_key
   ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
   SHIELD_API_KEY=your_shield_api_key
   ```

5. Create a model_config.json file in the project root with:
   ```
   {
     "model_name": "gpt-4o-mini",
     "temperature": 0.5,
     "max_tokens": 4096
   }
   ```

6. Create a `logs` folder in the project root
7. Create a `logs/assistant` folder in the project root
8. Create a `logs/agent` folder in the project root

## Configuration
For Orchestrator Only Workflow: Make sure to comment out the Orchestrator and Validator Workflow code in main.py. For Orchestrator and Validator Workflow: Make sure to uncomment the Orchestrator and Validator Workflow code in main.py.
The system can be configured through `model_config.json`:
- Model settings (temperature, max tokens, etc.)
- Agent parameters and roles
- Logging preferences
- API endpoints and timeouts
- Safety validation rules

## Usage
1. Start the application:
   ```bash
   python main.py
   ```

2. Interact with the system through the command line interface:
   ```
   Interact with the system through the command line interface
   ```

## Agent Architecture
The system employs multiple specialized agents:
- **Market Analyst Agent**: Processes financial data and generates insights
- **Validation Agent**: Ensures response quality and safety
- **Research Agent**: Gathers and synthesizes market information
- **Coordinator Agent**: Manages agent interactions and workflow

## Safety Features
- Content validation through Shield service
- Input sanitization and validation
- Response quality checks
- Rate limiting and throttling
- Secure API key management
- Audit logging

## Logging
Comprehensive logging is available in `logs/`:
- Agent interactions
- API calls and responses
- Error tracking
- Performance metrics
- Safety validation results

## Error Handling
The system implements robust error handling:
- API failure recovery
- Rate limit management
- Data validation
- Graceful degradation
- Automatic retries

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Testing
Run the test suite:

```bash
pytest tests/
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- AutoGen framework
- Shield service team
- Azure OpenAI
- Alpha Vantage API

## Support
For issues and feature requests, please use the GitHub issue tracker.
