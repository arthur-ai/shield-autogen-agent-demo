# Arthur Shield AutoGen Agent Demo

## Overview
This project provides an example of how Arthur Shield can be used to protect an Agentic Application. 

The agentic use-case in this repository is a Financial Analyst Agent that utilizes a handful of tools to query external systems used in 
generating the response about a user's query. 

## Key Features
- **Intelligent Stock Analysis**: Real-time market data processing and analysis
- **Safety First**: Integration with Arthur Shield for response validation and quality assurance
- **Multi-Agent System**: Coordinated interaction between specialized AI agents
- **Flexible Configuration**: Easy customization of model parameters and safety settings
- **Comprehensive Logging**: Detailed tracking of all agent interactions and system events
- **Error Resilience**: Robust error handling and recovery mechanisms

## Prerequisites
- Python 3.11 or higher
- Required Python packages (detailed in requirements.txt)
- Azure OpenAI API access and credentials
- Alpha Vantage API key for financial data access
- Arthur Shield API credentials and access

## Installation

### 1. Repository Setup
Clone the repository and navigate to the project directory:
```bash
git clone https://github.com/arthur-ai/shield-autogen-agent-demo.git
cd shield-autogen-agent-demo
```

### 2. Python Environment
Create and activate a virtual environment:
```bash
# Create virtual environment
python -m venv venv

# Activate on Unix/macOS
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

### 3. Dependencies
Install required packages:
```bash
pip install -r requirements.txt
```

### 4. Environment Configuration
Create a `.env` file in the project root with your API credentials:
```env
AZURE_OPENAI_API_KEY=your_azure_openai_key
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
SHIELD_API_KEY=your_shield_api_key
```

## Configuration Files

### 1. Model Configuration
Create `model_config.json` in the project root:
```json
{
  "model": {
    "name": "gpt-4",
    "temperature": 0.7,
    "max_tokens": 4096
  },
  "logging": {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  }
}
```

### 2. Shield Configuration
Create `shield_config.json` in the project root:
```json
{
  "tools": {
    "fetch_stock_data": {
      "name": "StockInfoTool",
      "shield_task": "INSERT_SHIELD_TASK_ID_HERE"
    }
  },
  "agents": {
    "OrchestratorAgent": {
      "name": "orchestrator",
      "shield_task": "INSERT_SHIELD_TASK_ID_HERE"
    }
  }
}
```

## System Architecture

### Agent System
The application employs a sophisticated multi-agent architecture:

1. **Orchestrator Agent**
   - Manages conversation flow and task delegation
   - Coordinates between user inputs and assistant responses
   - Ensures proper sequencing of operations
   - Handles high-level decision making

2. **Assistant Agent**
   - Processes specific user requests
   - Generates detailed market analysis
   - Provides stock predictions and insights
   - Handles specialized tasks as directed by the orchestrator

### Safety Integration
The system leverages Arthur Shield's capabilities for:
- Content validation and quality assurance
- Response appropriateness checking
- Safety boundary enforcement
- Bias detection and mitigation

## Usage

### Starting the System
Launch the application:
```bash
python main.py
```

### Example Interactions
```
User: "What's the current market trend for AAPL?"
System: *Processes request through agents with safety validation*
Response: *Provides validated market analysis*
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

## Logging System
The application maintains comprehensive logs including:
- Agent interactions and decisions
- API calls and responses
- Safety validation results
- Error events and recovery actions
- Performance metrics


## Testing
Run the test suite:

```bash
pytest tests/
```

## Error Handling
The system implements robust error handling mechanisms:
- Automatic retry for transient failures
- Graceful degradation when services are unavailable
- Clear error messaging to users
- Comprehensive error logging for debugging

## Contributing
We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Support and Resources
- GitHub Issues: For bug reports and feature requests
- Documentation: Detailed API documentation in `/docs`
- Wiki: Additional guides and best practices
- Email Support: support@arthur.ai

## Acknowledgments
- AutoGen team for the multi-agent framework
- Arthur AI team for the Shield service
- Azure OpenAI for API access
- Alpha Vantage for market data services

---
For the latest updates and detailed documentation, visit our [GitHub repository](https://github.com/arthur-ai/shield-autogen-agent-demo).
