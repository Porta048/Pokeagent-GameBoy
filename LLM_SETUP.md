# LLM Setup Guide

## Prerequisites

The Pokemon AI agent uses Ollama with the `qwen3-vl:2b` model for decision making. If the LLM is not properly configured, the agent will fall back to reinforcement learning (RL) for all decisions.

## Setting up Ollama

1. **Install Ollama**:
   - Visit [ollama.com](https://ollama.com/) and download/install Ollama for your system
   - Follow the installation instructions for your operating system

2. **Start Ollama Service**:
   ```bash
   ollama serve
   ```
   Keep this terminal running while using the Pokemon AI agent.

3. **Download the Required Model**:
   ```bash
   ollama pull qwen3-vl:2b
   ```
   This model has vision capabilities which are essential for the Pokemon AI to analyze the game screen.

## Testing the LLM Connection

Before running the Pokemon AI agent, test your LLM setup:

```bash
python test_llm_connection.py
```

This script will:
- Check if Ollama is running at the configured host
- Verify the required model is available
- Test a simple request to ensure the connection works

## Troubleshooting

If the LLM is not responding:

1. **Connection Issues**:
   - Verify Ollama is running: `ollama serve`
   - Check that the host URL in `src/cfg.py` matches your Ollama instance (default: `http://localhost:11434`)

2. **Model Issues**:
   - Ensure the model is pulled: `ollama pull qwen3-vl:2b`
   - Check available models: `ollama list`

3. **Performance Issues**:
   - Vision LLM calls can take 5-10 seconds to process
   - The system reuses responses for up to 5 seconds to improve performance
   - Rate limiting is configured to prevent overwhelming the LLM

## Configuration

You can adjust LLM settings in `src/cfg.py`:

- `LLM_ENABLED`: Enable/disable LLM integration
- `LLM_HOST`: Ollama server address
- `LLM_MODEL`: Model to use
- `LLM_TIMEOUT`: Request timeout in seconds
- `LLM_MIN_INTERVAL_MS`: Minimum time between requests
- `LLM_RETRY_ATTEMPTS`: Number of retry attempts for failed requests

## Fallback Behavior

When the LLM is unavailable, the system automatically falls back to the RL network for decision making. The agent will continue to function but without the LLM's strategic insights.