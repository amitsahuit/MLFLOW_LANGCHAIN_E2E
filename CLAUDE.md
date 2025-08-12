# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an MLflow 3 Proof of Concept demonstrating comprehensive GenAI model lifecycle management with HuggingFace integration. The architecture implements a production-ready LLM application lifecycle with experimentation, model registry, deployment, and serving phases.

## Key Architecture

**Layered Architecture:**
- **Experiment Layer** (`mlflow_huggingface_experiment.py`) - Core model development with MLflow decorators (`@trace`, `@task`)
- **Deployment Layer** (`deploy_model.py`) - Model registration using custom PyFunc wrapper
- **Serving Layer** (`serve_model.py`) - Production API with subprocess server management
- **Testing Layer** (`test_*.py`) - Multi-level testing including prompt A/B testing

**Core Integration Pattern:**
- HuggingFace Transformers + LangChain + MLflow 3 native features
- Custom `HuggingFaceModelWrapper` class bridges HuggingFace models with MLflow serving
- Comprehensive observability through nested runs and tracing
- CPU-optimized configuration for accessibility

## Development Commands

### Environment Setup
```bash
# Activate virtual environment
source .venv/bin/activate

# Start MLflow server (required for all operations)
mlflow server --host 127.0.0.1 --port 5050
```

### Core Workflows
```bash
# Run main experiment with tracing
python mlflow_huggingface_experiment.py

# Register model in MLflow Model Registry
python deploy_model.py

# Test deployed model locally
python quick_test.py

# Start model serving API
python serve_model.py
# OR serve via MLflow CLI:
mlflow models serve -m models:/huggingface-dialogpt-model/1 -p 5001 --host 0.0.0.0

# Test prompt templates and A/B testing
python test_prompts.py

# Test MLflow 3 native prompt features
python mlflow3_native_prompts.py
```

### API Testing
```bash
# Test served model via curl
curl -X POST http://localhost:5001/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "dataframe_split": {
      "columns": ["question"],
      "data": [["What is machine learning?"]]
    }
  }'
```

## Configuration Management

**Centralized Configuration Pattern:**
- All experiments use `ExperimentConfig` dataclass in `mlflow_huggingface_experiment.py`
- MLflow tracking URI: `http://127.0.0.1:5050`
- Model serving port: `5001`
- Default model: `microsoft/DialoGPT-small` (CPU-optimized)

**Key Configuration Points:**
- `MLFLOW_TRACKING_URI` - MLflow server endpoint
- `MODEL_NAME` - Registered model name in MLflow Registry
- `EXPERIMENT_NAME` - MLflow experiment for organizing runs

## Model Lifecycle Workflow

**Complete Lifecycle:**
1. **Experimentation** → Load HuggingFace model → Register prompt templates → Create LangChain chain → Test with tracing → Evaluate performance
2. **Registration** → Download model locally → Create PyFunc wrapper → Register in MLflow → Test deployment
3. **Serving** → Load from registry → Start API server → Handle predictions → Monitor performance
4. **Testing** → Prompt A/B testing → Quality evaluation → Performance benchmarking

**MLflow Integration Patterns:**
- Use `@mlflow.trace` decorator for automatic tracing
- Nested runs for individual inference steps
- Comprehensive parameter, metric, and artifact logging
- Model signatures with input examples for serving

## Prompt Management Architecture

**Prompt Template System:**
- Multiple template variations (simple, conversational, expert, educational)
- MLflow 3 native prompt registry when supported
- Fallback to artifact-based storage
- A/B testing framework for template comparison
- Performance metrics per template

**Prompt Testing Workflow:**
```python
# Template variations stored in prompt_templates dict
# Each template tested with same questions
# Results logged to MLflow for comparison
# Best-performing template identified through metrics
```

## Custom Model Wrapper

**HuggingFaceModelWrapper Class:**
- Implements `mlflow.pyfunc.PythonModel` interface
- Loads HuggingFace model and tokenizer in `load_context()`
- Creates LangChain pipeline for prompt management
- Handles multiple input formats (DataFrame, dict, string)
- Provides consistent prediction API

**Key Implementation Details:**
- Tokenizer padding configuration with fallback to EOS token
- CPU device mapping for broad compatibility
- LangChain integration for prompt templating
- Error handling with informative messages

## Testing Strategy

**Multi-Level Testing:**
- **Unit Testing** - Basic model functionality and API contracts
- **Integration Testing** - End-to-end workflow validation
- **Prompt Testing** - A/B testing different prompt templates
- **Performance Testing** - Response time and quality metrics
- **Interactive Testing** - Real-time Q&A sessions

**Test Data:**
Standard test questions cover AI/ML topics to validate model responses across different prompt templates.

## Observability and Monitoring

**MLflow Tracing:**
- Automatic tracing of all decorated functions
- Nested runs for granular operation tracking
- Parameter, metric, and artifact logging
- Detailed error tracking and debugging support

**Key Metrics Tracked:**
- Success rate, response length, processing time
- Question/response quality assessments
- Resource usage and performance benchmarks
- Prompt template effectiveness comparisons

## Production Considerations

**Server Management:**
- Subprocess-based MLflow model server
- Health checks and readiness validation
- Graceful startup/shutdown handling
- Error recovery and restart capabilities

**API Design:**
- MLflow standard prediction API format
- Support for batch and single predictions
- Comprehensive error handling and logging
- Interactive session capabilities for testing