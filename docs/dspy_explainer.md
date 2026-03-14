# DSPy: Programming Language Models

DSPy is a framework for building AI systems that automatically optimizes prompts and weights for language models. Instead of manually writing prompts, you declare what you want and DSPy figures out how to get it.

## The Problem DSPy Solves

**Traditional Approach:**
```
Write prompt → Test → Tweak prompt → Test → Repeat forever...
```

**DSPy Approach:**
```
Define signature → Provide examples → DSPy optimizes automatically
```

## Core Concepts

### 1. Signatures

A **Signature** defines the input/output contract for your LLM task:

```python
class Classify(dspy.Signature):
    """Classify sentiment of text."""
    text: str = dspy.InputField()
    sentiment: Literal["positive", "negative"] = dspy.OutputField()
```

The docstring becomes part of the prompt. Type hints constrain outputs.

### 2. Modules

**Modules** are composable building blocks:

| Module | What it does |
|--------|-------------|
| `dspy.Predict` | Simple zero-shot prediction |
| `dspy.ChainOfThought` | Adds reasoning before output |
| `dspy.ReAct` | Reasoning + tool use |
| `dspy.ProgramOfThought` | Generates and runs code |

```python
# Zero-shot
classifier = dspy.Predict(Classify)

# With reasoning
classifier = dspy.ChainOfThought(Classify)
```

### 3. Optimizers (Teleprompters)

**Optimizers** automatically improve your modules:

| Optimizer | What it learns |
|-----------|---------------|
| `BootstrapFewShot` | Good few-shot examples |
| `BootstrapFewShotWithRandomSearch` | Best combo of examples |
| `MIPROv2` | Instructions + examples |
| `BootstrapFinetune` | Model weights |

```python
optimizer = dspy.BootstrapFewShot(metric=my_metric)
optimized = optimizer.compile(classifier, trainset=data)
```

### 4. Metrics

A **Metric** function tells DSPy what "good" means:

```python
def accuracy_metric(example, prediction, trace=None):
    return 1.0 if example.label == prediction.label else 0.0
```

## How Optimization Works

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Training   │────▶│   Optimizer  │────▶│  Optimized  │
│   Examples  │     │              │     │   Program   │
└─────────────┘     │ 1. Try demos │     └─────────────┘
                    │ 2. Evaluate  │           │
┌─────────────┐     │ 3. Keep best │           ▼
│   Metric    │────▶│              │     ┌─────────────┐
│  Function   │     └──────────────┘     │  Better at  │
└─────────────┘                          │  your task  │
                                         └─────────────┘
```

1. **BootstrapFewShot** finds training examples where the LLM succeeds
2. These become few-shot demonstrations in the prompt
3. The optimized program includes learned examples automatically

## Example: Hate Speech Classification

```python
import dspy
from typing import Literal

# 1. Configure LM
lm = dspy.LM("openrouter/anthropic/claude-3.5-haiku", api_key=KEY)
dspy.configure(lm=lm)

# 2. Define Signature
class HateSpeechClassify(dspy.Signature):
    """Classify text as hate speech or normal."""
    text: str = dspy.InputField()
    label: Literal["hatespeech", "normal"] = dspy.OutputField()

# 3. Create Module
classifier = dspy.Predict(HateSpeechClassify)

# 4. Define Metric
def metric(example, pred, trace=None):
    return 1.0 if example.label == pred.label else 0.0

# 5. Optimize
optimizer = dspy.BootstrapFewShot(metric=metric, max_bootstrapped_demos=2)
optimized = optimizer.compile(classifier, trainset=train_data)

# 6. Use
result = optimized(text="some text to classify")
print(result.label)  # "hatespeech" or "normal"
```

## LiteLLM Integration

DSPy uses [LiteLLM](https://docs.litellm.ai/) under the hood, supporting 100+ LLM providers:

```python
# OpenAI
dspy.LM("openai/gpt-4o-mini")

# Anthropic
dspy.LM("anthropic/claude-3-haiku")

# OpenRouter (any model)
dspy.LM("openrouter/anthropic/claude-3.5-haiku", 
        api_base="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_KEY)

# Local Ollama
dspy.LM("ollama/llama3")
```

## Key Benefits

1. **No prompt engineering** - Define what, not how
2. **Automatic optimization** - Learns from examples
3. **Modular** - Compose complex pipelines from simple modules
4. **Provider agnostic** - Switch models easily
5. **Reproducible** - Save and load optimized programs

## Further Reading

- [DSPy Documentation](https://dspy.ai)
- [DSPy GitHub](https://github.com/stanfordnlp/dspy)
- [Getting Started Tutorial](https://dspy.ai/tutorials/rag/)
