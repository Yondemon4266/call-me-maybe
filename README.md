*This project has been created as part of the 42 curriculum by aluslu.*

# call-me-maybe

## Description

This project is an introduction to function calling with LLMs.

The goal is to transform user prompts into structured JSON function calls while enforcing strict output constraints. Instead of letting the model generate free text, the decoder guides token selection so the produced output follows a predefined JSON format and schema.

At a high level, the program:
- reads prompt inputs and function definitions,
- constrains generation to valid function names and parameter value types,
- emits one JSON object per prompt,
- writes final results as a JSON list.

## Instructions

### Prerequisites
- Python 3.10+
- `uv`

### Installation

From the repository root:

```bash
make install
```

or:

```bash
uv sync
```

### Execution

Default run:

```bash
make run
```

Equivalent command:

```bash
uv run python -m src
```

CLI options:
- `--input`
- `--functions_definition`
- `--output`

Example:

```bash
uv run python -m src \
  --input data/input/function_calling_tests.json \
  --functions_definition data/input/functions_definition.json \
  --output data/output/function_calling_results.json
```

### Lint and checks

```bash
make lint
make lint-strict
```

### Repository requirements reminder

This repository contains:
- `src/`
- `pyproject.toml` and `uv.lock`
- `llm_sdk/`
- `data/input/`
- this `README.md`

`data/output/` is generated at runtime and should not be committed.

## Algorithm explanation

The constrained decoding pipeline is built around a state machine:

1. **Fixed JSON skeleton emission**  
  Deterministic parts of the JSON are emitted directly (fast-forward), e.g. object opening and static keys.

2. **Function name constrained generation**  
  Allowed next tokens for function names are restricted using a trie built from the tokenizer encoding of each allowed function name.

3. **Parameter key/value emission**  
  Parameter keys come from the selected function schema. Parameter values are generated under token-level type constraints.

4. **Type-constrained token filtering**  
  For `string`, `number`, `integer`, and `boolean`, allowed token sets are precomputed from vocabulary and regex checks. At each step, logits for disallowed tokens are masked before argmax selection.

5. **End-of-value handling**  
  Special end-token rules handle transitions for string and numeric values, including split-token edge cases.

This approach keeps generation deterministic where possible and restricted where necessary.

## Design decisions

- **State machine architecture** for explicit control of generation phases.
- **Trie for function names** to guarantee only declared functions can be produced.
- **Type registry abstraction** to isolate per-type token constraints.
- **Pydantic models** for strict input validation and clear schemas.
- **Simple CLI structure** to keep usage and evaluation straightforward.

## Performance analysis

### Accuracy
- High structural accuracy for JSON format due to constrained decoding and deterministic skeleton generation.
- Function names are limited to valid choices by trie constraints.
- Parameter value types are constrained token-by-token.

### Speed
- Fast-forwarding deterministic segments reduces decoding overhead.
- Token masking introduces extra work per step but remains practical for this project scale.

### Reliability
- Input files are validated before generation.
- JSON parsing is checked after generation for each prompt.
- Error paths are explicit, and invalid prompt outputs are skipped with diagnostics.

## Challenges faced

- **Tokenizer-level constraints**: handling cases where delimiters and value fragments share a token.
- **Numeric termination rules**: allowing valid number endings without prematurely closing values.
- **String boundaries**: detecting safe quote-based termination in the presence of escapes.
- **Balancing simplicity and control**: keeping the code readable while enforcing strong constraints.

These were solved with end-token split rules, state transitions, and dedicated per-type handlers.

## Testing strategy

Validation relied on:
- schema validation of inputs (`PromptFormat`, `FunctionFormat`),
- end-to-end runs on provided files in `data/input/`,
- JSON parsing checks on each generated item,
- lint/type checks (`flake8`, `mypy`),
- manual verification of edge cases (empty/invalid data, malformed JSON, unavailable files).

## Example usage

Run with defaults:

```bash
uv run python -m src
```

Run with custom paths:

```bash
uv run python -m src \
  --input data/input/function_calling_tests.json \
  --functions_definition data/input/functions_definition.json \
  --output data/output/function_calling_results.json
```

## Resources

### Technical references
- Python documentation: https://docs.python.org/3/
- Pydantic documentation: https://docs.pydantic.dev/
- NumPy documentation: https://numpy.org/doc/
- JSON RFC 8259: https://www.rfc-editor.org/rfc/rfc8259
- OpenAI function calling concepts: https://platform.openai.com/docs/guides/function-calling

### AI usage disclosure

AI assistance was used for:
- drafting and improving documentation,
- reviewing code structure and robustness,
- suggesting refactors and docstring improvements.

All final integration decisions, project architecture, and validation were reviewed and applied manually in the repository.
