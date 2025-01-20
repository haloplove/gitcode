# Ollama Model File

> [!NOTE] > `Modelfile` syntax is in development

A model file is the blueprint to create and share models with Ollama.

## Table of Contents

- [Format](#format)
- [Examples](#examples)
- [Instructions](#instructions)
  - [FROM (Required)](#from-required)
    - [Build from existing model](#build-from-existing-model)
    - [Build from a Safetensors model](#build-from-a-safetensors-model)
    - [Build from a GGUF file](#build-from-a-gguf-file)
  - [PARAMETER](#parameter)
    - [Valid Parameters and Values](#valid-parameters-and-values)
  - [TEMPLATE](#template)
    - [Template Variables](#template-variables)
  - [SYSTEM](#system)
  - [ADAPTER](#adapter)
  - [LICENSE](#license)
  - [MESSAGE](#message)
- [Notes](#notes)

## Format

The format of the `Modelfile`:

```modelfile
# comment
INSTRUCTION arguments
```

| Instruction                         | Description                                                    |
| ----------------------------------- | -------------------------------------------------------------- |
| [`FROM`](#from-required) (required) | Defines the base model to use.                                 |
| [`PARAMETER`](#parameter)           | Sets the parameters for how Ollama will run the model.         |
| [`TEMPLATE`](#template)             | The full prompt template to be sent to the model.              |
| [`SYSTEM`](#system)                 | Specifies the system message that will be set in the template. |
| [`ADAPTER`](#adapter)               | Defines the (Q)LoRA adapters to apply to the model.            |
| [`LICENSE`](#license)               | Specifies the legal license.                                   |
| [`MESSAGE`](#message)               | Specify message history.                                       |

## Examples

### Basic `Modelfile`

An example of a `Modelfile` creating a mario blueprint:

```modelfile
FROM llama3.2
# sets the temperature to 1 [higher is more creative, lower is more coherent]
PARAMETER temperature 1
# sets the context window size to 4096, this controls how many tokens the LLM can use as context to generate the next token
PARAMETER num_ctx 4096

# sets a custom system message to specify the behavior of the chat assistant
SYSTEM You are Mario from super mario bros, acting as an assistant.
```

To use this:

1. Save it as a file (e.g. `Modelfile`)
2. `ollama create choose-a-model-name -f <location of the file e.g. ./Modelfile>`
3. `ollama run choose-a-model-name`
4. Start using the model!

More examples are available in the [examples directory](../examples).

To view the Modelfile of a given model, use the `ollama show --modelfile` command.

```bash
  > ollama show --modelfile llama3.2
  # Modelfile generated by "ollama show"
  # To build a new Modelfile based on this one, replace the FROM line with:
  # FROM llama3.2:latest
  FROM /Users/pdevine/.ollama/models/blobs/sha256-00e1317cbf74d901080d7100f57580ba8dd8de57203072dc6f668324ba545f29
  TEMPLATE """{{ if .System }}<|start_header_id|>system<|end_header_id|>

  {{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>

  {{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>

  {{ .Response }}<|eot_id|>"""
  PARAMETER stop "<|start_header_id|>"
  PARAMETER stop "<|end_header_id|>"
  PARAMETER stop "<|eot_id|>"
  PARAMETER stop "<|reserved_special_token"
```

## Instructions

### FROM (Required)

The `FROM` instruction defines the base model to use when creating a model.

```modelfile
FROM <model name>:<tag>
```

#### Build from existing model

```modelfile
FROM llama3.2
```

A list of available base models:
[https://github.com/ollama/ollama#model-library](https://github.com/ollama/ollama#model-library)
Additional models can be found at:
[https://ollama.com/library](https://ollama.com/library)

#### Build from a Safetensors model

```modelfile
FROM <model directory>
```

The model directory should contain the Safetensors weights for a supported architecture.

Currently supported model architectures:

- Llama (including Llama 2, Llama 3, Llama 3.1, and Llama 3.2)
- Mistral (including Mistral 1, Mistral 2, and Mixtral)
- Gemma (including Gemma 1 and Gemma 2)
- Phi3

#### Build from a GGUF file

```modelfile
FROM ./ollama-model.gguf
```

The GGUF file location should be specified as an absolute path or relative to the `Modelfile` location.

### PARAMETER

The `PARAMETER` instruction defines a parameter that can be set when the model is run.

```modelfile
PARAMETER <parameter> <parametervalue>
```

#### Valid Parameters and Values

| Parameter      | Description                                                                                                                                                                                                                                                                                                                                                                   | Value Type | Example Usage        |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | -------------------- |
| mirostat       | Enable Mirostat sampling for controlling perplexity. (default: 0, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)                                                                                                                                                                                                                                                               | int        | mirostat 0           |
| mirostat_eta   | Influences how quickly the algorithm responds to feedback from the generated text. A lower learning rate will result in slower adjustments, while a higher learning rate will make the algorithm more responsive. (Default: 0.1)                                                                                                                                              | float      | mirostat_eta 0.1     |
| mirostat_tau   | Controls the balance between coherence and diversity of the output. A lower value will result in more focused and coherent text. (Default: 5.0)                                                                                                                                                                                                                               | float      | mirostat_tau 5.0     |
| num_ctx        | Sets the size of the context window used to generate the next token. (Default: 2048)                                                                                                                                                                                                                                                                                          | int        | num_ctx 4096         |
| repeat_last_n  | Sets how far back for the model to look back to prevent repetition. (Default: 64, 0 = disabled, -1 = num_ctx)                                                                                                                                                                                                                                                                 | int        | repeat_last_n 64     |
| repeat_penalty | Sets how strongly to penalize repetitions. A higher value (e.g., 1.5) will penalize repetitions more strongly, while a lower value (e.g., 0.9) will be more lenient. (Default: 1.1)                                                                                                                                                                                           | float      | repeat_penalty 1.1   |
| temperature    | The temperature of the model. Increasing the temperature will make the model answer more creatively. (Default: 0.8)                                                                                                                                                                                                                                                           | float      | temperature 0.7      |
| seed           | Sets the random number seed to use for generation. Setting this to a specific number will make the model generate the same text for the same prompt. (Default: 0)                                                                                                                                                                                                             | int        | seed 42              |
| stop           | Sets the stop sequences to use. When this pattern is encountered the LLM will stop generating text and return. Multiple stop patterns may be set by specifying multiple separate `stop` parameters in a modelfile.                                                                                                                                                            | string     | stop "AI assistant:" |
| tfs_z          | Tail free sampling is used to reduce the impact of less probable tokens from the output. A higher value (e.g., 2.0) will reduce the impact more, while a value of 1.0 disables this setting. (default: 1)                                                                                                                                                                     | float      | tfs_z 1              |
| num_predict    | Maximum number of tokens to predict when generating text. (Default: -1, infinite generation)                                                                                                                                                                                                                                                                                  | int        | num_predict 42       |
| top_k          | Reduces the probability of generating nonsense. A higher value (e.g. 100) will give more diverse answers, while a lower value (e.g. 10) will be more conservative. (Default: 40)                                                                                                                                                                                              | int        | top_k 40             |
| top_p          | Works together with top-k. A higher value (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text. (Default: 0.9)                                                                                                                                                                                       | float      | top_p 0.9            |
| min_p          | Alternative to the top_p, and aims to ensure a balance of quality and variety. The parameter*p* represents the minimum probability for a token to be considered, relative to the probability of the most likely token. For example, with _p_=0.05 and the most likely token having a probability of 0.9, logits with a value less than 0.045 are filtered out. (Default: 0.0) | float      | min_p 0.05           |

### TEMPLATE

`TEMPLATE` of the full prompt template to be passed into the model. It may include (optionally) a system message, a user's message and the response from the model. Note: syntax may be model specific. Templates use Go [template syntax](https://pkg.go.dev/text/template).

#### Template Variables

| Variable          | Description                                                                                   |
| ----------------- | --------------------------------------------------------------------------------------------- |
| `{{ .System }}`   | The system message used to specify custom behavior.                                           |
| `{{ .Prompt }}`   | The user prompt message.                                                                      |
| `{{ .Response }}` | The response from the model. When generating a response, text after this variable is omitted. |

```
TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
"""
```

### SYSTEM

The `SYSTEM` instruction specifies the system message to be used in the template, if applicable.

```modelfile
SYSTEM """<system message>"""
```

### ADAPTER

The `ADAPTER` instruction specifies a fine tuned LoRA adapter that should apply to the base model. The value of the adapter should be an absolute path or a path relative to the Modelfile. The base model should be specified with a `FROM` instruction. If the base model is not the same as the base model that the adapter was tuned from the behaviour will be erratic.

#### Safetensor adapter

```modelfile
ADAPTER <path to safetensor adapter>
```

Currently supported Safetensor adapters:

- Llama (including Llama 2, Llama 3, and Llama 3.1)
- Mistral (including Mistral 1, Mistral 2, and Mixtral)
- Gemma (including Gemma 1 and Gemma 2)

#### GGUF adapter

```modelfile
ADAPTER ./ollama-lora.gguf
```

### LICENSE

The `LICENSE` instruction allows you to specify the legal license under which the model used with this Modelfile is shared or distributed.

```modelfile
LICENSE """
<license text>
"""
```

### MESSAGE

The `MESSAGE` instruction allows you to specify a message history for the model to use when responding. Use multiple iterations of the MESSAGE command to build up a conversation which will guide the model to answer in a similar way.

```modelfile
MESSAGE <role> <message>
```

#### Valid roles

| Role      | Description                                                  |
| --------- | ------------------------------------------------------------ |
| system    | Alternate way of providing the SYSTEM message for the model. |
| user      | An example message of what the user could have asked.        |
| assistant | An example message of how the model should respond.          |

#### Example conversation

```modelfile
MESSAGE user Is Toronto in Canada?
MESSAGE assistant yes
MESSAGE user Is Sacramento in Canada?
MESSAGE assistant no
MESSAGE user Is Ontario in Canada?
MESSAGE assistant yes
```

## Notes

- the **`Modelfile` is not case sensitive**. In the examples, uppercase instructions are used to make it easier to distinguish it from arguments.
- Instructions can be in any order. In the examples, the `FROM` instruction is first to keep it easily readable.

[1]: https://ollama.com/library