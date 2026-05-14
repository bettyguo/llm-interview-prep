# Inference & Serving — aggregated references

## Foundational

- [Pope et al. — "Efficiently Scaling Transformer Inference"](https://arxiv.org/abs/2211.05102) — canonical inference math.
- [Kwon et al. — "vLLM / PagedAttention"](https://arxiv.org/abs/2309.06180) — paged attention.
- [Yu et al. — "Orca"](https://www.usenix.org/conference/osdi22/presentation/yu) — continuous batching.

## Speculative decoding

- [Leviathan, Kalman, Matias — "Fast Inference from Transformers via Speculative Decoding"](https://arxiv.org/abs/2211.17192).
- [Chen et al. — "Accelerating LLM Decoding with Speculative Sampling"](https://arxiv.org/abs/2302.01318).
- [Cai et al. — "Medusa"](https://arxiv.org/abs/2401.10774).
- [Li et al. — "EAGLE"](https://arxiv.org/abs/2401.15077).

## Quantization

- [Frantar et al. — "GPTQ"](https://arxiv.org/abs/2210.17323).
- [Lin et al. — "AWQ"](https://arxiv.org/abs/2306.00978).
- [Xiao et al. — "SmoothQuant"](https://arxiv.org/abs/2211.10438).
- [Micikevicius et al. — "FP8 Formats"](https://arxiv.org/abs/2209.05433).
- [Liu et al. — "KIVI (KV cache quantization)"](https://arxiv.org/abs/2402.02750).

## Latency / scheduling

- [Patel et al. — "Splitwise"](https://arxiv.org/abs/2311.18677) — phase splitting.
- [Agrawal et al. — "Sarathi-Serve" (chunked prefill)](https://arxiv.org/abs/2403.02310).
- [Jiang et al. — "LLMLingua"](https://arxiv.org/abs/2310.05736) — prompt compression.

## Structured generation

- [Willard & Louf — "Outlines"](https://arxiv.org/abs/2307.09702).
- [Dong et al. — "XGrammar"](https://arxiv.org/abs/2411.15100).
- [OpenAI Structured Outputs docs](https://platform.openai.com/docs/guides/structured-outputs).

## Multi-LoRA / multi-tenant serving

- [Chen et al. — "Punica"](https://arxiv.org/abs/2310.18547).
- [Sheng et al. — "S-LoRA"](https://arxiv.org/abs/2311.03285).

## Decoding strategies

- [Holtzman et al. — "The Curious Case of Neural Text Degeneration"](https://arxiv.org/abs/1904.09751) — top-p / beam-search critique.
- [Fan, Lewis, Dauphin — "Hierarchical Neural Story Generation"](https://arxiv.org/abs/1805.04833) — top-k.

## Serving stacks (primary docs)

- [vLLM project](https://github.com/vllm-project/vllm).
- [TensorRT-LLM project](https://github.com/NVIDIA/TensorRT-LLM).
- [SGLang project](https://github.com/sgl-project/sglang) and [SGLang paper](https://arxiv.org/abs/2312.07104).
- [TGI project](https://github.com/huggingface/text-generation-inference).

## Caching

- [Anthropic Prompt Caching docs](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching).
- [OpenAI Prompt Caching docs](https://platform.openai.com/docs/guides/prompt-caching).
