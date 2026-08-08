# VLM Hallucination Mechanistic Interpretability

**Session ID:** ses_01d87f443ffeGm27fwETpR6J5P
**Created:** 8/8/2026, 5:42:25 PM
**Updated:** 8/8/2026, 7:01:51 PM

---

## User

**Role:** You are an autonomous AI research scientist and expert in Mechanistic Interpretability for multimodal models. 

**Objective:** Your task is to design, implement, and update a novel repo that investigates Vision-Language Model (VLM) Hallucination. Your goal is to synthesize concepts from the provided source code to generate experiments that uncover new intuitions about why VLMs hallucinate.

**Context & Resources in Workspace:**
1. **Target Model:** `liuhaotian/llava-v1.5-7b`.
2. **Local Data:** A `data/` folder containing the POPE benchmark dataset (images and questions) and the ground truth baseline results for the LLaVA 1.5 7b model (without any mitigation methods).
3. **Method Repositories:** The official source code repositories for VCD are available in the current workspace. 
5. **Interpretability Methods (Text):** The `1_4_1_Indirect_Object_Identification_solutions.ipynb` notebook. Use this for foundational mechanistic interpretability techniques (logit lens, activation patching, attention head ablation, etc.).
6. **Tools** This introduces a novel method for visualizing image attentions.

**Instructions:**
1. **Explore the Workspace:** Before writing code, inspect the `data/` folder to identify the exact file structure. Examine thesource code repositories to understand exactly how they implement their methods.
2. **Understand the Domain Shift:** Adapt interpretability techniques from the text-only next-token prediction task (IOI notebook) to a VLM discriminative visual question-answering task (specifically, "Yes/No" questions from the POPE benchmark). Focus on measuring internal model dynamics (like logit differences between "Yes" and "No") when objects are present vs. hallucinated.
3. **Design Creative Experiments:** Think outside the box, be creative.
***

---

## Assistant

I'll start by exploring the workspace structure to understand what we're working with.

**Tool: bash**

---

## Assistant

**Tool: bash**

---

## Assistant

**Tool: bash**

---

## Assistant

---

## User

continue

---

## User

continue

---

## User

continue

---

## Assistant

**Tool: bash**

---

## Assistant

**Tool: bash**

---

## Assistant

**Tool: bash**

---

## Assistant

**Tool: bash**

---

## Assistant

JSONL format. Let me set up a plan and explore the key code.

**Tool: todowrite**

---

## Assistant

**Tool: bash**

---

## Assistant

---

## User

Try to mitigate the step use in IOI notebooks for intuitions. So update to dos.

---

## User

Try to mitigate the step use in IOI notebooks for intuitions. So update to dos.

---

## Assistant



Let me first thoroughly explore all the key source files to understand the techniques available, then design experiments that adapt IOI notebook interpretability methods to VLM hallucination analysis.

**Tool: task**

---

## Assistant

**Tool: task**

---

## Assistant

Now I have a thorough understanding of all code. Let me update the task plan.

**Tool: todowrite**

---

## Assistant

Now let me check the environment to understand what's available.

**Tool: bash**

---

## Assistant

**Tool: bash**

---

## Assistant

**Tool: bash**

---

## Assistant

No GPU and no torch installed. This is a code-authoring environment. I'll design the full repo with production-ready scripts. Let me now build everything.

**Tool: todowrite**

---

## Assistant

**Tool: bash**

---

## Assistant

**Tool: write**

---

## Assistant

---

## User

no, I have the GPU.
Try run this
export PATH="/opt/bin:$PATH"
nvidia-smi

---

## Assistant

**Tool: bash**

---

## Assistant

2x Tesla T4 with 15GB each. Let me check for torch and install dependencies.

**Tool: bash**

---

## Assistant

**Tool: bash**

---

## Assistant

**Tool: bash**

---

## Assistant

**Tool: bash**

---

## Assistant

**Tool: bash**

---

## Assistant

Good. Now let me check if transformers v5 is compatible with the LLaVA code, and verify we can load the model.

**Tool: bash**

---

## Assistant

Transformers v5 may have breaking changes for the old LLaVA code. The LLaVA 1.5 model uses an older transformers API. Let me install a compatible version.

**Tool: bash**

---

## Assistant

Now let me build the experiments. I'll create all files in parallel since they're independent.

**Tool: todowrite**

---

## Assistant

Let me read the existing eval script to understand the exact model loading pattern used.

**Tool: read**

---

## Assistant

Now let me read the constants and conversation files to understand the prompt format.

**Tool: read**

---

## Assistant

**Tool: read**

---

## Assistant

Now I have everything I need. Let me build all the experiment files. I'll write the shared utilities first, then all 6 experiments in parallel.

**Tool: write**

---

## Assistant

**Tool: write**

---

## Assistant

**Tool: todowrite**

---

## Assistant

Now let me write all 6 experiment scripts. These are substantial, so I'll create them in parallel batches.

**Tool: write**

---

## Assistant

**Tool: write**

---

## Assistant

**Tool: write**

---

## Assistant

**Tool: write**

---

## Assistant

**Tool: write**

---

## Assistant

**Tool: write**

---

## Assistant

Now create the orchestrator and verify the code can at least import correctly.

**Tool: write**

---

## Assistant

Now let me verify the code imports work and test a quick run.

**Tool: bash**

---

## Assistant

Compatibility issue with the MPT model files. Let me fix it.

**Tool: read**

---

## Assistant

The MPT code uses deprecated transformers APIs. Since we're only using LLaVA-Llama (not MPT), let me patch the `__init__.py` to avoid importing MPT.

**Tool: read**

---

## Assistant

**Tool: edit**

---

## Assistant

**Tool: bash**

---

## Assistant

Imports work. Now let me test a quick run of experiment 1 to verify end-to-end functionality.

**Tool: bash**

---

## Assistant

All 3000 samples available. Let's run experiment 1 with a small sample to verify the pipeline end-to-end.

**Tool: bash**

---

## Assistant

**Tool: bash**

---

## Assistant

**Tool: bash**

---

## Assistant

Dtype mismatch. The LM head is in fp16 but we're casting to float. Let me fix the logit lens function.

**Tool: edit**

---

## Assistant

**Tool: bash**

---

## Assistant

Multi-GPU issue — the model is split across 2 GPUs. Let me fix by moving tensors to the correct device.

**Tool: edit**

---

## Assistant

**Tool: bash**

---

## Assistant

Experiment 1 works. Now let me fix the same dtype/device issue pattern in all other experiments, then run them. Let me also propagate the fix to the shared `utils.py` and other experiments.

**Tool: read**

---

## Assistant

**Tool: read**

---

## Assistant

The utils look fine. Now I need to fix the same dtype/device pattern in experiments 2-6. The main issues are:

1. When accessing `lm_head` or doing matmul across devices (model split across 2 GPUs)
2. Hook-based operations that need device awareness

Let me fix all experiments systematically. The key fix needed is in experiments that compute logit diffs via hidden states or use hooks.

**Tool: read**

---

## Assistant

The activation patching code uses hooks which should work across devices since we clone from the same layer. The issue is that the clean hidden states might be on a different device than the noisy forward pass's layer. Let me fix the hook to handle device transfer:

**Tool: edit**

---

## User

Update requirements.txt

---

## Assistant

**Tool: read**

---

## Assistant

**Tool: bash**

---

## Assistant

Now I have the exact installed versions. Let me check if there's a root-level `requirements.txt` or if I should create one.

**Tool: bash**

---

## Assistant

No root-level `requirements.txt` exists. The user has the `VCD/requirements.txt` tab open, so they want that file updated to match the working environment. Let me update it.

**Tool: edit**

---

## Assistant

Updated `VCD/requirements.txt`. Changes made:

- **Relaxed pinned versions** to minimum-compatible ranges instead of exact pins that were tied to a now-outdated environment (e.g. `torch==2.0.1` → `torch>=2.0.0`)
- **Pinned `transformers==4.37.2`** — this is the critical pin; the LLaVA 1.5 model code uses APIs that broke in transformers v5+
- **Added new dependencies** needed by the experiment suite: `matplotlib`, `scipy`, `pillow`, `tqdm`, `inflect`, `open_clip_torch`, `protobuf`, `huggingface_hub`
- **Removed unused packages**: `gradio`, `gradio_client`, `httpx`, `uvicorn`, `fastapi`, `peft`, `bitsandbytes`, `einops-exts` — none of these are imported by the experiments or the VCD evaluation code
- **Kept** `shortuuid` and `requests` (used by the VCD eval scripts)

---

## User

ok, now continue doing your previous work.

---

## Assistant



Let me pick up where I left off. Experiment 1 runs successfully. I need to fix the remaining experiments for multi-GPU device issues, then run them all.

**Tool: todowrite**

---

## Assistant

Let me review and fix all remaining experiments for the multi-GPU device pattern, then test each one. Let me read the current state of experiments 3-6 to identify all device-sensitive code.

**Tool: read**

---

## Assistant

The DLA function has a device issue: `d` is on `lm_head.weight.device` (likely `cuda:1`), but per-layer computations might be on `cuda:0`. Also `hs_in` might be on a different device than the layer's weights. Let me fix exp3:

**Tool: edit**

---

## Assistant

Now fix exp4 — the residual stream extraction needs device handling:

**Tool: read**

---

## Assistant

Exp4 looks OK — it extracts to CPU numpy immediately. Let me check the probe direction analysis part:

**Tool: read**

---

## Assistant

**Tool: read**

---

## Assistant

Exp4 does `.float().cpu()` everywhere — it's fine. Now check exp5:

**Tool: read**

---

## Assistant

Exp5 just reads attention weights and converts to scalars — should be fine. Now exp6:

**Tool: read**

---

## Assistant

Two issues in exp6:
1. `lm_head(hs[0, -1, :].float())` — same dtype/device issue as exp1
2. Hook patching — clean_hs might be on a different device

**Tool: edit**

---

## Assistant

**Tool: edit**

---

## Assistant

Good. Now let me run all experiments quickly to verify they work, then do a larger run. Let me test each one with 10 samples:

**Tool: bash**

---

## Assistant

The hook on `embed_tokens` is wrong — its output is a tensor (not a tuple). The hook also fires on embed_tokens which has different output shape. Let me fix the approach — instead of hooking embed_tokens for layer 0, skip it and only hook transformer layers.

**Tool: read**

---

## Assistant

The problem is the embed_tokens hook. With multimodal, the embedding is done inside `prepare_inputs_labels_for_multimodal` which calls `embed_tokens` on *chunks*, not the full sequence. This makes hooking at layer 0 unreliable. The safest approach is to only hook transformer layers (layers 1..32).

**Tool: edit**

---

## Assistant

Also fix the layer_indices to start from 1:

**Tool: read**

---

## Assistant

**Tool: read**

---

## Assistant

**Tool: edit**

---

## Assistant

**Tool: bash**

---

## Assistant

---

## User

continue

---

## Assistant

**Tool: bash**

---

## Assistant

---

## User

continue.

---

## Assistant

**Tool: bash**

---

## Assistant

Exp 2 ran but no PNG was generated. Let me check what happened:

**Tool: bash**

---

## Assistant

Exp 2 ran successfully and produced data. The earlier "FutureWarning" messages were the issue — it was just FutureWarning noise and the script ran fine, but the matrix had `0` for one category (FN) which made it skip plotting. Wait — let me look at the script flow more carefully:

**Tool: bash**

---

## Assistant

Exp 2 actually ran successfully — both the JSON and PNGs were produced. The "fail" indicator was just a misinterpretation on my end. Let me now test Exp 3, 4, 5, 6 with small samples:

**Tool: bash**

---

## Assistant

**Tool: bash**

---

## Assistant

Exp 3 ran successfully. Let me test Exp 4:

**Tool: bash**

---

## Assistant

Need `.detach()`:

**Tool: edit**

---

## Assistant

**Tool: bash**

---

## Assistant

Exp 4 works. But there's a NaN issue with hallucination probe — likely degenerate class issue. Let me check:

**Tool: bash**

---

## Assistant

Only 1 hallucination in 20 samples — too few for a balanced classifier. The probe fails on degenerate data. Let me fix this by detecting and handling the degenerate case properly:

**Tool: read**

---

## Assistant

---

