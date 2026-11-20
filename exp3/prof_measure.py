#!/usr/bin/env python3
"""Dev tool: scorer equivalence check and timing on the base model + a fresh (zero) LoRA."""
import faulthandler, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from data import PERSONA_PAIRS
from measure import Prober
faulthandler.cancel_dump_traceback_later()
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", dtype=getattr(torch, sys.argv[1] if len(sys.argv) > 1 else "bfloat16"))
model = get_peft_model(model, LoraConfig(r=8, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"], bias="none", task_type="CAUSAL_LM"))
model.to("cuda:0").eval()
pr = Prober(tok, PERSONA_PAIRS)
print(json.dumps(pr.check_vs_full_sequence(model), indent=1))
if len(sys.argv) > 1: sys.exit()
t = time.time(); r = pr.measure(model); print("measure", round(time.time() - t, 1), "s; bird", r["t_bird_s"])
