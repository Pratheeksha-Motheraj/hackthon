# ================================================================
# SINGLE-CELL FULL PROJECT (FASTAPI + GRADIO + GRANITE NLP)
# ================================================================

!pip -q install fastapi uvicorn[standard] gradio requests transformers accelerate huggingface-hub sentencepiece safetensors

import os, json, time, threading, subprocess, sys, re, textwrap
from typing import List, Optional, Dict, Any
from fastapi import FastAPI
from pydantic import BaseModel
import requests

# -------------------------
# HuggingFace Granite Setup
# -------------------------
HF_TOKEN = os.environ.get("HF_TOKEN")  # Set if available
USE_HF = bool(HF_TOKEN)

print("Initializing NLP Model...")

model_client = None
local_pipeline = None
HF_MODEL_ID = "ibm-granite/granite-3.3-2b-instruct"

if USE_HF:
    try:
        from huggingface_hub import InferenceClient
        model_client = InferenceClient(token=HF_TOKEN)
        print("Using Granite via HuggingFace Inference API.")
    except:
        USE_HF = False

if not USE_HF:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    tok = AutoTokenizer.from_pretrained("distilgpt2")
    mdl = AutoModelForCausalLM.from_pretrained("distilgpt2")
    local_pipeline = pipeline("text-generation", model=mdl, tokenizer=tok)
    print("Using fallback local NLP model (distilgpt2).")

# -------------------------
# LLM Extraction Function
# -------------------------
def extract_info(text: str):
    prompt = f"""
Return valid JSON only in this format:
{{
 "drugs": [
    {{"name":"...", "dose":"...", "frequency":"...", "route":"..."}}
 ],
 "notes": "..."
}}
Text: \"\"\"{text}\"\"\"
"""
    if USE_HF:
        try:
            out = model_client.text_generation(
                model=HF_MODEL_ID,
                inputs=prompt,
                max_new_tokens=250,
                temperature=0.0
            )
            if isinstance(out, dict) and "generated_text" in out:
                g = out["generated_text"]
            elif isinstance(out, list) and "generated_text" in out[0]:
                g = out[0]["generated_text"]
            else:
                g = str(out)

            m = re.search(r"(\{.*\})", g, re.DOTALL)
            return json.loads(m.group(1)) if m else {"drugs": [], "notes": g[:200]}
        except Exception as e:
            return {"drugs": [], "notes": f"HF Error: {e}"}

    # local fallback
    out = local_pipeline(prompt, max_new_tokens=150)[0]["generated_text"]
    m = re.search(r"(\{.*\})", out, re.DOTALL)
    if m:
        return json.loads(m.group(1))

    names = re.findall(r"[A-Za-z]{3,}", text)
    return {"drugs": [{"name": names[0] if names else "unknown", "dose": None, "freq": None, "route": None}], "notes": "fallback extractor"}

# -------------------------
# Interaction Logic
# -------------------------
LOCAL_INTERACTIONS = {
    frozenset(["warfarin", "ibuprofen"]): ("high", "Bleeding risk."),
    frozenset(["amoxicillin", "methotrexate"]): ("moderate", "Reduced methotrexate clearance."),
}

def check_interactions(drugs):
    drugs_lower = [d.lower() for d in drugs]
    out = []
    for i in range(len(drugs_lower)):
        for j in range(i+1, len(drugs_lower)):
            pair = frozenset([drugs_lower[i], drugs_lower[j]])
            if pair in LOCAL_INTERACTIONS:
                sev, desc = LOCAL_INTERACTIONS[pair]
                out.append({"pair": list(pair), "severity": sev, "description": desc})
    return out

# -------------------------
# Dosage Logic
# -------------------------
DOSAGE_DB = {
    "paracetamol": {"child_mgkg": 15, "adult_mg": 500},
    "ibuprofen": {"child_mgkg": 10, "adult_mg": 400},
    "amoxicillin": {"child_mgkg": 20, "adult_mg": 500},
}

def dosage(drug, age, weight=None):
    d = drug.lower()
    if weight is None:
        weight = 30 if age < 14 else 60

    if d in DOSAGE_DB:
        if age < 18:
            return {
                "drug": drug,
                "dose_mg": round(DOSAGE_DB[d]["child_mgkg"] * weight),
                "note": "child mg/kg rule"
            }
        return {"drug": drug, "dose_mg": DOSAGE_DB[d]["adult_mg"], "note": "standard adult dose"}

    return {"drug": drug, "dose_mg": round(10 * weight), "note": "generic mg/kg fallback"}

# -------------------------
# Alternatives
# -------------------------
ALTS = {
    "ibuprofen": ["naproxen", "paracetamol"],
    "amoxicillin": ["azithromycin", "doxycycline (adult)"],
}

def alternatives(drug):
    return ALTS.get(drug.lower(), [])

# -------------------------
# FastAPI Backend
# -------------------------
app = FastAPI()

class Req(BaseModel):
    age: float
    weight: Optional[float] = None
    free_text: Optional[str] = ""
    drugs: Optional[List[str]] = []

@app.post("/analyze")
def analyze(payload: Req):
    extracted = extract_info(payload.free_text) if payload.free_text else {"drugs": [], "notes": ""}
    ext_drugs = [d["name"] for d in extracted["drugs"] if d.get("name")]
    all_drugs = list(dict.fromkeys(payload.drugs + ext_drugs))

    return {
        "drugs": all_drugs,
        "extracted": extracted,
        "dosage": {d: dosage(d, payload.age, payload.weight) for d in all_drugs},
        "interactions": check_interactions(all_drugs),
        "alternatives": {d: alternatives(d) for d in all_drugs}
    }

# Run API thread
def run_api():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

threading.Thread(target=run_api, daemon=True).start()
time.sleep(2)
print("FastAPI running at http://127.0.0.1:8000")

# -------------------------
# GRADIO UI
# -------------------------
import gradio as gr

def ui_fn(age, weight, free_text, drugs_csv):
    drugs = [d.strip() for d in drugs_csv.split(",") if d.strip()]
    payload = {
        "age": float(age),
        "weight": float(weight) if weight else None,
        "free_text": free_text,
        "drugs": drugs
    }
    r = requests.post("http://127.0.0.1:8000/analyze", json=payload)
    return json.dumps(r.json(), indent=2)

gr.Interface(
    fn=ui_fn,
    inputs=[
        gr.Number(label="Age (years)", value=30),
        gr.Number(label="Weight (kg, optional)", value=0),
        gr.Textbox(label="Free Text Prescription", lines=4),
        gr.Textbox(label="Drug List (comma separated)")
    ],
    outputs=gr.Textbox(label="Analysis Output"),
    title="Drug Interaction & Dosage Analyzer (Granite + FastAPI)"
).launch(share=True)
