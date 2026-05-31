"""
Local LoRA Q&A Chatbot
PDF → Ollama Q&A generation → LoRA fine-tune SmolLM2-135M on CPU → chat
"""

import io, json, os, queue, shutil, threading, time, traceback

import gradio as gr
import requests
import torch
from datasets import Dataset
from peft import LoraConfig
from pypdf import PdfReader
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTConfig, SFTTrainer

# ── Config ────────────────────────────────────────────────────────────────────
OLLAMA_URL        = "http://localhost:11434"
OLLAMA_MODEL      = "llama3.2:latest"    # 2 GB — ~2× faster than mistral on CPU
BASE_MODEL        = "HuggingFaceTB/SmolLM2-135M-Instruct"
MODELS_DIR        = os.path.join(os.path.dirname(__file__), "models")
OLLAMA_TIMEOUT    = 300                  # seconds per chunk (5 min — generous for CPU)
QA_CHUNK_CHARS    = 800                  # shorter chunks → faster Ollama response
QA_PER_CHUNK      = 3                    # pairs per chunk
MAX_CHUNKS        = 12                   # cap total chunks processed
os.makedirs(MODELS_DIR, exist_ok=True)

# ── Shared state ──────────────────────────────────────────────────────────────
_state = {
    "phase":        "upload",   # upload | training | chat
    "project_name": None,
    "model_path":   None,
    "qa_count":     0,
    "log_lines":    [],
    "error":        None,
    "progress":     0.0,
    "total_steps":  1,
    "current_step": 0,
}
_log_q        = queue.Queue()
_model_cache  = {"path": None, "model": None, "tok": None}


# ── PDF extraction ─────────────────────────────────────────────────────────────

def extract_pages(pdf_bytes: bytes) -> list[str]:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    return [
        p.extract_text().strip()
        for p in reader.pages
        if p.extract_text() and p.extract_text().strip()
    ]


# ── Q&A generation via Ollama ──────────────────────────────────────────────────

def _parse_qa(text: str) -> list[dict]:
    """Parse Q&A pairs tolerant of numbering/indentation from Ollama."""
    import re
    # Match lines containing Q: or A: regardless of leading prefix/whitespace/number
    q_pat = re.compile(r'(?:^|\d+[:.]\s*)Q:\s*(.+)', re.IGNORECASE)
    a_pat = re.compile(r'(?:^|\d+[:.]\s*)A:\s*(.+)', re.IGNORECASE)
    pairs = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        qm = q_pat.search(lines[i])
        if qm and i + 1 < len(lines):
            # Look ahead for the A: line (may be next or skip blank lines)
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines):
                am = a_pat.search(lines[j])
                if am:
                    q, a = qm.group(1).strip(), am.group(1).strip()
                    if q and a:
                        pairs.append({"text": f"User: {q}\nAssistant: {a}"})
                    i = j + 1
                    continue
        i += 1
    return pairs


def generate_qa(pages: list[str]) -> list[dict]:
    pairs   = []
    chunks  = pages[:MAX_CHUNKS]
    total   = len(chunks)
    for idx, chunk in enumerate(chunks, 1):
        _log_q.put(f"  Q&A chunk {idx}/{total} ({len(chunk)} chars)...")
        prompt = (
            f"Read the text below and write {QA_PER_CHUNK} question-answer pairs.\n"
            "Output ONLY this format, nothing else:\n"
            "Q: <question>\nA: <answer>\n\n"
            f"TEXT:\n{chunk[:QA_CHUNK_CHARS]}\n\nQ&A pairs:"
        )
        try:
            r = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
                timeout=OLLAMA_TIMEOUT,
            )
            out         = r.json().get("response", "")
            chunk_pairs = _parse_qa(out)
            pairs.extend(chunk_pairs)
            _log_q.put(f"    → {len(chunk_pairs)} pair(s) (total so far: {len(pairs)})")
            if not chunk_pairs:
                _log_q.put(f"    [warn] No pairs parsed. Raw: {out[:80]!r}")
        except requests.exceptions.Timeout:
            _log_q.put(
                f"    [warn] Chunk {idx} timed out after {OLLAMA_TIMEOUT}s — skipping. "
                f"Consider using a smaller model or reducing QA_CHUNK_CHARS."
            )
        except Exception as e:
            _log_q.put(f"    [warn] Chunk {idx} failed: {e}")
    return pairs


# ── Training callback ──────────────────────────────────────────────────────────

class _LogCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        _state["total_steps"]  = state.max_steps
        _state["current_step"] = 0
        _log_q.put(f"Training started — {state.max_steps} steps total.")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        loss = logs.get("loss", logs.get("train_loss"))
        step = state.global_step
        total = state.max_steps or 1
        _state["current_step"] = step
        _state["progress"]     = step / total
        epoch = logs.get("epoch", "?")
        msg = f"Step {step}/{total} | epoch {epoch}"
        if isinstance(loss, float):
            msg += f" | loss {loss:.4f}"
        _log_q.put(msg)

    def on_epoch_end(self, args, state, control, **kwargs):
        _log_q.put(f"── Epoch {round(state.epoch)} complete ──")


# ── Background training worker ─────────────────────────────────────────────────

def _worker(all_pdf_bytes: list[bytes], project_name: str):
    _state.update({
        "log_lines": [], "error": None, "qa_count": 0,
        "phase": "training", "progress": 0.0,
    })
    model_path = os.path.join(MODELS_DIR, project_name)
    tmp_dir    = os.path.join(MODELS_DIR, f"_tmp_{project_name}")

    try:
        # 1. Extract all PDFs and merge pages
        _log_q.put(f"Extracting text from {len(all_pdf_bytes)} PDF(s)...")
        pages = []
        for idx, pdf_bytes in enumerate(all_pdf_bytes, 1):
            doc_pages = extract_pages(pdf_bytes)
            _log_q.put(f"  PDF {idx}: {len(doc_pages)} page(s)")
            pages.extend(doc_pages)
        _log_q.put(f"Total: {len(pages)} page(s) across {len(all_pdf_bytes)} file(s).")

        # 2. Q&A via Ollama
        _log_q.put(
            f"Generating Q&A pairs with Ollama ({OLLAMA_MODEL}) — "
            f"timeout {OLLAMA_TIMEOUT}s/chunk, {QA_CHUNK_CHARS} chars/chunk..."
        )
        pairs = generate_qa(pages)
        _state["qa_count"] = len(pairs)
        if not pairs:
            _state["error"] = "No Q&A pairs generated — try a PDF with more readable text."
            _state["phase"] = "upload"
            return
        _log_q.put(f"Generated {len(pairs)} Q&A pairs.")

        # 3. Load base model
        _log_q.put(f"Downloading / loading {BASE_MODEL}...")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.float32, low_cpu_mem_usage=True
        )
        _log_q.put("Base model loaded.")

        # 4. Dataset
        dataset = Dataset.from_list(pairs)

        # 5. LoRA
        lora_cfg = LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.05,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            bias="none", task_type="CAUSAL_LM",
        )

        # 6. Training args
        steps_per_epoch = max(1, len(pairs) // 8)   # effective batch = grad_accum × batch
        _log_q.put(
            f"Training: {len(pairs)} samples | eff. batch 8 | "
            f"~{steps_per_epoch} steps/epoch × 3 epochs"
        )
        train_cfg = SFTConfig(
            output_dir=tmp_dir,
            num_train_epochs=3,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            max_seq_length=256,
            dataset_text_field="text",
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            fp16=False,
            bf16=False,
            dataloader_num_workers=0,  # avoid multiprocessing on Windows
        )

        # 7. Train
        trainer = SFTTrainer(
            model=model,
            args=train_cfg,
            train_dataset=dataset,
            peft_config=lora_cfg,
            callbacks=[_LogCallback()],
        )
        trainer.train()
        _log_q.put("Training done. Merging LoRA weights into base model...")

        # 8. Merge + save
        merged = trainer.model.merge_and_unload()
        shutil.rmtree(model_path, ignore_errors=True)
        os.makedirs(model_path, exist_ok=True)
        merged.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        shutil.rmtree(tmp_dir, ignore_errors=True)

        _log_q.put(f"Model saved to {model_path}")
        _state.update({
            "model_path": model_path,
            "phase":      "chat",
            "progress":   1.0,
        })

    except Exception as e:
        _state["error"] = str(e)
        _log_q.put(f"ERROR: {e}")
        _log_q.put(traceback.format_exc()[-800:])
        _state["phase"] = "upload"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ── Drain log queue into state ─────────────────────────────────────────────────

def _drain_logs():
    while not _log_q.empty():
        line = _log_q.get_nowait()
        _state["log_lines"].append(line)
    _state["log_lines"] = _state["log_lines"][-120:]


# ── Model discovery ───────────────────────────────────────────────────────────

def list_models() -> list[str]:
    """Return names of completed fine-tuned models in MODELS_DIR."""
    if not os.path.isdir(MODELS_DIR):
        return []
    return sorted(
        d for d in os.listdir(MODELS_DIR)
        if not d.startswith("_")                                   # exclude tmp dirs
        and os.path.isdir(os.path.join(MODELS_DIR, d))
        and os.path.exists(os.path.join(MODELS_DIR, d, "config.json"))
    )


# ── Chat inference ─────────────────────────────────────────────────────────────

def _load_model(path: str):
    if _model_cache["path"] == path:
        return _model_cache["model"], _model_cache["tok"]
    tok   = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32)
    model.eval()
    _model_cache.update({"path": path, "model": model, "tok": tok})
    return model, tok


def chat_fn(message: str, history: list[dict], selected_model: str | None = None):
    # Prefer dropdown selection; fall back to last trained model
    name = selected_model or _state.get("project_name")
    if not name:
        yield "No model selected — train one (Step 1) or pick one from the dropdown."
        return
    path = os.path.join(MODELS_DIR, name)
    if not os.path.exists(os.path.join(path, "config.json")):
        yield f"Model '{name}' not found in {MODELS_DIR}. Train it first."
        return

    model, tok = _load_model(path)

    # Build prompt in same format as training data
    prompt = ""
    for msg in history:
        role    = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            prompt += f"User: {content}\n"
        elif role == "assistant":
            prompt += f"Assistant: {content}\n"
    prompt += f"User: {message}\nAssistant:"

    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=512)
    try:
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tok.eos_token_id,
                repetition_penalty=1.1,
            )
        response = tok.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
        yield response or "(empty response — try rephrasing)"
    except Exception as e:
        yield f"Generation error: {e}"


# ── Gradio handlers (all return immediately) ───────────────────────────────────

def start_training(pdf_files, project_name_input):
    if not pdf_files:
        return gr.update(), "Please upload at least one PDF file."
    if _state["phase"] == "training":
        return gr.update(), "Training already running — check the Progress tab."

    # Normalise to list (Gradio may pass a single path string or a list)
    if isinstance(pdf_files, (str, bytes)):
        pdf_files = [pdf_files]

    name = (project_name_input or "").strip().replace(" ", "-").lower()
    if not name:
        import uuid
        name = f"model-{uuid.uuid4().hex[:6]}"

    _state["project_name"] = name

    # Read all PDFs into memory before spawning the thread
    all_bytes = []
    for f in pdf_files:
        all_bytes.append(open(f, "rb").read() if isinstance(f, str) else f)

    threading.Thread(target=_worker, args=(all_bytes, name), daemon=True).start()

    n = len(all_bytes)
    return gr.update(selected="progress"), f"Started '{name}' with {n} PDF(s). Switch to the Progress tab."


def get_progress():
    _drain_logs()
    phase    = _state["phase"]
    err      = _state["error"]
    logs     = "\n".join(_state["log_lines"][-40:]) or "Waiting..."
    pct      = int(_state["progress"] * 100)
    prog_bar = f"[{'#' * (pct // 5)}{' ' * (20 - pct // 5)}] {pct}%"

    if err:
        status        = f"ERROR: {err}"
        ready         = False
        dropdown_upd  = gr.update()
    elif phase == "upload":
        status        = "Idle — upload PDF(s) to begin."
        ready         = False
        dropdown_upd  = gr.update()
    elif phase == "training":
        status = (
            f"Training `{BASE_MODEL}` on {_state['qa_count']} Q&A pairs\n"
            f"{prog_bar}\n"
            f"Step {_state['current_step']}/{_state['total_steps']}"
        )
        ready        = False
        dropdown_upd = gr.update()
    elif phase == "chat":
        status       = f"Done! Model saved to {_state['model_path']}"
        ready        = True
        # Auto-populate + select the freshly trained model in the dropdown
        models       = list_models()
        new_name     = _state.get("project_name")
        dropdown_upd = gr.update(choices=models, value=new_name if new_name in models else (models[0] if models else None))
    else:
        status, ready, dropdown_upd = "", False, gr.update()

    return status, logs, gr.update(visible=ready), dropdown_upd


# ── UI ─────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="Local LoRA Chatbot") as demo:

    gr.Markdown(
        "# Local LoRA Q&A Chatbot\n"
        "**PDF → Ollama Q&A generation → LoRA fine-tune SmolLM2-135M (CPU) → chat**\n\n"
        f"Q&A model: `{OLLAMA_MODEL}` ({OLLAMA_TIMEOUT}s timeout, {QA_CHUNK_CHARS} chars/chunk) | "
        f"Train model: `{BASE_MODEL}`"
    )

    with gr.Tabs() as tabs:

        # Tab 1 — Upload
        with gr.Tab("Step 1 — Upload & Train", id="upload"):
            gr.Markdown(
                "Upload one or more PDFs. Ollama generates Q&A pairs from all of them, "
                "then LoRA fine-tuning runs locally on the combined dataset. "
                "Expect **10–30 min** on CPU."
            )
            pdf_input  = gr.File(
                label="PDF files (select multiple)",
                file_types=[".pdf"],
                file_count="multiple",
            )
            proj_input = gr.Textbox(
                label="Model name (blank = auto)", placeholder="my-qa-model"
            )
            train_btn  = gr.Button("Start Training", variant="primary", size="lg")
            upload_msg = gr.Markdown("")

        # Tab 2 — Progress
        with gr.Tab("Step 2 — Training Progress", id="progress"):
            status_box = gr.Textbox(label="Status", lines=4, interactive=False)
            log_box    = gr.Textbox(label="Live log", lines=20, interactive=False)
            open_btn   = gr.Button("Open Chat", variant="primary", visible=False)
            gr.Markdown(
                "_Auto-refreshes every 10 seconds. "
                "The 'Open Chat' button appears when training completes._"
            )

        # Tab 3 — Chat
        with gr.Tab("Step 3 — Chat", id="chat"):
            gr.Markdown(
                "Select a fine-tuned model from the dropdown, then ask questions. "
                "The model was trained on Q&A pairs extracted from your PDF(s)."
            )
            # Model selector row — sits above the chat interface
            with gr.Row():
                model_dropdown = gr.Dropdown(
                    label="Fine-tuned model",
                    choices=list_models(),
                    value=None,
                    interactive=True,
                    scale=4,
                )
                refresh_models_btn = gr.Button("Refresh", size="sm", scale=1)

            # gr.State holds the selected model name and is passed as additional input
            model_state = gr.State(None)

            chatbot = gr.ChatInterface(
                fn=chat_fn,
                additional_inputs=[model_state],
                chatbot=gr.Chatbot(height=400),
                textbox=gr.Textbox(
                    placeholder="Ask a question...",
                    container=False,
                    submit_btn="Enter",
                ),
            )
            reset_btn = gr.Button("Train a new model")

    # Wiring
    train_btn.click(
        fn=start_training,
        inputs=[pdf_input, proj_input],
        outputs=[tabs, upload_msg],
    )

    open_btn.click(fn=lambda: gr.update(selected="chat"), outputs=[tabs])
    reset_btn.click(fn=lambda: gr.update(selected="upload"), outputs=[tabs])

    # Dropdown → State (so ChatInterface fn receives the selection)
    model_dropdown.change(
        fn=lambda v: v,
        inputs=[model_dropdown],
        outputs=[model_state],
    )

    # Refresh button re-scans MODELS_DIR
    refresh_models_btn.click(
        fn=lambda: gr.update(choices=list_models()),
        outputs=[model_dropdown],
    )

    # Timer: also updates dropdown when a new model finishes training
    gr.Timer(10).tick(
        fn=get_progress,
        outputs=[status_box, log_box, open_btn, model_dropdown],
    )


if __name__ == "__main__":
    print(f"Ollama: {OLLAMA_URL}  model: {OLLAMA_MODEL}")
    print(f"Train model: {BASE_MODEL}")
    print(f"Models saved to: {MODELS_DIR}")
    demo.launch(server_name="0.0.0.0", server_port=7861, share=False, theme=gr.themes.Soft())
