# Local LoRA Q&A Chatbot

A fully local, zero-cloud pipeline that fine-tunes a language model on your PDF and lets you chat with it.

## Pipeline

```
PDF → Ollama (Q&A generation) → LoRA fine-tune SmolLM2-135M (CPU) → Chat
```

## Requirements

- [Ollama](https://ollama.com) running locally with `mistral` pulled:
  ```bash
  ollama pull mistral
  ```
- Python 3.11+

## Setup

```bash
pip install -r requirements.txt
python app.py
```

Open `http://localhost:7861` in your browser.

## Usage

| Step | Action |
|---|---|
| **Step 1** | Upload a PDF, give it a project name, click **Start Training** |
| **Step 2** | Watch live logs — Ollama generates Q&A pairs, then LoRA trains |
| **Step 3** | Chat with your fine-tuned model once training completes |

## Training details

| Setting | Value |
|---|---|
| Base model | `HuggingFaceTB/SmolLM2-135M-Instruct` |
| Fine-tuning | LoRA (r=8, α=16) via `trl` SFTTrainer |
| Q&A generation | `mistral:latest` via Ollama |
| Hardware | CPU (no GPU required) |
| Estimated time | 10–30 min depending on PDF size |

## Notes

- Trained models are saved to `./models/<project-name>/`
- No internet required after first model download
- Fine-tuned model is weak on small datasets — this is a proof of concept
