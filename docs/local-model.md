# Running the local model (Windows desktop with an NVIDIA GPU)

The bot works with no AI at all. With a local model it also understands messages
the rules are unsure about ("me and 3 college friends, two are still studying").
The rules brain always answers first; the model is asked only when the rules are
not sure, and if the model is down the bot carries on without it.

Default model: `qwen3:8b`, 8K context, thinking off (5.6 GB on the GPU, fits a 3060 Ti).
On a Colab T4 it scored better than `qwen3.5:9b-q8_0` and answered 3x faster (1.0 s vs 3.3 s).

## 1. One-time setup

```powershell
# GPU driver: update from nvidia.com or the NVIDIA app, then check it is seen
nvidia-smi

# Tools
winget install --id Ollama.Ollama -e
winget install --id Git.Git -e
winget install --id astral-sh.uv -e
# Close and reopen the terminal so the new commands are on PATH.

ollama --version
ollama pull qwen3:8b        # ~5 GB download, the default
ollama pull qwen3.5:9b      # optional, to compare
```

Optional, saves VRAM (then quit Ollama from the tray icon and start it again):

```powershell
setx OLLAMA_FLASH_ATTENTION 1
setx OLLAMA_KV_CACHE_TYPE q8_0
```

Do **not** set `OLLAMA_HOST=0.0.0.0`. Ollama has no login, so it must only
listen on this machine (the default, `127.0.0.1:11434`).

## 2. Get the code

```powershell
git clone https://github.com/AakashBhat1/gvs-chatbot.git C:\dev\gvs-chatbot
cd C:\dev\gvs-chatbot\backend
uv sync --locked
```

The repository is private: Git asks you to sign in to GitHub the first time.

## 3. Configure

All settings are environment variables. `setx` saves them for new terminals.

```powershell
setx GVS_BRAINS "rules,local"
setx GVS_LOCAL_MODEL "qwen3:8b"
# Open a new terminal after setx.
```

| Variable | Default | Meaning |
|---|---|---|
| `GVS_BRAINS` | `rules` | Brains to use, cheapest first. `rules` is always included. |
| `GVS_LOCAL_URL` | `http://127.0.0.1:11434` | Where Ollama listens. Keep it on this machine. |
| `GVS_LOCAL_MODEL` | `qwen3:8b` | Any model you have pulled (`ollama list`). |
| `GVS_LOCAL_TIMEOUT` | `20` | Seconds before giving up on the model (1st load can be slow). |
| `GVS_LOCAL_THINK` | `false` | `false` for qwen3 / qwen3.5. `auto` for models without a thinking switch. |
| `GVS_LOCAL_NUM_CTX` | `8192` | Context size. 8K fits a 9B model in 8 GB. |
| `GVS_LOCAL_KEEP_ALIVE` | `30m` | How long the model stays loaded after the last message. `-1` = always. |

To try a setting for one terminal only: `$env:GVS_LOCAL_MODEL = "qwen3.5:9b"`.

## 4. Check it

```powershell
uv run gvs-local-check
ollama ps          # PROCESSOR should say 100% GPU
```

`gvs-local-check` confirms Ollama is running and has the model, sends one test
message twice (cold, then warm) and checks the answer. It ends with `OK` or a
`FAIL` line that says what to fix.

## 5. Score it

```powershell
# What visitors see: rules first, model when unsure
uv run gvs-eval --brains rules,local --set evals/holdout.jsonl --runs 3
# The model on its own
uv run gvs-eval --brains local --solo --set evals/holdout.jsonl --runs 3
# Same again with the other model, to compare
$env:GVS_LOCAL_MODEL = "qwen3.5:9b"; uv run gvs-eval --brains rules,local --set evals/holdout.jsonl --runs 3
```

The no-AI baseline on the held-out set is 68% exact. Keep the model with the
better `exact` score; if they tie, keep the faster one. Don't edit
`holdout.jsonl` to raise a score: it is only for measuring.

## 6. Chat with it

```powershell
uv run gvs-chat -v
```

`-v` shows which brain answered each message and how long the model took.

## Testing on Colab instead (T4, 16 GB)

`notebooks/colab-t4-local-model.ipynb` does all of the above on a Colab GPU:
installs Ollama and uv, pulls the models, clones the repo, runs
`gvs-local-check` and scores every model (rules only, rules -> model, model
alone). Open it in Colab, choose a T4 GPU, then Run all:
https://colab.research.google.com/github/AakashBhat1/gvs-chatbot/blob/main/notebooks/colab-t4-local-model.ipynb

## Troubleshooting

| Symptom | Fix |
|---|---|
| `cannot reach Ollama` | Start Ollama from the Start menu; check `ollama list` works. |
| `is not pulled` | `ollama pull <model>` with the exact name shown. |
| Warm run is slow, `ollama ps` shows CPU | Update the NVIDIA driver, close other GPU apps, or set `GVS_LOCAL_NUM_CTX=4096`. |
| `does not match the form schema` often | Update Ollama (`winget upgrade Ollama.Ollama`); old versions ignore the JSON format with qwen3.5. |
| `model output was cut off` | Rare; the message described more than 8 people. The bot falls back to the rules. |

## Security notes

- The model only fills a form of ids and numbers. Its output is checked twice
  and nothing it writes is shown to visitors, so a prompt injection can at most
  fill the form wrongly, and the summary shows the form before any checklist.
- Messages are never logged. They stay on this machine when the model is local.
- The bot never follows redirects or uses a proxy for the model server.
