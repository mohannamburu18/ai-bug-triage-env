# 🚀 Deployment & Submission Guide

## Pre-submission checklist

- [x] HF Space deploys (Dockerfile + port 7860)
- [x] `/health` returns 200 + `{"status": "ok"}`
- [x] `/reset` returns valid TriageObservation
- [x] `/step` returns observation, reward (0–1), done, info
- [x] `/state` returns EpisodeState
- [x] `/tasks` returns 3 tasks with action schema
- [x] `/grader` scores a list of actions (0.0–1.0)
- [x] `/baseline` runs and returns reproducible scores
- [x] `openenv.yaml` present with correct metadata
- [x] 3+ tasks with difficulty range easy → medium → hard
- [x] Graders are deterministic (same seed → same score)
- [x] Reward function provides partial-progress signal (not binary)
- [x] 17/17 unit tests pass
- [x] README with action/observation space, setup, baseline scores

---

## Deploying to Hugging Face Spaces

### Step 1 — Create the Space
1. Go to https://huggingface.co/new-space
2. Choose **Docker** as the SDK
3. Set visibility to **Public**
4. Name it: `bug-triage-env`

### Step 2 — Push the code
```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/bug-triage-env
cd bug-triage-env

# Copy all project files into the cloned repo
cp -r /path/to/bug-triage-env/* .

# The HF Space README must be the root README.md with the YAML header
# Prepend SPACES_README.md content to README.md
cat SPACES_README.md README.md > README_combined.md
mv README_combined.md README.md

git add .
git commit -m "Initial deployment: Bug Triage OpenEnv v1.0.0"
git push
```

### Step 3 — Verify
```bash
SPACE_URL="https://YOUR_USERNAME-bug-triage-env.hf.space"

# Health
curl $SPACE_URL/health

# Reset
curl -X POST $SPACE_URL/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy_triage", "seed": 42}'

# Tasks
curl $SPACE_URL/tasks

# Baseline
curl $SPACE_URL/baseline
```

### Step 4 — Submit
Submit `https://YOUR_USERNAME-bug-triage-env.hf.space` as your Space URL.

---

## Running locally (no Docker)

```bash
pip install -r requirements.txt
cd src
uvicorn server:app --host 0.0.0.0 --port 7860
# Visit http://localhost:7860/web for the interactive demo
```

## Running with Docker

```bash
docker build -t bug-triage-env .
docker run -p 7860:7860 bug-triage-env
```

## Running tests

```bash
cd src
python tests.py           # unit tests (fast, no server)
python tests.py --http    # includes HTTP server integration tests
```

## Running baseline

```bash
cd src
python baseline.py --mode heuristic   # no API key needed

# LLM baseline
export OPENAI_API_KEY=sk-...
python baseline.py --mode llm --model gpt-4o-mini
```
