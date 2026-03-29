# 🐛 Bug Triage OpenEnv v4

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi)
![Gradio](https://img.shields.io/badge/Gradio-4.0+-orange?logo=gradio)
![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Spaces-yellow)
![OpenEnv](https://img.shields.io/badge/OpenEnv-Compliant-purple)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

**A Reinforcement Learning Environment for AI-Powered Software Bug Triage**

[🚀 Live Demo](https://mohannamburu-ai-bug-triage-env.hf.space) • [📖 API Docs](https://mohannamburu-ai-bug-triage-env.hf.space/docs) • [🎯 Hackathon](https://scaler.com/school-of-technology/meta-pytorch-hackathon)

</div>

---

## 🎯 What is This?

**Bug Triage OpenEnv** is a reinforcement learning environment that simulates the real-world task of **software bug triage** — the process engineers perform daily to prioritize, categorize, and route bugs to the right teams.

This environment allows AI agents to learn and be evaluated on:
- 🎯 **Prioritizing bugs** (critical → low)
- 🏷️ **Assigning labels** (bug, security, performance, etc.)
- 👥 **Routing to teams** (backend, frontend, security, etc.)
- ⏱️ **Estimating fix time** (1-90 days)
- 🚨 **Escalation decisions** (on-call alerts)

---

## 🧠 Why Bug Triage?

| Real-World Problem | Our Solution |
|-------------------|--------------|
| Engineers spend **hours daily** triaging bugs | Train AI to automate triage decisions |
| Inconsistent prioritization across teams | Standardized scoring with clear criteria |
| Critical bugs get buried in backlogs | AI learns to identify and escalate urgently |
| New engineers struggle with triage | Environment teaches optimal triage strategies |

**This is NOT a toy problem** — it's a genuine workflow that every software company deals with.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    BUG TRIAGE OPENENV v4                        │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │   GRADIO    │    │   FASTAPI   │    │     INFERENCE.PY    │  │
│  │   Web UI    │───▶│   Server    │◀───│     LLM Agent       │  │
│  │  (ui.py)    │    │ (server.py) │    │   (OpenAI Client)   │  │
│  └─────────────┘    └──────┬──────┘    └─────────────────────┘  │
│                            │                                    │
│                            ▼                                    │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                  ENVIRONMENT ENGINE                       │  │
│  │                  (environment.py)                         │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │  │
│  │  │ 30 Bugs  │  │  State   │  │  Grader  │  │  Reward  │   │  │
│  │  │ Database │  │  Machine │  │  Engine  │  │  Scorer  │   │  │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🎮 How It Works

### The 6-Step Triage Pipeline

Each bug goes through a sequential decision pipeline:

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ PRIORITY │───▶│  LABELS  │───▶│   TEAM   │───▶│  NEEDS   │───▶│   FIX    │───▶│ ESCALATE │
│          │    │          │    │          │    │   INFO?  │    │   DAYS   │    │ TO ONCALL│
│ critical │    │ bug      │    │ backend  │    │          │    │          │    │          │
│ high     │    │ security │    │ frontend │    │ true/    │    │  1-90    │    │ true/    │
│ medium   │    │ ux       │    │ security │    │ false    │    │  days    │    │ false    │
│ low      │    │ perf     │    │ mobile   │    │          │    │          │    │          │
└──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
```

### Reward System

The environment provides **meaningful partial rewards** (not just binary pass/fail):

| Decision | Reward Calculation |
|----------|-------------------|
| Priority | `1.0 - (0.35 × level_distance)` |
| Labels | F1 Score (precision × recall) |
| Team | 1.0 correct, 0.9 if overloaded, 0.0 wrong |
| Needs Info | Binary (1.0 or 0.0) |
| Fix Days | Gaussian decay from expected |
| Escalate | Binary (1.0 or 0.0) |

---

## 📋 Tasks & Difficulty Levels

| Task | Bugs | Step Budget | Pass Score | Description |
|------|------|-------------|------------|-------------|
| `easy_triage` | 4 | 30 | 0.75 | Clear-cut bugs, obvious priorities |
| `medium_triage` | 6 | 50 | 0.65 | Ambiguous cases, conflicting signals |
| `hard_triage` | 10 | 80 | 0.55 | Multi-queue selection, team load balancing |

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/ai-bug-triage-env.git
cd ai-bug-triage-env
pip install -r requirements.txt
```

### 2. Run Locally

```bash
# Start the server
python server.py

# Visit http://localhost:7860/web/
```

### 3. Run Inference

```bash
# Set your HuggingFace token
export HF_TOKEN=hf_your_token_here

# Run LLM baseline on all tasks
python inference.py

# Run specific task
python inference.py --task easy_triage --seed 42
```

---

## 🔌 API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/reset` | Reset environment with task_id and seed |
| `POST` | `/step` | Execute action, get (obs, reward, done, info) |
| `GET` | `/state` | Get current state |
| `GET` | `/tasks` | List available tasks |
| `POST` | `/grader` | Evaluate action sequence |
| `GET` | `/baseline` | Get heuristic baseline scores |

### Example: Reset Environment

```python
import requests

response = requests.post("https://mohannamburu-ai-bug-triage-env.hf.space/reset", 
    json={"task_id": "easy_triage", "seed": 42}
)
observation = response.json()
print(observation["current_bug"]["title"])
```

### Example: Take a Step

```python
response = requests.post("https://mohannamburu-ai-bug-triage-env.hf.space/step",
    json={
        "action": {
            "priority": "high",
            "confidence": 0.9
        }
    }
)
result = response.json()
print(f"Reward: {result['reward']}, Done: {result['done']}")
```

---

## 📊 Observation Space

```python
TriageObservation {
    current_bug: BugReport {
        title: str           # Bug title (may be misleading!)
        body: str            # Full description (read carefully)
        component: str       # UI, API, Database, etc.
        reporter: str        # Who reported it
        reproducible: bool   # Can it be reproduced?
        affected_users: int  # Number of affected users (if known)
        stacktrace: str?     # Error stacktrace (if available)
        severity_hint: str?  # Reporter's severity guess
    }
    current_sub_step: str    # priority/labels/team/etc.
    bugs_remaining: int      # Bugs left to triage
    steps_remaining: int     # Step budget remaining
    team_loads: Dict[str, int]  # Current team workloads
    queue_summary: List[BugSummary]  # For hard_triage multi-queue
    done: bool               # Episode finished?
}
```

---

## 🎬 Action Space

```python
TriageAction {
    # For select_bug step (hard_triage only)
    select_bug_index: int?
    
    # For priority step
    priority: "critical" | "high" | "medium" | "low"
    confidence: float  # 0.0 - 1.0
    
    # For labels step
    labels: List[str]  # ["bug", "security", "ux", ...]
    
    # For team step
    assigned_team: "backend" | "frontend" | "security" | "mobile" | "infra" | "data" | "qa"
    
    # For needs_more_info step
    needs_more_info: bool
    
    # For estimated_fix_days step
    estimated_fix_days: int  # 1-90
    
    # For escalate_to_oncall step
    escalate_to_oncall: bool
    
    # Special actions
    revise: bool?  # Request revision (budget=2)
    defer: bool?   # Skip bug (-0.08 penalty)
}
```

---

## 📈 Baseline Results

### Heuristic Baseline

| Task | Score | Threshold | Status |
|------|-------|-----------|--------|
| easy_triage | **0.799** | 0.75 | ✅ PASS |
| medium_triage | **0.753** | 0.65 | ✅ PASS |
| hard_triage | **0.772** | 0.55 | ✅ PASS |

### LLM Baseline (Llama-3.1-8B-Instruct)

| Task | Score | Threshold | Status |
|------|-------|-----------|--------|
| easy_triage | **0.846** | 0.75 | ✅ PASS |

---

## 🛠️ Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `API_BASE_URL` | LLM API endpoint | `https://router.huggingface.co/v1` |
| `MODEL_NAME` | Model for inference | `meta-llama/Llama-3.1-8B-Instruct` |
| `HF_TOKEN` | HuggingFace API token | (required) |

---

## 📁 Project Structure

```
ai-bug-triage-env/
├── 📄 server.py          # FastAPI server with all endpoints
├── 🎨 ui.py              # Gradio web interface
├── 🧠 environment.py     # Core RL environment & grading
├── 🤖 inference.py       # LLM baseline agent
├── 📊 baseline.py        # Heuristic baseline
├── 🧪 tests.py           # Unit tests
├── 📦 bug_triage_env.py  # Python client library
├── 📋 openenv.yaml       # OpenEnv specification
├── 🐳 Dockerfile         # Container configuration
├── 📜 requirements.txt   # Python dependencies
└── 📖 README.md          # This file
```

---

## 🎨 Screenshots

<!-- Add your screenshots here -->
*Coming soon...*

---

## ✅ OpenEnv Compliance

This environment fully implements the [OpenEnv specification](https://github.com/openenv-ai/openenv):

- ✅ **Typed Pydantic models** for Observation, Action, Reward
- ✅ **step(action)** → returns (observation, reward, done, info)
- ✅ **reset(task_id, seed)** → returns initial observation
- ✅ **state()** → returns current state
- ✅ **openenv.yaml** with metadata
- ✅ **3+ tasks** with programmatic graders
- ✅ **Meaningful rewards** (partial progress, not binary)
- ✅ **Baseline inference script** with reproducible scores

---

## 🏆 Hackathon

Built for the **AgentBeats × OpenEnv Challenge** (Meta PyTorch Hackathon)

- **Team Lead:** Likhitha Sankula
- **Live Demo:** [HuggingFace Spaces](https://mohannamburu-ai-bug-triage-env.hf.space)
- **Submission Deadline:** April 8, 2026

---

## 🤝 Contributing

Contributions welcome! Please feel free to submit a Pull Request.

---

## 📄 License

MIT License - feel free to use this for your own projects!

---

<div align="center">

**⭐ Star this repo if you find it useful!**

Made with ❤️ for the OpenEnv Hackathon

</div>
