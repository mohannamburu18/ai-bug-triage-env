---
title: Bug Triage OpenEnv
emoji: 🐛
colorFrom: red
colorTo: orange
sdk: docker
app_port: 7860
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - bug-triage
  - nlp
  - real-world-agent
license: mit
---

# 🐛 Bug Triage OpenEnv v4

An **OpenEnv-compliant Reinforcement Learning environment** designed to train AI agents on **real-world software bug triage workflows**.

This environment simulates realistic decision-making scenarios where agents must analyze, prioritize, and route bugs efficiently under constraints.

---

## 🚀 Live Demo

👉 https://mohannamburu-ai-bug-triage-env.hf.space

---

## ⚡ Key Features

* 🧠 **Reinforcement Learning Environment**
* 🐞 **30 Realistic Bug Reports** (easy, medium, hard)
* 🔄 **Sequential Decision Making** (6-step triage pipeline)
* 🎯 **Multi-Queue Bug Selection**
* 👀 **Partial Observability**
* ⚠️ **Conflicting Signals in Data**
* 💸 **Action Costs & Penalties**
* 📊 **Confidence-Based Scoring**
* 🔁 **Revision Mechanism**
* 👥 **Team Load Balancing**
* ⏱️ **Step Budget Constraints**

---

## 🧪 API Endpoints

| Endpoint    | Method | Description         |
| ----------- | ------ | ------------------- |
| `/health`   | GET    | Health check        |
| `/reset`    | POST   | Start a new episode |
| `/step`     | POST   | Perform an action   |
| `/state`    | GET    | Get current state   |
| `/tasks`    | GET    | List all tasks      |
| `/baseline` | GET    | Run baseline agent  |
| `/grader`   | POST   | Evaluate actions    |
| `/web`      | GET    | Gradio UI           |

---

## 🎯 Tasks

| Task          | Bugs | Steps | Pass Score |
| ------------- | ---- | ----- | ---------- |
| easy_triage   | 4    | 30    | 0.75       |
| medium_triage | 6    | 50    | 0.65       |
| hard_triage   | 10   | 80    | 0.55       |

---

## 📈 Baseline Performance

| Task          | Score | Status |
| ------------- | ----- | ------ |
| easy_triage   | 0.799 | ✅ Pass |
| medium_triage | 0.753 | ✅ Pass |
| hard_triage   | 0.772 | ✅ Pass |

---

## 🛠️ Local Setup

```bash
pip install -r requirements.txt
cd src
uvicorn server:app --host 0.0.0.0 --port 7860
```

---

## 🐳 Docker Setup

```bash
docker build -t bug-triage-env .
docker run -p 7860:7860 bug-triage-env
```

---

## 🔗 Example API Usage

### Reset Environment

```bash
curl -X POST https://YOUR_SPACE.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy_triage", "seed": 42}'
```

### Take Step

```bash
curl -X POST https://YOUR_SPACE.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"priority": "critical", "confidence": 0.9}}'
```

### Get Baseline

```bash
curl https://YOUR_SPACE.hf.space/baseline
```

---

## 🧠 Use Cases

* Training RL agents for DevOps automation
* AI-assisted bug triage systems
* Research in sequential decision-making
* Real-world agent benchmarking

---

## 🏆 Hackathon Value

This project demonstrates:

* End-to-end **AI system design**
* Real-world **environment simulation**
* Practical **ML + backend integration**
* Deployable **Docker-based infrastructure**

---

## 👨‍💻 Author

**Mohan Namburu**

---

## 📄 License

MIT License
