---
title: Support Ticket Routing OpenEnv
emoji: 🎫
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
license: apache-2.0
---

# 🎫 Support Ticket Routing — OpenEnv Environment

A **real-world OpenEnv-compliant RL environment** where an AI agent learns to
triage customer support tickets: classify urgency, route to the right team, and
draft professional first-response messages.

---

## 🌍 Environment Description

Customer support teams receive hundreds of tickets per day. Routing them
correctly — balancing urgency, department expertise, and customer tier — is a
high-value real-world task that requires language understanding and judgment.

This environment presents the agent with a realistic support ticket and asks
it to perform increasingly difficult triage tasks.

---

## 📋 Tasks

| ID | Name | Difficulty | What the agent must do |
|----|------|-----------|----------------------|
| `task1` | Urgency Classification | Easy | Classify ticket urgency: `low / medium / high / critical` |
| `task2` | Ticket Routing | Medium | Classify urgency **+** route to correct department |
| `task3` | Full Triage | Hard | Urgency + department + draft first-response message |

All rewards are in `[0.0, 1.0]`.

---

## 📐 Observation Space

```json
{
  "ticket_id": "TKT-3F9A12",
  "subject": "Cannot login to my account",
  "body": "I've been trying to log in for 2 hours...",
  "customer_tier": "pro",
  "previous_contacts": 0,
  "task_id": "task1",
  "task_description": "Classify urgency as low/medium/high/critical"
}
```

## 🎮 Action Space

**Task 1 (easy):**
```json
{"urgency": "high"}
```

**Task 2 (medium):**
```json
{"urgency": "high", "department": "authentication"}
```

**Task 3 (hard):**
```json
{
  "urgency": "high",
  "department": "authentication",
  "response_draft": "We're sorry to hear you're having trouble..."
}
```

---

## 🏆 Reward Function

| Task | Formula |
|------|---------|
| task1 | `urgency_score` (1.0 if exact; -0.35 per level off) |
| task2 | `0.5 × urgency_score + 0.5 × department_score` |
| task3 | `0.3 × urgency + 0.3 × department + 0.4 × response_quality` |

Response quality is measured by keyword coverage and appropriate length.

---

## 🚀 API

Base URL: `https://<your-space>.hf.space`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `GET /` | GET | Health check |
| `GET /tasks` | GET | List all tasks |
| `POST /reset?task_id=task1` | POST | Start a new episode |
| `POST /step?session_id=<id>` | POST | Submit an action |
| `GET /state?session_id=<id>` | GET | Get current state |

---

## ⚡ Quick Start

```python
import requests

BASE = "https://<your-space>.hf.space"

# 1. Reset
r = requests.post(f"{BASE}/reset", params={"task_id": "task2"})
data = r.json()
session_id = data["info"]["session_id"]
obs = data["observation"]
print(obs["subject"], obs["body"])

# 2. Step
action = {"urgency": "high", "department": "authentication"}
result = requests.post(f"{BASE}/step", json=action,
                       params={"session_id": session_id}).json()
print("Reward:", result["reward"])
print("Breakdown:", result["info"]["breakdown"])
```

---

## 🛠️ Local Setup

```bash
git clone https://huggingface.co/spaces/<your-username>/support-ticket-routing-env
cd support-ticket-routing-env
pip install -r requirements.txt
uvicorn app:app --port 7860
```

Or with Docker:
```bash
docker build -t support-env .
docker run -p 7860:7860 support-env
```

---

## 🤖 Running the Baseline

```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="your-api-key"
export ENV_BASE_URL="http://localhost:7860"

python inference.py
```

Expected output format:
```json
{"event": "START", "run_id": "...", "model": "...", ...}
{"event": "STEP",  "task_id": "task1", "episode": 0, "reward": 0.85, ...}
...
{"event": "END",   "results": {"task1": {"avg_reward": 0.83}, ...}}
```

---

## 📁 File Structure

```
.
├── app.py            # FastAPI OpenEnv server
├── openenv.yaml      # OpenEnv spec
├── inference.py      # Baseline inference script
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 📊 Baseline Scores (gpt-4o-mini, 3 episodes each)

| Task | Avg Reward |
|------|-----------|
| task1 | ~0.82 |
| task2 | ~0.73 |
| task3 | ~0.61 |

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `API_BASE_URL` | LLM API endpoint (OpenAI-compatible) |
| `MODEL_NAME` | Model identifier for inference |
| `HF_TOKEN` | Hugging Face / API key |
| `ENV_BASE_URL` | URL of the running environment |
