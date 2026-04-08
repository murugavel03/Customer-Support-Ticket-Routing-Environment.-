"""
Customer Support Ticket Routing Environment
OpenEnv-compliant FastAPI server

Tasks:
  1. (Easy)   Classify a ticket's urgency (low/medium/high/critical)
  2. (Medium) Route a ticket to the correct department + urgency
  3. (Hard)   Full triage: department + urgency + draft first-response message
"""

import json
import random
import re
import uuid
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ── Pydantic models ────────────────────────────────────────────────────────────

class TicketObservation(BaseModel):
    ticket_id: str
    subject: str
    body: str
    customer_tier: str          # "free" | "pro" | "enterprise"
    previous_contacts: int
    task_id: str
    task_description: str

class RoutingAction(BaseModel):
    urgency: str                # "low" | "medium" | "high" | "critical"
    department: Optional[str] = None   # required for tasks 2 & 3
    response_draft: Optional[str] = None  # required for task 3

class StepResult(BaseModel):
    observation: TicketObservation
    reward: float
    done: bool
    info: Dict[str, Any]

class ResetResult(BaseModel):
    observation: TicketObservation
    info: Dict[str, Any]

class StateResult(BaseModel):
    task_id: str
    episode_id: str
    step: int
    done: bool
    cumulative_reward: float

# ── Ticket data ────────────────────────────────────────────────────────────────

TICKETS = [
    {
        "subject": "Cannot login to my account",
        "body": "I've been trying to log in for 2 hours. Password reset email never arrives. I have a presentation tomorrow and need access urgently!",
        "customer_tier": "pro",
        "previous_contacts": 0,
        "true_urgency": "high",
        "true_department": "authentication",
        "ideal_response_keywords": ["sorry", "reset", "team", "help", "urgent"],
    },
    {
        "subject": "Billing charge I don't recognise",
        "body": "There is a $299 charge on my card from last Tuesday that I did not authorise. Please refund immediately.",
        "customer_tier": "enterprise",
        "previous_contacts": 1,
        "true_urgency": "critical",
        "true_department": "billing",
        "ideal_response_keywords": ["refund", "investigate", "sorry", "24 hours"],
    },
    {
        "subject": "How do I export my data?",
        "body": "Hi, I'd like to know how to export all my project data as CSV. No rush, just wondering.",
        "customer_tier": "free",
        "previous_contacts": 0,
        "true_urgency": "low",
        "true_department": "support",
        "ideal_response_keywords": ["export", "settings", "csv", "instructions"],
    },
    {
        "subject": "API rate limit errors in production",
        "body": "We're getting 429 errors every few minutes. Our production service is degraded. Thousands of users affected. Fix ASAP.",
        "customer_tier": "enterprise",
        "previous_contacts": 3,
        "true_urgency": "critical",
        "true_department": "engineering",
        "ideal_response_keywords": ["escalate", "engineering", "monitor", "sorry", "priority"],
    },
    {
        "subject": "Feature request: dark mode",
        "body": "Would love to see dark mode added to the dashboard. My eyes hurt after long sessions.",
        "customer_tier": "free",
        "previous_contacts": 0,
        "true_urgency": "low",
        "true_department": "product",
        "ideal_response_keywords": ["feature", "roadmap", "noted", "team"],
    },
    {
        "subject": "Wrong invoice amount",
        "body": "Invoice #INV-2024-882 shows $500 but we agreed on $350. Need corrected invoice for accounting.",
        "customer_tier": "pro",
        "previous_contacts": 2,
        "true_urgency": "medium",
        "true_department": "billing",
        "ideal_response_keywords": ["invoice", "correct", "accounting", "resend"],
    },
    {
        "subject": "Security vulnerability found in your API",
        "body": "I'm a security researcher and found an SQL injection vulnerability in your /api/v1/search endpoint. Responsible disclosure – please contact me.",
        "customer_tier": "free",
        "previous_contacts": 0,
        "true_urgency": "critical",
        "true_department": "security",
        "ideal_response_keywords": ["security", "team", "disclosure", "thank", "investigate"],
    },
    {
        "subject": "Slow dashboard loading",
        "body": "The dashboard takes 10+ seconds to load today. Not urgent but a bit annoying.",
        "customer_tier": "pro",
        "previous_contacts": 0,
        "true_urgency": "medium",
        "true_department": "engineering",
        "ideal_response_keywords": ["performance", "look into", "sorry", "team"],
    },
]

TASKS = {
    "task1": {
        "id": "task1",
        "description": "Classify the urgency of the support ticket as: low, medium, high, or critical.",
        "difficulty": "easy",
    },
    "task2": {
        "id": "task2",
        "description": "Route the ticket: provide both urgency (low/medium/high/critical) and department (authentication/billing/engineering/support/product/security).",
        "difficulty": "medium",
    },
    "task3": {
        "id": "task3",
        "description": "Full triage: provide urgency, department, and a draft first-response message to the customer.",
        "difficulty": "hard",
    },
}

URGENCY_ORDER = ["low", "medium", "high", "critical"]

# ── In-memory session state ────────────────────────────────────────────────────

sessions: Dict[str, Dict] = {}


def new_ticket(task_id: str) -> Dict:
    ticket = random.choice(TICKETS).copy()
    ticket["ticket_id"] = f"TKT-{uuid.uuid4().hex[:6].upper()}"
    ticket["task_id"] = task_id
    ticket["task_description"] = TASKS[task_id]["description"]
    return ticket


# ── Reward functions ───────────────────────────────────────────────────────────

def urgency_score(predicted: str, true: str) -> float:
    if predicted == true:
        return 1.0
    diff = abs(URGENCY_ORDER.index(predicted) - URGENCY_ORDER.index(true))
    return max(0.0, 1.0 - diff * 0.35)


def department_score(predicted: Optional[str], true: str) -> float:
    if predicted is None:
        return 0.0
    return 1.0 if predicted.lower() == true.lower() else 0.0


def response_score(draft: Optional[str], keywords: List[str]) -> float:
    if not draft:
        return 0.0
    draft_lower = draft.lower()
    hits = sum(1 for kw in keywords if kw.lower() in draft_lower)
    length_ok = 30 <= len(draft.split()) <= 200
    base = hits / len(keywords)
    return round(min(1.0, base + (0.1 if length_ok else 0.0)), 3)


def compute_reward(action: RoutingAction, ticket: Dict, task_id: str) -> tuple[float, Dict]:
    u_score = urgency_score(action.urgency, ticket["true_urgency"])
    breakdown = {"urgency_score": round(u_score, 3)}

    if task_id == "task1":
        reward = u_score
    elif task_id == "task2":
        d_score = department_score(action.department, ticket["true_department"])
        breakdown["department_score"] = d_score
        reward = 0.5 * u_score + 0.5 * d_score
    else:  # task3
        d_score = department_score(action.department, ticket["true_department"])
        r_score = response_score(action.response_draft, ticket["ideal_response_keywords"])
        breakdown["department_score"] = d_score
        breakdown["response_score"] = r_score
        reward = 0.3 * u_score + 0.3 * d_score + 0.4 * r_score

    return round(reward, 4), breakdown


# ── FastAPI app ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Support Ticket Routing OpenEnv",
    description="Real-world customer support triage environment for RL agents.",
    version="1.0.0",
)


@app.get("/", status_code=200)
def health():
    return {"status": "ok", "env": "support_ticket_routing", "version": "1.0.0"}


@app.get("/tasks")
def list_tasks():
    return {"tasks": list(TASKS.values())}


@app.post("/reset", response_model=ResetResult)
def reset(task_id: str = "task1"):
    if task_id not in TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task_id: {task_id}")

    session_id = str(uuid.uuid4())
    ticket = new_ticket(task_id)

    sessions[session_id] = {
        "session_id": session_id,
        "task_id": task_id,
        "ticket": ticket,
        "step": 0,
        "done": False,
        "cumulative_reward": 0.0,
        "episode_id": session_id,
    }

    obs = TicketObservation(
        ticket_id=ticket["ticket_id"],
        subject=ticket["subject"],
        body=ticket["body"],
        customer_tier=ticket["customer_tier"],
        previous_contacts=ticket["previous_contacts"],
        task_id=task_id,
        task_description=ticket["task_description"],
    )
    return ResetResult(observation=obs, info={"session_id": session_id, "task": TASKS[task_id]})


@app.post("/step", response_model=StepResult)
def step(action: RoutingAction, session_id: str = "default"):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found. Call /reset first.")

    sess = sessions[session_id]
    if sess["done"]:
        raise HTTPException(status_code=400, detail="Episode is done. Call /reset to start a new one.")

    # Validate action fields
    if action.urgency not in URGENCY_ORDER:
        raise HTTPException(status_code=422, detail=f"urgency must be one of {URGENCY_ORDER}")

    task_id = sess["task_id"]
    ticket = sess["ticket"]

    if task_id in ("task2", "task3") and not action.department:
        raise HTTPException(status_code=422, detail="department is required for task2 and task3")

    if task_id == "task3" and not action.response_draft:
        raise HTTPException(status_code=422, detail="response_draft is required for task3")

    reward, breakdown = compute_reward(action, ticket, task_id)
    sess["step"] += 1
    sess["cumulative_reward"] += reward
    sess["done"] = True  # one-step episode per ticket

    # Next observation is a fresh ticket (for multi-step rollouts)
    new_t = new_ticket(task_id)
    obs = TicketObservation(
        ticket_id=new_t["ticket_id"],
        subject=new_t["subject"],
        body=new_t["body"],
        customer_tier=new_t["customer_tier"],
        previous_contacts=new_t["previous_contacts"],
        task_id=task_id,
        task_description=new_t["task_description"],
    )

    return StepResult(
        observation=obs,
        reward=reward,
        done=sess["done"],
        info={
            "breakdown": breakdown,
            "true_urgency": ticket["true_urgency"],
            "true_department": ticket["true_department"],
            "step": sess["step"],
            "session_id": session_id,
        },
    )


@app.get("/state", response_model=StateResult)
def state(session_id: str = "default"):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    sess = sessions[session_id]
    return StateResult(
        task_id=sess["task_id"],
        episode_id=sess["episode_id"],
        step=sess["step"],
        done=sess["done"],
        cumulative_reward=round(sess["cumulative_reward"], 4),
    )
