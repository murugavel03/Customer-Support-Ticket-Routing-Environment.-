"""
inference.py — Baseline inference script for Support Ticket Routing OpenEnv
Uses OpenAI-compatible client with structured stdout logs.

Required env vars:
  API_BASE_URL  — LLM API endpoint
  MODEL_NAME    — model identifier
  HF_TOKEN      — Hugging Face / API key

Environment URL is read from ENV_BASE_URL (default: http://localhost:7860)
"""

import json
import os
import sys
import time
import uuid

import requests
from openai import OpenAI

# ── Config ─────────────────────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")
ENV_URL      = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "sk-placeholder")

TASKS = ["task1", "task2", "task3"]
EPISODES_PER_TASK = 3   # keep total runtime < 5 min

# ── Helpers ────────────────────────────────────────────────────────────────────

def env_reset(task_id: str):
    r = requests.post(f"{ENV_URL}/reset", params={"task_id": task_id}, timeout=30)
    r.raise_for_status()
    return r.json()

def env_step(action: dict, session_id: str):
    r = requests.post(
        f"{ENV_URL}/step",
        json=action,
        params={"session_id": session_id},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()

def env_state(session_id: str):
    r = requests.get(f"{ENV_URL}/state", params={"session_id": session_id}, timeout=30)
    r.raise_for_status()
    return r.json()


def build_prompt(obs: dict, task_id: str) -> str:
    base = f"""You are an expert customer support triage agent.

Ticket ID: {obs['ticket_id']}
Subject: {obs['subject']}
Body: {obs['body']}
Customer Tier: {obs['customer_tier']}
Previous Contacts: {obs['previous_contacts']}

Task: {obs['task_description']}
"""
    if task_id == "task1":
        base += """
Respond ONLY with a valid JSON object (no markdown):
{"urgency": "<low|medium|high|critical>"}
"""
    elif task_id == "task2":
        base += """
Respond ONLY with a valid JSON object (no markdown):
{"urgency": "<low|medium|high|critical>", "department": "<authentication|billing|engineering|support|product|security>"}
"""
    else:
        base += """
Respond ONLY with a valid JSON object (no markdown):
{
  "urgency": "<low|medium|high|critical>",
  "department": "<authentication|billing|engineering|support|product|security>",
  "response_draft": "<professional first-response message, 30-200 words>"
}
"""
    return base


def call_llm(prompt: str) -> dict:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0.2,
    )
    text = response.choices[0].message.content.strip()
    # Strip markdown fences if present
    text = text.replace("```json", "").replace("```", "").strip()
    return json.loads(text)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    run_id = uuid.uuid4().hex[:8]

    print(json.dumps({
        "event": "START",
        "run_id": run_id,
        "model": MODEL_NAME,
        "env_url": ENV_URL,
        "tasks": TASKS,
        "episodes_per_task": EPISODES_PER_TASK,
    }))
    sys.stdout.flush()

    results = {}

    for task_id in TASKS:
        task_rewards = []

        for ep in range(EPISODES_PER_TASK):
            # Reset environment
            reset_data = env_reset(task_id)
            session_id = reset_data["info"]["session_id"]
            obs = reset_data["observation"]

            # Build prompt and call LLM
            prompt = build_prompt(obs, task_id)
            t0 = time.time()
            try:
                action = call_llm(prompt)
            except Exception as e:
                # Fallback to safe defaults
                action = {"urgency": "medium"}
                if task_id in ("task2", "task3"):
                    action["department"] = "support"
                if task_id == "task3":
                    action["response_draft"] = (
                        "Thank you for reaching out. We have received your ticket and our "
                        "team will investigate and get back to you as soon as possible. "
                        "We apologise for any inconvenience caused."
                    )
            latency = round(time.time() - t0, 3)

            # Step environment
            step_result = env_step(action, session_id)
            reward = step_result["reward"]
            task_rewards.append(reward)

            print(json.dumps({
                "event": "STEP",
                "run_id": run_id,
                "task_id": task_id,
                "episode": ep,
                "session_id": session_id,
                "ticket_id": obs["ticket_id"],
                "action": action,
                "reward": reward,
                "breakdown": step_result["info"].get("breakdown", {}),
                "true_urgency": step_result["info"].get("true_urgency"),
                "true_department": step_result["info"].get("true_department"),
                "llm_latency_s": latency,
            }))
            sys.stdout.flush()

        avg_reward = round(sum(task_rewards) / len(task_rewards), 4)
        results[task_id] = {
            "episodes": EPISODES_PER_TASK,
            "rewards": task_rewards,
            "avg_reward": avg_reward,
        }

    print(json.dumps({
        "event": "END",
        "run_id": run_id,
        "results": results,
        "overall_avg": round(
            sum(v["avg_reward"] for v in results.values()) / len(results), 4
        ),
    }))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
