"""
validate.py — Pre-submission validation for OpenEnv compliance.
Run this before submitting: python validate.py [BASE_URL]

Checks:
  1. Health endpoint returns 200
  2. /tasks lists 3+ tasks
  3. reset() works for each task
  4. step() works and returns reward in [0, 1]
  5. state() works
  6. openenv.yaml exists and is valid
  7. inference.py exists
  8. Dockerfile exists
"""

import json
import os
import sys
import time

import requests
import yaml

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:7860"
PASS = "✅"
FAIL = "❌"
results = []


def check(name, fn):
    try:
        fn()
        print(f"  {PASS} {name}")
        results.append((name, True))
    except Exception as e:
        print(f"  {FAIL} {name}: {e}")
        results.append((name, False))


print(f"\n🔍 Validating OpenEnv at: {BASE_URL}\n")

# 1. Health
def test_health():
    r = requests.get(f"{BASE_URL}/", timeout=10)
    assert r.status_code == 200, f"status {r.status_code}"

check("GET / returns 200", test_health)

# 2. Tasks endpoint
def test_tasks():
    r = requests.get(f"{BASE_URL}/tasks", timeout=10)
    assert r.status_code == 200
    data = r.json()
    assert "tasks" in data
    assert len(data["tasks"]) >= 3, f"Only {len(data['tasks'])} tasks, need 3+"

check("GET /tasks returns 3+ tasks", test_tasks)

# 3-5. Per-task reset/step/state
for task_id in ["task1", "task2", "task3"]:
    def test_task(tid=task_id):
        # reset
        r = requests.post(f"{BASE_URL}/reset", params={"task_id": tid}, timeout=15)
        assert r.status_code == 200, f"reset status {r.status_code}: {r.text}"
        data = r.json()
        assert "observation" in data
        assert "info" in data
        sid = data["info"]["session_id"]
        obs = data["observation"]
        assert obs["task_id"] == tid

        # step
        if tid == "task1":
            action = {"urgency": "medium"}
        elif tid == "task2":
            action = {"urgency": "medium", "department": "support"}
        else:
            action = {
                "urgency": "medium",
                "department": "support",
                "response_draft": "Thank you for contacting us. Our team will review your ticket and respond within 24 hours. We apologise for any inconvenience this may have caused and will resolve your issue promptly.",
            }

        r2 = requests.post(f"{BASE_URL}/step", json=action, params={"session_id": sid}, timeout=15)
        assert r2.status_code == 200, f"step status {r2.status_code}: {r2.text}"
        step_data = r2.json()
        reward = step_data["reward"]
        assert isinstance(reward, (int, float)), "reward must be numeric"
        assert 0.0 <= reward <= 1.0, f"reward {reward} out of [0,1]"

        # state
        r3 = requests.get(f"{BASE_URL}/state", params={"session_id": sid}, timeout=10)
        assert r3.status_code == 200
        state = r3.json()
        assert state["task_id"] == tid

    check(f"Task {task_id}: reset/step/state cycle", test_task)

# 6. openenv.yaml
def test_yaml():
    assert os.path.exists("openenv.yaml"), "openenv.yaml not found"
    with open("openenv.yaml") as f:
        spec = yaml.safe_load(f)
    assert "name" in spec
    assert "tasks" in spec
    assert len(spec["tasks"]) >= 3

check("openenv.yaml valid with 3+ tasks", test_yaml)

# 7. inference.py
def test_inference():
    assert os.path.exists("inference.py"), "inference.py not found in root"

check("inference.py exists in root", test_inference)

# 8. Dockerfile
def test_dockerfile():
    assert os.path.exists("Dockerfile"), "Dockerfile not found"

check("Dockerfile exists", test_dockerfile)

# Summary
print()
passed = sum(1 for _, ok in results if ok)
total = len(results)
print(f"{'='*40}")
print(f"Result: {passed}/{total} checks passed")
if passed == total:
    print("🎉 All checks passed! Ready to submit.")
else:
    print("⚠️  Fix the failing checks before submitting.")
    sys.exit(1)
