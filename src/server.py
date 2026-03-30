"""FastAPI server for Bug Triage OpenEnv v4."""

from contextlib import asynccontextmanager
import gradio as gr
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from environment import (
    BugTriageEnvironment, TriageAction, TASKS, BUG_BANK,
    _grade_priority, _grade_labels, _grade_team, _grade_info,
    _grade_days, _grade_escalation
)

# ------------------- INIT -------------------

env = BugTriageEnvironment()

@asynccontextmanager
async def lifespan(app: FastAPI):
    env.reset(task_id="medium_triage", seed=42)
    yield

app = FastAPI(
    title="Bug Triage OpenEnv v4",
    version="4.0.0",
    lifespan=lifespan
)

# ------------------- MODELS -------------------

class ResetRequest(BaseModel):
    task_id: str = "medium_triage"
    seed: int | None = None

class StepRequest(BaseModel):
    action: TriageAction

class GraderRequest(BaseModel):
    task_id: str
    actions: list[TriageAction]

# ------------------- ROUTES -------------------

@app.get("/health")
async def health():
    return {"status": "alive"}

@app.post("/reset")
async def reset(req: ResetRequest = ResetRequest()):
    obs = env.reset(task_id=req.task_id, seed=req.seed)
    return obs.model_dump()

@app.post("/step")
async def step(req: StepRequest):
    try:
        obs, reward, done, info = env.step(req.action)
        return {
            "observation": obs.model_dump(),
            "reward": reward,
            "done": done,
            "info": info
        }
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/state")
async def state():
    return env.state().model_dump()

@app.get("/tasks")
async def tasks():
    return {"tasks": env.get_tasks()}

@app.post("/grader")
async def grader(req: GraderRequest):
    import random

    task = TASKS.get(req.task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    eligible = [b for b in BUG_BANK if task["bug_filter"](b)]
    bugs = random.Random(42).sample(eligible, min(task["max_bugs"], len(eligible)))

    results = []
    total = 0.0
    idx = 0

    for bug in bugs:
        sol = bug["_solution"]
        bug_scores = {}

        for sub in ["priority","labels","team","needs_more_info","estimated_fix_days","escalate_to_oncall"]:
            if idx >= len(req.actions):
                bug_scores[sub] = 0.0
                continue

            a = req.actions[idx]
            idx += 1

            if sub == "priority":
                s,_ = _grade_priority(a.priority or "low", sol, "normal")
            elif sub == "labels":
                s,_,_,_ = _grade_labels(a.labels or [], sol)
            elif sub == "team":
                s,_ = _grade_team(a.assigned_team or "backend", sol, {})
            elif sub == "needs_more_info":
                s,_,_ = _grade_info(
                    a.needs_more_info or False,
                    sol,
                    bug["reproducible"],
                    bug["partially_observable"]
                )
            elif sub == "estimated_fix_days":
                s,_ = _grade_days(a.estimated_fix_days or 14, sol)
            elif sub == "escalate_to_oncall":
                s,_,_ = _grade_escalation(
                    a.escalate_to_oncall or False,
                    sol,
                    "normal"
                )

            bug_scores[sub] = s

        bug_score = round(sum(bug_scores.values()) / 6, 3)
        total += bug_score

        results.append({
            "bug": bug["title"],
            "score": bug_score,
            "sub_scores": bug_scores
        })

    n = len(bugs)
    ep_score = round(total / n, 3) if n > 0 else 0.0

    return {
        "task_id": req.task_id,
        "episode_score": ep_score,
        "passed": ep_score >= task["passing_score"],
        "passing_threshold": task["passing_score"],
        "bugs": results
    }

@app.get("/baseline")
async def baseline():
    from baseline import run_heuristic_baseline
    return {"baseline_results": run_heuristic_baseline()}

# ------------------- UI -------------------

from ui import build_ui

gradio_app = build_ui()

# ✅ FINAL FIX: mount UI at ROOT
app = gr.mount_gradio_app(app, gradio_app, path="/")