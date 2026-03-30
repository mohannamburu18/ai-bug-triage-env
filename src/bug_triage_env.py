"""Bug Triage OpenEnv v4 — Python client library."""
from __future__ import annotations
import asyncio
from typing import Optional
import httpx
from pydantic import BaseModel
from environment import TriageAction, TriageObservation, EpisodeState

class StepResult(BaseModel):
    observation: TriageObservation
    reward: float
    done: bool
    info: dict

class _Sync:
    def __init__(self, c):
        self._c = c; self._loop = asyncio.new_event_loop()
    def __enter__(self):
        self._loop.run_until_complete(self._c.__aenter__()); return self
    def __exit__(self, *a):
        self._loop.run_until_complete(self._c.__aexit__(*a)); self._loop.close()
    def reset(self, task_id="medium_triage", seed=None):
        return self._loop.run_until_complete(self._c.reset(task_id, seed))
    def step(self, action): return self._loop.run_until_complete(self._c.step(action))
    def state(self): return self._loop.run_until_complete(self._c.state())

class BugTriageEnv:
    """Async HTTP client for Bug Triage OpenEnv v4. Use .sync() for sync access."""
    def __init__(self, base_url="http://localhost:7860", timeout=30.0):
        self.base_url = base_url.rstrip("/"); self._timeout = timeout; self._http = None
    async def __aenter__(self):
        self._http = httpx.AsyncClient(base_url=self.base_url, timeout=self._timeout); return self
    async def __aexit__(self, *a):
        if self._http: await self._http.aclose(); self._http = None
    def sync(self): return _Sync(self)
    async def reset(self, task_id="medium_triage", seed=None):
        payload = {"task_id": task_id}
        if seed is not None: payload["seed"] = seed
        r = await self._http.post("/reset", json=payload); r.raise_for_status()
        return TriageObservation(**r.json())
    async def step(self, action: TriageAction):
        r = await self._http.post("/step", json={"action": action.model_dump()}); r.raise_for_status()
        d = r.json()
        return StepResult(observation=TriageObservation(**d["observation"]),
                          reward=d["reward"], done=d["done"], info=d.get("info",{}))
    async def state(self):
        r = await self._http.get("/state"); r.raise_for_status()
        return EpisodeState(**r.json())
    async def tasks(self):
        r = await self._http.get("/tasks"); r.raise_for_status()
        return r.json()["tasks"]
