"""Tests for Bug Triage OpenEnv v4. Run: python3 tests.py"""
import sys, json, subprocess, time, urllib.request, unittest
sys.path.insert(0, ".")
from environment import BugTriageEnvironment, TriageAction, TASKS, BUG_BANK

def _a(**kwargs):
    base = {k:None for k in ["priority","labels","assigned_team","needs_more_info","estimated_fix_days","escalate_to_oncall"]}
    base.update(kwargs)
    return TriageAction(**base)

def _run_bug(env, bug_overrides=None):
    """Run all 6 sub-steps for current bug with sensible defaults."""
    actions = {"priority":"medium","labels":["bug"],"assigned_team":"backend",
               "needs_more_info":False,"estimated_fix_days":7,"escalate_to_oncall":False}
    if bug_overrides: actions.update(bug_overrides)
    results = []
    for sub in ["priority","labels","team","needs_more_info","estimated_fix_days","escalate_to_oncall"]:
        if sub == "team": a = _a(assigned_team=actions["assigned_team"])
        elif sub == "priority": a = _a(priority=actions["priority"])
        elif sub == "labels": a = _a(labels=actions["labels"])
        elif sub == "needs_more_info": a = _a(needs_more_info=actions["needs_more_info"])
        elif sub == "estimated_fix_days": a = _a(estimated_fix_days=actions["estimated_fix_days"])
        elif sub == "escalate_to_oncall": a = _a(escalate_to_oncall=actions["escalate_to_oncall"])
        obs, r, done, info = env.step(a)
        results.append((r, info))
        if done: break
    return results, obs, done


class TestMultiQueue(unittest.TestCase):
    def test_hard_task_starts_with_select_bug(self):
        env = BugTriageEnvironment()
        obs = env.reset("hard_triage", seed=42)
        self.assertEqual(obs.current_sub_step, "select_bug")

    def test_select_bug_advances_to_priority(self):
        env = BugTriageEnvironment()
        env.reset("hard_triage", seed=42)
        obs, r, done, info = env.step(TriageAction(select_bug_index=0))
        self.assertEqual(obs.current_sub_step, "priority")

    def test_select_critical_bug_gives_bonus(self):
        env = BugTriageEnvironment()
        env.reset("hard_triage", seed=42)
        # Find a critical bug
        for i, bug in enumerate(env._queue):
            if bug["_solution"]["priority"] == "critical":
                obs, r, done, info = env.step(TriageAction(select_bug_index=i))
                self.assertGreater(r, 0)
                break

    def test_queue_status_updates(self):
        env = BugTriageEnvironment()
        env.reset("hard_triage", seed=42)
        env.step(TriageAction(select_bug_index=0))
        self.assertEqual(env._queue_status[0], "in_progress")

    def test_easy_task_skips_select_bug(self):
        env = BugTriageEnvironment()
        obs = env.reset("easy_triage", seed=42)
        self.assertEqual(obs.current_sub_step, "priority")


class TestDeferAction(unittest.TestCase):
    def test_defer_gives_penalty(self):
        env = BugTriageEnvironment()
        env.reset("easy_triage", seed=42)
        obs, r, done, info = env.step(TriageAction(defer=True))
        self.assertLess(r, 0)
        self.assertEqual(obs.deferred_bugs, 1)

    def test_deferred_status_set(self):
        env = BugTriageEnvironment()
        env.reset("easy_triage", seed=42)
        env.step(TriageAction(defer=True))
        self.assertIn("deferred", env._queue_status)


class TestRevision(unittest.TestCase):
    def test_revision_uses_budget(self):
        env = BugTriageEnvironment()
        obs = env.reset("easy_triage", seed=42)
        initial_budget = obs.revision_budget
        results, obs, _ = _run_bug(env)
        obs, r, done, info = env.step(TriageAction(revise_field="priority", revise_value="critical"))
        self.assertEqual(obs.revision_budget, initial_budget - 1)

    def test_no_revision_after_budget_exhausted(self):
        env = BugTriageEnvironment()
        env.reset("easy_triage", seed=42)
        # Exhaust revision budget
        _run_bug(env)
        for _ in range(env._revision_budget):
            env.step(TriageAction(revise_field="priority", revise_value="high"))
        obs, r, done, info = env.step(TriageAction(revise_field="priority", revise_value="low"))
        self.assertIn("error", info)


class TestConfidenceScoring(unittest.TestCase):
    def test_overconfident_wrong_gets_penalty(self):
        env = BugTriageEnvironment()
        env.reset("easy_triage", seed=42)
        # Find a critical bug and classify it wrong with high confidence
        for i, bug in enumerate(env._queue):
            if bug["_solution"]["priority"] == "critical":
                env._bug_index = i
                env._sub_step_index = 1
                env._current_decisions = {}
                break
        obs, r, done, info = env.step(_a(priority="low", confidence=0.95))
        self.assertLess(r, 0.2)

    def test_low_confidence_wrong_less_penalty(self):
        # Use a non-critical bug so floor of 0.0 doesn't mask confidence difference
        env1 = BugTriageEnvironment()
        env2 = BugTriageEnvironment()
        env1.reset("easy_triage", seed=42); env2.reset("easy_triage", seed=42)
        for env in [env1, env2]:
            for i, bug in enumerate(env._queue):
                if bug["_solution"]["priority"] == "high":  # wrong=medium, not missed critical
                    env._bug_index = i; env._sub_step_index = 1; env._current_decisions = {}; break
        _, r_high_conf, _, _ = env1.step(_a(priority="low", confidence=0.95))
        _, r_low_conf,  _, _ = env2.step(_a(priority="low", confidence=0.3))
        # High confidence wrong should be penalised more (lower reward)
        self.assertGreaterEqual(r_low_conf, r_high_conf)


class TestConflictingSignals(unittest.TestCase):
    def test_conflicting_signal_bugs_exist(self):
        conflict_bugs = [b for b in BUG_BANK if b.get("conflicting_signal")]
        self.assertGreater(len(conflict_bugs), 0)

    def test_misleading_title_bug_is_critical(self):
        # "Minor display glitch" is actually SQL injection
        bug = next((b for b in BUG_BANK if "Minor display glitch" in b["title"]), None)
        self.assertIsNotNone(bug)
        self.assertEqual(bug["_solution"]["priority"], "critical")

    def test_urgent_title_staging_is_low(self):
        bug = next((b for b in BUG_BANK if "URGENT" in b["title"] and "staging" in b.get("severity_hint","").lower()), None)
        self.assertIsNotNone(bug)
        self.assertEqual(bug["_solution"]["priority"], "low")


class TestTeamLoad(unittest.TestCase):
    def test_team_load_tracked(self):
        env = BugTriageEnvironment()
        env.reset("easy_triage", seed=42)
        env._bug_index = 0; env._sub_step_index = 3; env._current_decisions = {}
        env.step(_a(assigned_team="backend"))
        self.assertGreater(env._team_load.get("backend",0), 0)

    def test_overloaded_team_reduces_reward(self):
        env = BugTriageEnvironment()
        env.reset("easy_triage", seed=42)
        env._team_load["backend"] = 5  # overloaded
        env._bug_index = 0; env._sub_step_index = 3; env._current_decisions = {}
        _, r, _, info = env.step(_a(assigned_team="backend"))
        self.assertIn("overloaded", info.get("explanation",""))


class TestStepBudget(unittest.TestCase):
    def test_budget_tracked(self):
        env = BugTriageEnvironment()
        obs = env.reset("easy_triage", seed=42)
        initial = obs.steps_remaining
        _a_action = _a(priority="medium")
        env.step(_a_action)
        obs2 = env._build_obs()
        self.assertEqual(obs2.steps_remaining, initial - 1)

    def test_budget_exhaustion_ends_episode(self):
        env = BugTriageEnvironment()
        env.reset("easy_triage", seed=42)
        env._step_budget = 2  # force tiny budget
        env.step(_a(priority="medium"))
        env.step(_a(priority="medium"))
        obs, r, done, info = env.step(_a(priority="medium"))
        self.assertTrue(done)


class TestPartialObservability(unittest.TestCase):
    def test_hidden_fields_are_none(self):
        env = BugTriageEnvironment()
        env.reset("hard_triage", seed=42)
        for i, bug in enumerate(env._queue):
            if bug["partially_observable"]:
                env._bug_index = i; env._sub_step_index = 1; env._current_decisions = {}
                obs = env._build_obs()
                self.assertIsNone(obs.current_bug.affected_users)
                break

    def test_info_request_reveals_fields(self):
        env = BugTriageEnvironment()
        env.reset("hard_triage", seed=42)
        for i, bug in enumerate(env._queue):
            if bug["partially_observable"]:
                env._bug_index = i; env._sub_step_index = 4; env._current_decisions = {}
                env.step(_a(needs_more_info=True))
                self.assertTrue(env._current_decisions.get("needs_more_info_revealed"))
                break


class TestBaseline(unittest.TestCase):
    def test_all_tasks_pass(self):
        from baseline import run_heuristic_baseline
        results = run_heuristic_baseline()
        self.assertEqual(len(results), 3)
        for r in results:
            self.assertTrue(r["passed"], f"{r['task_id']}: {r['episode_score']} < {r['passing_threshold']}")

    def test_reproducible(self):
        from baseline import run_heuristic_baseline
        r1 = run_heuristic_baseline()
        r2 = run_heuristic_baseline()
        for a, b in zip(r1, r2):
            self.assertEqual(a["episode_score"], b["episode_score"])


SERVER_URL = "http://localhost:17863"
def _post(p,d):
    req=urllib.request.Request(SERVER_URL+p,json.dumps(d).encode(),{"Content-Type":"application/json"})
    with urllib.request.urlopen(req,timeout=15) as r: return json.loads(r.read())
def _get(p):
    with urllib.request.urlopen(SERVER_URL+p,timeout=15) as r: return json.loads(r.read())
def _up():
    try: _get("/health"); return True
    except: return False

class TestHTTP(unittest.TestCase):
    _proc=None
    @classmethod
    def setUpClass(cls):
        cls._proc=subprocess.Popen(
            ["python3","-m","uvicorn","server:app","--host","0.0.0.0","--port","17863"],
            stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        for _ in range(30):
            if _up(): break
            time.sleep(0.5)
        else:
            cls._proc.terminate(); raise RuntimeError("Server didn't start")
    @classmethod
    def tearDownClass(cls):
        if cls._proc: cls._proc.terminate()

    def test_health_has_features(self):
        r=_get("/health")
        self.assertIn("features",r)
        self.assertIn("multi_queue",r["features"])

    def test_reset_easy(self):
        r=_post("/reset",{"task_id":"easy_triage","seed":42})
        self.assertEqual(r["current_sub_step"],"priority")

    def test_reset_hard_starts_select(self):
        r=_post("/reset",{"task_id":"hard_triage","seed":42})
        self.assertEqual(r["current_sub_step"],"select_bug")

    def test_step_with_confidence(self):
        _post("/reset",{"task_id":"easy_triage","seed":42})
        r=_post("/step",{"action":{"priority":"critical","confidence":0.9}})
        self.assertIn("reward",r)

    def test_step_defer(self):
        _post("/reset",{"task_id":"easy_triage","seed":42})
        r=_post("/step",{"action":{"defer":True}})
        self.assertLess(r["reward"],0)

    def test_state_has_all_metrics(self):
        r=_get("/state")
        for field in ["team_load","wrong_escalations","missed_criticals","deferred_bugs","revision_budget","label_tp"]:
            self.assertIn(field,r)

    def test_baseline_all_pass(self):
        r=_get("/baseline")
        for res in r["baseline_results"]:
            self.assertTrue(res["passed"])


if __name__ == "__main__":
    print("="*65)
    print("Bug Triage OpenEnv v4 — Test Suite")
    print("="*65)
    loader=unittest.TestLoader(); suite=unittest.TestSuite()
    for cls in [TestMultiQueue,TestDeferAction,TestRevision,TestConfidenceScoring,
                TestConflictingSignals,TestTeamLoad,TestStepBudget,
                TestPartialObservability,TestBaseline]:
        suite.addTests(loader.loadTestsFromTestCase(cls))
    if "--http" in sys.argv:
        suite.addTests(loader.loadTestsFromTestCase(TestHTTP))
        print("Including HTTP tests...\n")
    else:
        print("Unit tests only. Add --http for server tests.\n")
    runner=unittest.TextTestRunner(verbosity=2)
    result=runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
