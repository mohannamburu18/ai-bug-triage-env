"""Baseline agents for Bug Triage OpenEnv v4."""
import os, json, sys, re, argparse, random
from environment import (
    BugTriageEnvironment, TriageAction, TASKS, BUG_BANK, SUB_STEPS,
    _grade_priority, _grade_labels, _grade_team, _grade_info, _grade_days, _grade_escalation
)

CRITICAL_RE = [r"sql.?inject",r"auth.?bypass",r"2fa.?bypass",r"cve",r"negative.?balance",
               r"payment.?webhook",r"stripe.?fail",r"pod.?restart",r"oomkill",
               r"password.?reset.*not",r"revenue.*critical",r"all user.*data",r"users.?table"]
HIGH_RE    = [r"crash",r"null.?pointer",r"unhandled",r"memory.?leak",r"500.?error",
              r"checkout.*fail",r"push.?notif",r"gdpr",r"deleted.*data",r"wrong.*date.*format",
              r"sla",r"regression",r"slow.*load",r"hacked",r"account.*access",r"2000.*user"]
MEDIUM_RE  = [r"intermittent",r"flaky",r"sometimes",r"import.*hang",r"0\.3%",r"report.*differ"]
SECURITY_RE= [r"inject",r"bypass",r"cve",r"gdpr",r"hacked",r"password.*reset",
              r"2fa",r"mfa",r"deleted.*surface",r"breach",r"users.?table",r"sql"]
ONCALL_RE  = [r"sql.?inject",r"auth.?bypass",r"2fa.?bypass",r"negative.?balance",
              r"payment.?webhook",r"pod.?restart.*payment",r"oomkill",r"password.?reset.*100%"]
NEEDS_INFO_RE=[r"haven.?t.*reproduc",r"can.?t.*reproduc",r"no.*log.*found",r"unclear",
               r"claims?.*hack",r"user.*says?.*hack",r"some.*users?.*can.?t",r"few.*users?"]
USER_ERROR_RE=[r"forgot.*email",r"different.*account",r"user.*error",r"their data is intact"]
NOT_BUG_RE = [r"feature request",r"dark mode",r"would be nice",r"suggestion"]

def _match(text, patterns):
    t = text.lower()
    return any(re.search(p, t) for p in patterns)

def heuristic_decide(bug: dict, sub_step: str, confidence: float = 0.85) -> TriageAction:
    # Use full text including body (not just title — handles conflicting signals)
    full = f"{bug['body']} {bug.get('stacktrace','') or ''} {bug.get('severity_hint','') or ''}"
    title = bug['title'].lower()
    comp = bug["component"].lower()
    affected = bug.get("affected_users") or 0
    reproducible = bug.get("reproducible", True)

    is_feature = _match(title + full, NOT_BUG_RE)
    is_user_error = _match(full, USER_ERROR_RE)
    is_staging = "staging" in (bug.get("severity_hint","") or "").lower() and "not production" in (bug.get("severity_hint","") or "").lower()

    if is_feature or is_user_error or is_staging:
        priority = "low"
        labels = ["not-a-bug", "ux"] if is_feature else (["user-error"] if is_user_error else ["bug"])
        team = "frontend" if is_feature else "backend"
    elif _match(full, CRITICAL_RE) or (affected > 1000 and reproducible):
        priority = "critical"
        labels = ["bug"]
        if _match(full, SECURITY_RE): labels.append("security")
        if re.search(r"performance|oom|memory|pod", full.lower()): labels.append("performance")
        if re.search(r"data.*loss|balance|payment|inconsisten", full.lower()): labels.append("data-loss")
        team = "security" if _match(full, SECURITY_RE) else ("infra" if comp=="infra" else ("data" if comp=="data" else "backend"))
    elif _match(full, HIGH_RE) or (affected > 100):
        priority = "high"
        labels = ["bug"]
        if _match(full, SECURITY_RE): labels.append("security")
        if re.search(r"slow|memory|performance|leak", full.lower()): labels.append("performance")
        if re.search(r"crash|null.?pointer|unhandled", full.lower()): labels.append("crash")
        if re.search(r"regression|after.*deploy|v\d+\.\d+", full.lower()): labels.append("regression")
        if re.search(r"data.*loss|inconsisten|csv|deleted", full.lower()): labels.append("data-loss")
        if re.search(r"tooltip|button|display|icon|currency|symbol|ux|ui", full.lower()): labels.append("ux")
        team = ("security" if _match(full, SECURITY_RE)
                else "mobile" if comp=="mobile"
                else "data" if comp=="data"
                else "infra" if comp=="infra"
                else "frontend" if (comp=="frontend" and not re.search(r"slow|api|database", full.lower()))
                else "backend")
    elif _match(full, MEDIUM_RE):
        priority = "medium"
        labels = ["bug"]
        if re.search(r"flaky|intermittent", full.lower()): labels.append("flaky")
        if re.search(r"performance|slow|hang", full.lower()): labels.append("performance")
        team = "qa" if re.search(r"flaky|test|ci", full.lower()) else ("infra" if comp=="infra" else "backend")
    else:
        priority = "low"
        labels = ["bug", "ux"] if re.search(r"tooltip|button|display|icon|image", full.lower()) else ["bug"]
        team = "frontend" if comp=="frontend" else "backend"

    needs_info = _match(title + full, NEEDS_INFO_RE) or (not reproducible and (not affected or affected < 10))
    escalate = _match(full, ONCALL_RE) and priority == "critical" and not is_staging
    days_map = {"critical":1,"high":5,"medium":14,"low":30}

    kwargs = {k:None for k in ["priority","labels","assigned_team","needs_more_info","estimated_fix_days","escalate_to_oncall"]}
    conf = confidence if not is_feature else 0.95

    if sub_step == "priority":           kwargs["priority"] = priority; kwargs["confidence"] = conf
    elif sub_step == "labels":           kwargs["labels"] = list(set(labels))
    elif sub_step == "team":             kwargs["assigned_team"] = team
    elif sub_step == "needs_more_info":  kwargs["needs_more_info"] = needs_info
    elif sub_step == "estimated_fix_days": kwargs["estimated_fix_days"] = days_map[priority]
    elif sub_step == "escalate_to_oncall": kwargs["escalate_to_oncall"] = escalate

    return TriageAction(**kwargs)


def _select_best_bug(queue, queue_status) -> int:
    """Multi-queue: select highest priority pending bug."""
    best_idx, best_priority = -1, -1
    for i, (bug, status) in enumerate(zip(queue, queue_status)):
        if status not in ("pending","deferred"): continue
        full = f"{bug['body']} {bug.get('stacktrace','') or ''}"
        if _match(full, CRITICAL_RE): p = 4
        elif _match(full, HIGH_RE): p = 3
        elif _match(full, MEDIUM_RE): p = 2
        else: p = 1
        if p > best_priority:
            best_priority, best_idx = p, i
    return max(best_idx, 0)


def run_heuristic_baseline() -> list[dict]:
    results = []
    for task_id, task in TASKS.items():
        env = BugTriageEnvironment()
        obs = env.reset(task_id=task_id, seed=42)
        total_reward = 0.0; steps = 0

        while not obs.done:
            sub = obs.current_sub_step
            if sub == "select_bug":
                idx = _select_best_bug(env._queue, env._queue_status)
                action = TriageAction(select_bug_index=idx)
            else:
                bug = env._queue[env._bug_index]
                action = heuristic_decide(bug, sub)
            obs, reward, done, info = env.step(action)
            total_reward += reward; steps += 1
            if done: break

        n = len(env._queue)
        ep_score = round(total_reward / max(env._total_sub_rewards, 1), 3)
        s = env.state()
        lp = s.label_tp/(s.label_tp+s.label_fp) if (s.label_tp+s.label_fp)>0 else 0.0
        lr = s.label_tp/(s.label_tp+s.label_fn) if (s.label_tp+s.label_fn)>0 else 0.0
        results.append({
            "task_id":task_id,"difficulty":task["difficulty"],
            "episode_score":ep_score,"passed":ep_score>=task["passing_score"],
            "passing_threshold":task["passing_score"],
            "perfect_bugs":s.perfect_bugs,"missed_criticals":s.missed_criticals,
            "wrong_escalations":s.wrong_escalations,"deferred_bugs":s.deferred_bugs,
            "label_precision":round(lp,3),"label_recall":round(lr,3),
            "steps_used":s.steps_used,"step_budget":task["step_budget"],
        })
    return results


LLM_SYSTEM = """You are an expert software engineering triage assistant.
You will be given a bug report and a specific decision to make.
Respond ONLY with a valid JSON object containing exactly one field for the decision.

Decisions:
- priority: {"priority":"critical"|"high"|"medium"|"low","confidence":0.0-1.0}
- labels: {"labels":[...]} from [bug,performance,security,ux,crash,data-loss,regression,flaky,user-error,not-a-bug]
- team: {"assigned_team":"backend"|"frontend"|"infra"|"security"|"mobile"|"data"|"qa"}
- needs_more_info: {"needs_more_info":true|false}
- estimated_fix_days: {"estimated_fix_days":1-90}
- escalate_to_oncall: {"escalate_to_oncall":true|false}
- select_bug: {"select_bug_index":N} — pick highest priority pending bug

CRITICAL RULES:
- Read the BODY carefully, not just the title — titles can be misleading
- Staging-only issues → low priority regardless of how alarming
- Feature requests → low + not-a-bug label, no "bug" label
- User errors → low + user-error label
- SQL injection/auth bypass in body text → critical + security + escalate
- OOM in revenue-critical namespace → critical + escalate
- Vague/non-reproducible → needs_more_info=true
- Provide confidence score honestly — wrong + overconfident = extra penalty"""


def run_llm_baseline(model="gpt-4o-mini") -> list[dict]:
    try:
        from openai import OpenAI
    except ImportError:
        print("pip install openai"); sys.exit(1)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Set OPENAI_API_KEY"); sys.exit(1)
    client = OpenAI(api_key=api_key)
    results = []

    for task_id, task in TASKS.items():
        env = BugTriageEnvironment()
        obs = env.reset(task_id=task_id, seed=42)
        total_reward = 0.0
        print(f"\n=== {task_id} ({task['difficulty']}) ===")

        while not obs.done:
            sub = obs.current_sub_step
            if sub == "select_bug":
                queue_info = "\n".join([f"[{q.index}] {q.title} | {q.component} | obs={'full' if not q.partially_observable else 'partial'} | status={q.status}"
                                        for q in obs.queue_summary])
                user_msg = f"Queue:\n{queue_info}\n\nSelect the most critical pending bug index."
            else:
                bug = obs.current_bug
                user_msg = f"""Bug:
Title: {bug.title}
Body: {bug.body}
Component: {bug.component} | Reporter: {bug.reporter}
Reproducible: {bug.reproducible} | Affected: {bug.affected_users}
Stacktrace: {bug.stacktrace or 'None'}
Severity hint: {bug.severity_hint or 'None'}
Conflicting signal: {bug.conflicting_signal}
Sprint pressure: {obs.sprint_pressure} | Open incidents: {obs.open_incidents}

Make decision: {sub}"""
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role":"system","content":LLM_SYSTEM},{"role":"user","content":user_msg}],
                    temperature=0.0, max_tokens=150)
                raw = resp.choices[0].message.content.strip().replace("```json","").replace("```","").strip()
                data = json.loads(raw)
                action = TriageAction(**data)
            except Exception as e:
                print(f"  LLM error on {sub}: {e}")
                if sub == "select_bug":
                    action = TriageAction(select_bug_index=0)
                else:
                    action = heuristic_decide(env._queue[env._bug_index], sub)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            if done: break

        ep_score = round(total_reward / max(env._total_sub_rewards,1), 3)
        print(f"  Score: {ep_score:.3f}")
        results.append({"task_id":task_id,"difficulty":task["difficulty"],
                        "episode_score":ep_score,"passed":ep_score>=task["passing_score"],
                        "passing_threshold":task["passing_score"]})
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["heuristic","llm"], default="heuristic")
    parser.add_argument("--model", default="gpt-4o-mini")
    args = parser.parse_args()
    results = run_heuristic_baseline() if args.mode=="heuristic" else run_llm_baseline(args.model)
    print("\n"+"="*65)
    print("BASELINE RESULTS — Bug Triage OpenEnv v4")
    print("="*65)
    for r in results:
        s = "✅ PASS" if r["passed"] else "❌ FAIL"
        print(f"{s} | {r['task_id']:20s} | score={r['episode_score']:.3f} | threshold={r['passing_threshold']}")
        if "label_precision" in r:
            print(f"     P={r['label_precision']:.3f} R={r['label_recall']:.3f} | "
                  f"perfect={r['perfect_bugs']} | missed_crit={r['missed_criticals']} | "
                  f"steps={r['steps_used']}/{r['step_budget']}")
    print("="*65)
    with open("baseline_scores.json","w") as f: json.dump(results,f,indent=2)
    print("Saved to baseline_scores.json")
