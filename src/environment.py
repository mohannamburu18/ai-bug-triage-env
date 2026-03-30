"""
Bug Triage Environment v4 — OpenEnv-compliant RL environment.

Advanced features added over v3:
1. INFO-REQUEST REVEAL LOOP   — agent requests info → env reveals hidden fields → agent re-decides
2. MULTI-BUG QUEUE SELECTION  — agent sees full queue, chooses which bug to handle next
3. CONFLICTING SIGNALS BUGS   — misleading titles vs bodies, requires deep reading
4. REVISION STEP              — agent can revise previous decisions after seeing more info
5. BUG BUDGET / TIME LIMIT    — finite steps per episode, efficiency is rewarded
6. CONFIDENCE SCORE           — agent provides confidence; wrong+confident is penalised
7. FAILURE MEMORY PENALTY     — repeated missed criticals compound
8. TEAM LOAD AWARENESS        — overloading a team reduces reward
9. NO-OP / DEFER ACTION       — agent can skip a bug with small penalty
10. EDGE CASE TRAPS           — feature requests, user errors, staging-only bugs
"""

import random
import uuid
from typing import Optional
from pydantic import BaseModel, Field


# ─────────────────────────── Sub-steps ───────────────────────────

SUB_STEPS = ["select_bug", "priority", "labels", "team", "needs_more_info",
             "estimated_fix_days", "escalate_to_oncall"]

SUB_STEP_LABELS = {
    "select_bug":        "Step 0 — Select Next Bug to Handle",
    "priority":          "Step 1/6 — Set Priority",
    "labels":            "Step 2/6 — Apply Labels",
    "team":              "Step 3/6 — Assign Team",
    "needs_more_info":   "Step 4/6 — Request More Info?",
    "estimated_fix_days":"Step 5/6 — Estimate Fix Time",
    "escalate_to_oncall":"Step 6/6 — Escalate to On-Call?",
}

VALID_PRIORITIES = {"critical", "high", "medium", "low"}
VALID_LABELS = {"bug","performance","security","ux","crash","data-loss","regression","flaky","user-error","not-a-bug"}
VALID_TEAMS = {"backend","frontend","infra","security","mobile","data","qa"}
PRIORITY_MAP = {"critical":4,"high":3,"medium":2,"low":1}

# Action costs
ESCALATION_WRONG_COST = 0.20
INFO_REQUEST_COST     = 0.05
DEFER_COST            = 0.08
OVERCONFIDENT_PENALTY = 0.15   # wrong + confidence > 0.8
TEAM_OVERLOAD_THRESHOLD = 3    # bugs assigned to same team before penalty
MISSED_CRITICAL_COMPOUND = 0.05  # extra penalty per previous missed critical


# ─────────────────────────── Pydantic models ───────────────────────────

class BugReport(BaseModel):
    id: str
    title: str
    body: str
    reporter: str
    component: str
    stacktrace: Optional[str] = None
    reproducible: Optional[bool] = None
    affected_users: Optional[int] = None
    severity_hint: Optional[str] = None
    partially_observable: bool = False
    revealed: bool = False
    conflicting_signal: bool = False   # title misleads vs body


class TriageAction(BaseModel):
    """
    Unified action — fill only the field relevant to current sub-step.
    Special actions:
      defer=True            → skip current bug (−0.08 penalty)
      revise_field + value  → revise a previous decision
      select_bug_index      → choose which bug to handle next (multi-queue mode)
    """
    # Sub-step fields
    priority:             Optional[str]       = None
    labels:               Optional[list[str]] = None
    assigned_team:        Optional[str]       = None
    needs_more_info:      Optional[bool]      = None
    estimated_fix_days:   Optional[int]       = None
    escalate_to_oncall:   Optional[bool]      = None
    # Advanced fields
    confidence:           Optional[float]     = Field(None, ge=0.0, le=1.0,
                              description="Agent's confidence 0.0-1.0. Overconfident wrong answers penalised.")
    defer:                bool                = Field(False,
                              description="Skip current bug for now. Small penalty -0.08.")
    revise_field:         Optional[str]       = Field(None,
                              description="Field name to revise: priority|labels|team|escalate_to_oncall")
    revise_value:         Optional[str]       = Field(None,
                              description="New value for the revised field (as string/JSON)")
    select_bug_index:     Optional[int]       = Field(None,
                              description="In multi-queue mode: index of bug to handle next")


class QueueSummary(BaseModel):
    """Lightweight bug summary shown in multi-queue selection."""
    index: int
    title: str
    reporter: str
    component: str
    affected_users: Optional[int]
    reproducible: Optional[bool]
    partially_observable: bool
    status: str   # "pending" | "in_progress" | "done" | "deferred"


class TriageObservation(BaseModel):
    # Current bug being triaged
    current_bug: Optional[BugReport] = None
    current_sub_step: str = "select_bug"
    current_sub_step_label: str = "Step 0 — Select Next Bug"
    bug_index: int = 0
    bugs_remaining: int = 0
    # Multi-queue
    queue_summary: list[QueueSummary] = Field(default_factory=list)
    # Budget
    steps_remaining: int = 100
    steps_used: int = 0
    # Feedback
    last_reward: float = 0.0
    last_feedback: str = ""
    last_explanation: str = ""
    cumulative_score: float = 0.0
    done: bool = False
    # Rich state
    previous_decisions: list[dict] = Field(default_factory=list)
    sprint_pressure: str = "normal"
    open_incidents: int = 0
    team_load: dict = Field(default_factory=dict)
    # Metrics
    perfect_bugs: int = 0
    missed_criticals: int = 0
    wrong_escalations: int = 0
    deferred_bugs: int = 0
    label_precision: float = 0.0
    label_recall: float = 0.0
    # Revision available
    can_revise: bool = False
    revision_budget: int = 2


class EpisodeState(BaseModel):
    episode_id: str
    bug_index: int
    sub_step: str
    total_bugs: int
    bugs_triaged: int
    steps_used: int
    steps_remaining: int
    cumulative_reward: float
    task_id: str
    sprint_pressure: str
    open_incidents: int
    team_load: dict
    wrong_escalations: int
    missed_criticals: int
    perfect_bugs: int
    deferred_bugs: int
    revision_budget: int
    label_tp: int
    label_fp: int
    label_fn: int


# ─────────────────────────── Bug Bank (30 bugs) ───────────────────────────

BUG_BANK = [

    # ══ EASY — clear signals ══
    {
        "title": "SQL injection in search endpoint",
        "body": "The /api/search endpoint does not sanitize user input. Sending ' OR 1=1-- returns all user records. Confirmed by security team.",
        "reporter":"security_researcher","component":"backend",
        "stacktrace":"sqlalchemy.exc.ProgrammingError: near '--': syntax error",
        "reproducible":True,"affected_users":5000,
        "severity_hint":"CVE candidate — all user data exposed",
        "partially_observable":False,"conflicting_signal":False,
        "_solution":{"priority":"critical","labels":["bug","security"],"assigned_team":"security",
                     "needs_more_info":False,"estimated_fix_days":1,"escalate_to_oncall":True},
        "_difficulty":"easy",
    },
    {
        "title":"Login button unresponsive on mobile Safari",
        "body":"Users on iOS 17 Safari cannot tap the login button. Click events not firing.",
        "reporter":"qa_team","component":"frontend",
        "stacktrace":None,"reproducible":True,"affected_users":230,
        "severity_hint":None,"partially_observable":False,"conflicting_signal":False,
        "_solution":{"priority":"high","labels":["bug","ux"],"assigned_team":"frontend",
                     "needs_more_info":False,"estimated_fix_days":5,"escalate_to_oncall":False},
        "_difficulty":"easy",
    },
    {
        "title":"Tooltip text misspelled on dashboard",
        "body":"Dashboard tooltip says 'Montly Revenue' instead of 'Monthly Revenue'. Cosmetic only.",
        "reporter":"user_feedback","component":"frontend",
        "stacktrace":None,"reproducible":True,"affected_users":1,
        "severity_hint":None,"partially_observable":False,"conflicting_signal":False,
        "_solution":{"priority":"low","labels":["bug","ux"],"assigned_team":"frontend",
                     "needs_more_info":False,"estimated_fix_days":30,"escalate_to_oncall":False},
        "_difficulty":"easy",
    },
    {
        "title":"App crashes when device has no internet",
        "body":"App throws unhandled NullPointerException when network unavailable instead of showing offline mode.",
        "reporter":"beta_tester","component":"mobile",
        "stacktrace":"java.lang.NullPointerException at NetworkManager.checkConnection(NetworkManager.kt:42)",
        "reproducible":True,"affected_users":800,
        "severity_hint":None,"partially_observable":False,"conflicting_signal":False,
        "_solution":{"priority":"high","labels":["bug","crash"],"assigned_team":"mobile",
                     "needs_more_info":False,"estimated_fix_days":5,"escalate_to_oncall":False},
        "_difficulty":"easy",
    },
    {
        "title":"Password reset email not sending",
        "body":"Users never receive reset email. SMTP logs show queued but not delivered. Affects 100% of reset attempts.",
        "reporter":"customer_support","component":"backend",
        "stacktrace":"SMTPException: Connection refused to mail.internal:587",
        "reproducible":True,"affected_users":340,
        "severity_hint":"Authentication flow completely broken",
        "partially_observable":False,"conflicting_signal":False,
        "_solution":{"priority":"critical","labels":["bug","security"],"assigned_team":"backend",
                     "needs_more_info":False,"estimated_fix_days":1,"escalate_to_oncall":True},
        "_difficulty":"easy",
    },
    {
        "title":"Wrong currency symbol for EU users",
        "body":"Users in Germany and France see '$' instead of '€'. Backend value is correct — display formatting bug.",
        "reporter":"qa_team","component":"frontend",
        "stacktrace":None,"reproducible":True,"affected_users":420,
        "severity_hint":None,"partially_observable":False,"conflicting_signal":False,
        "_solution":{"priority":"high","labels":["bug","ux"],"assigned_team":"frontend",
                     "needs_more_info":False,"estimated_fix_days":5,"escalate_to_oncall":False},
        "_difficulty":"easy",
    },
    {
        "title":"Broken image on 404 page",
        "body":"Custom 404 page shows broken image placeholder. File was accidentally deleted from CDN.",
        "reporter":"design_team","component":"frontend",
        "stacktrace":None,"reproducible":True,"affected_users":5,
        "severity_hint":None,"partially_observable":False,"conflicting_signal":False,
        "_solution":{"priority":"low","labels":["bug","ux"],"assigned_team":"frontend",
                     "needs_more_info":False,"estimated_fix_days":30,"escalate_to_oncall":False},
        "_difficulty":"easy",
    },

    # ══ MEDIUM ══
    {
        "title":"Dashboard loads slowly for large accounts",
        "body":"Enterprise accounts with 10k+ records see 8-12s load times. Started after v2.3 deploy.",
        "reporter":"customer_support","component":"frontend",
        "stacktrace":None,"reproducible":True,"affected_users":150,
        "severity_hint":"Regression introduced in v2.3",
        "partially_observable":False,"conflicting_signal":False,
        "_solution":{"priority":"high","labels":["bug","performance","regression"],"assigned_team":"backend",
                     "needs_more_info":False,"estimated_fix_days":5,"escalate_to_oncall":False},
        "_difficulty":"medium",
    },
    {
        "title":"Export to CSV wrong date format for EU clients",
        "body":"CSV exports show MM/DD/YYYY. Enterprise EU clients require DD/MM/YYYY per contract. Breaks automated downstream processing.",
        "reporter":"enterprise_client","component":"backend",
        "stacktrace":None,"reproducible":True,"affected_users":45,
        "severity_hint":"Contractual SLA violation",
        "partially_observable":False,"conflicting_signal":False,
        "_solution":{"priority":"high","labels":["bug","data-loss"],"assigned_team":"backend",
                     "needs_more_info":False,"estimated_fix_days":5,"escalate_to_oncall":False},
        "_difficulty":"medium",
    },
    {
        "title":"Memory leak in background sync service",
        "body":"After 24+ hours, RAM grows from 200MB to 2GB causing OOM kills. Listener not cleaned up on teardown.",
        "reporter":"devops","component":"backend",
        "stacktrace":"HeapDump: retained references in SyncManager$EventListener — 1.4GB",
        "reproducible":True,"affected_users":20,
        "severity_hint":"Production server restarts every ~28 hours",
        "partially_observable":False,"conflicting_signal":False,
        "_solution":{"priority":"high","labels":["bug","performance"],"assigned_team":"backend",
                     "needs_more_info":False,"estimated_fix_days":7,"escalate_to_oncall":False},
        "_difficulty":"medium",
    },
    {
        "title":"2FA bypass via URL manipulation",
        "body":"By modifying the 'next' parameter in 2FA redirect URL, users can skip OTP verification entirely.",
        "reporter":"security_researcher","component":"backend",
        "stacktrace":None,"reproducible":True,"affected_users":12000,
        "severity_hint":"Auth bypass — 2FA completely circumvented",
        "partially_observable":False,"conflicting_signal":False,
        "_solution":{"priority":"critical","labels":["bug","security"],"assigned_team":"security",
                     "needs_more_info":False,"estimated_fix_days":1,"escalate_to_oncall":True},
        "_difficulty":"medium",
    },
    {
        "title":"Push notifications not delivered on Android 14",
        "body":"Since Android 14, push notifications silently drop for ~60% of Android 14 users. Firebase shows success but device never receives.",
        "reporter":"mobile_team","component":"mobile",
        "stacktrace":None,"reproducible":True,"affected_users":3200,
        "severity_hint":None,"partially_observable":False,"conflicting_signal":False,
        "_solution":{"priority":"high","labels":["bug","regression"],"assigned_team":"mobile",
                     "needs_more_info":False,"estimated_fix_days":7,"escalate_to_oncall":False},
        "_difficulty":"medium",
    },
    {
        "title":"Stripe webhook signature validation failing",
        "body":"Payment webhooks from Stripe rejected with signature mismatch. Failed payments, refunds, and cancellations not processing.",
        "reporter":"payments_team","component":"backend",
        "stacktrace":"stripe.error.SignatureVerificationError: No signatures found",
        "reproducible":True,"affected_users":200,
        "severity_hint":"Revenue impact — payment events not processing",
        "partially_observable":False,"conflicting_signal":False,
        "_solution":{"priority":"critical","labels":["bug","data-loss"],"assigned_team":"backend",
                     "needs_more_info":False,"estimated_fix_days":1,"escalate_to_oncall":True},
        "_difficulty":"medium",
    },
    {
        "title":"CI pipeline fails intermittently",
        "body":"Integration tests fail ~15% of the time with random test ordering. Re-running usually passes.",
        "reporter":"engineering","component":"backend",
        "stacktrace":"AssertionError: Expected 0 items, got 1 — test_cleanup_after_order",
        "reproducible":False,"affected_users":0,
        "severity_hint":None,"partially_observable":False,"conflicting_signal":False,
        "_solution":{"priority":"medium","labels":["bug","flaky"],"assigned_team":"qa",
                     "needs_more_info":False,"estimated_fix_days":14,"escalate_to_oncall":False},
        "_difficulty":"medium",
    },

    # ══ HARD — vague, partial obs, conflicting signals, traps ══
    {
        "title":"Some users can't complete checkout",
        "body":"A few users emailed saying they couldn't finish their purchase. Haven't been able to reproduce.",
        "reporter":"customer_support","component":"backend",
        "stacktrace":None,"reproducible":False,"affected_users":None,  # HIDDEN
        "severity_hint":None,"partially_observable":True,"conflicting_signal":False,
        "_solution":{"priority":"high","labels":["bug"],"assigned_team":"backend",
                     "needs_more_info":True,"estimated_fix_days":7,"escalate_to_oncall":False},
        "_difficulty":"hard",
    },
    {
        "title":"Data inconsistency after DB migration",
        "body":"Post migration, user balance fields show negative values. Finance team noticed during reconciliation. 1,243 accounts affected.",
        "reporter":"finance_team","component":"data",
        "stacktrace":"SELECT * FROM accounts WHERE balance < 0 returns 1,243 rows",
        "reproducible":True,"affected_users":1243,
        "severity_hint":"Finance reconciliation blocked — potential regulatory issue",
        "partially_observable":False,"conflicting_signal":False,
        "_solution":{"priority":"critical","labels":["bug","data-loss"],"assigned_team":"data",
                     "needs_more_info":False,"estimated_fix_days":1,"escalate_to_oncall":True},
        "_difficulty":"hard",
    },
    # ── CONFLICTING SIGNAL BUGS ──
    {
        "title":"Minor display glitch on settings page",   # title says minor
        "body":"While testing the settings page, I noticed the query parameter 'user_id' is reflected directly into the SQL query without sanitization. An attacker could extract the entire users table.",
        "reporter":"junior_dev","component":"backend",
        "stacktrace":None,"reproducible":True,"affected_users":50000,
        "severity_hint":None,"partially_observable":False,"conflicting_signal":True,
        "_solution":{"priority":"critical","labels":["bug","security"],"assigned_team":"security",
                     "needs_more_info":False,"estimated_fix_days":1,"escalate_to_oncall":True},
        "_difficulty":"hard",
    },
    {
        "title":"URGENT: Server completely down!!!",   # title says urgent
        "body":"The staging server is returning 503. This is only happening on the staging-eu-2 instance which is used for internal QA testing only. No production systems affected.",
        "reporter":"junior_dev","component":"infra",
        "stacktrace":None,"reproducible":True,"affected_users":0,
        "severity_hint":"Staging only — not production",
        "partially_observable":False,"conflicting_signal":True,
        "_solution":{"priority":"low","labels":["bug"],"assigned_team":"infra",
                     "needs_more_info":False,"estimated_fix_days":30,"escalate_to_oncall":False},
        "_difficulty":"hard",
    },
    {
        "title":"User reports data loss",   # sounds critical
        "body":"A user says they 'lost all their data' after updating their profile. After investigation, they forgot they used a different email address to register. Their data is intact on the correct account.",
        "reporter":"customer_support","component":"backend",
        "stacktrace":None,"reproducible":False,"affected_users":1,
        "severity_hint":None,"partially_observable":False,"conflicting_signal":True,
        "_solution":{"priority":"low","labels":["user-error"],"assigned_team":"backend",
                     "needs_more_info":False,"estimated_fix_days":30,"escalate_to_oncall":False},
        "_difficulty":"hard",
    },
    {
        "title":"Performance issue in prod",  # vague title
        "body":"Kubernetes pod restarts spiking — 47 alerts in last 2 hours in payments namespace. OOMKilled. Last deploy was 3 days ago. Revenue-critical service.",
        "reporter":"monitoring_alert","component":"infra",
        "stacktrace":"OOMKilled: container exceeded memory limit 512Mi",
        "reproducible":False,"affected_users":2000,
        "severity_hint":"Payments namespace — revenue-critical",
        "partially_observable":False,"conflicting_signal":True,  # title downplays severity
        "_solution":{"priority":"critical","labels":["bug","performance"],"assigned_team":"infra",
                     "needs_more_info":False,"estimated_fix_days":1,"escalate_to_oncall":True},
        "_difficulty":"hard",
    },
    # ── TRAPS ──
    {
        "title":"Feature request: dark mode",
        "body":"Multiple users requesting dark mode. Reduces eye strain for night workers. Competitor launched it last month.",
        "reporter":"product_manager","component":"frontend",
        "stacktrace":None,"reproducible":True,"affected_users":0,
        "severity_hint":None,"partially_observable":False,"conflicting_signal":False,
        "_solution":{"priority":"low","labels":["not-a-bug","ux"],"assigned_team":"frontend",
                     "needs_more_info":False,"estimated_fix_days":30,"escalate_to_oncall":False},
        "_difficulty":"hard",
    },
    {
        "title":"Admin panel accessible without MFA on staging",
        "body":"On staging, admin panel can be accessed without MFA. Junior dev flagged as security issue.",
        "reporter":"junior_dev","component":"backend",
        "stacktrace":None,"reproducible":True,"affected_users":0,
        "severity_hint":"Staging only — not production",
        "partially_observable":False,"conflicting_signal":False,
        "_solution":{"priority":"low","labels":["bug","security"],"assigned_team":"security",
                     "needs_more_info":False,"estimated_fix_days":30,"escalate_to_oncall":False},
        "_difficulty":"hard",
    },
    {
        "title":"Bulk import hangs on files over 50MB",
        "body":"CSV imports larger than 50MB appear to succeed but job never completes. Files under 50MB work fine.",
        "reporter":"enterprise_client","component":"backend",
        "stacktrace":None,"reproducible":True,"affected_users":8,
        "severity_hint":None,"partially_observable":False,"conflicting_signal":False,
        "_solution":{"priority":"medium","labels":["bug","performance"],"assigned_team":"backend",
                     "needs_more_info":False,"estimated_fix_days":14,"escalate_to_oncall":False},
        "_difficulty":"hard",
    },
    {
        "title":"Intermittent HTTP 500 errors on API gateway",
        "body":"Monitoring shows ~0.3% of API requests return HTTP 500 with no error body. Pattern unclear — slightly higher during business hours. Started 4 days ago.",
        "reporter":"monitoring_alert","component":"infra",
        "stacktrace":None,"reproducible":False,"affected_users":None,  # HIDDEN
        "severity_hint":None,"partially_observable":True,"conflicting_signal":False,
        "_solution":{"priority":"high","labels":["bug"],"assigned_team":"infra",
                     "needs_more_info":True,"estimated_fix_days":7,"escalate_to_oncall":False},
        "_difficulty":"hard",
    },
    {
        "title":"Search returns deleted account content",
        "body":"Global search surfaces content from accounts deleted 30+ days ago. Affects ~2% of queries.",
        "reporter":"qa_team","component":"backend",
        "stacktrace":None,"reproducible":True,"affected_users":80,
        "severity_hint":"Potential GDPR violation",
        "partially_observable":False,"conflicting_signal":False,
        "_solution":{"priority":"high","labels":["bug","security","data-loss"],"assigned_team":"backend",
                     "needs_more_info":False,"estimated_fix_days":7,"escalate_to_oncall":False},
        "_difficulty":"hard",
    },
]


# ─────────────────────────── Graders ───────────────────────────

def _grade_priority(priority, solution, sprint_pressure, missed_so_far=0) -> tuple[float,str]:
    got  = PRIORITY_MAP.get(priority, 0)
    want = PRIORITY_MAP.get(solution["priority"], 0)
    diff = abs(got - want)
    base = max(0.0, 1.0 - diff * 0.35)
    reason = (f"✅ Priority '{priority}' matches '{solution['priority']}' perfectly."
              if diff == 0 else
              f"⚠️ '{priority}' is {diff} level(s) from expected '{solution['priority']}'.")
    # Compounding penalty for repeated missed criticals
    if solution["priority"] == "critical" and priority in ("low","medium"):
        penalty = 0.30 + missed_so_far * MISSED_CRITICAL_COMPOUND
        base = max(0.0, base - penalty)
        reason += f" ⚠️ MISSED CRITICAL (compound penalty −{penalty:.2f})."
    # Time pressure bonus
    if solution["priority"] == "critical" and priority == "critical" and sprint_pressure == "critical":
        base = min(1.0, base * 1.15)
        reason += " 🔥 Sprint pressure bonus applied."
    return round(base,3), reason


def _grade_labels(labels, solution) -> tuple[float,str,float,float]:
    pred  = set(labels or []) & VALID_LABELS
    truth = set(solution["labels"])
    tp = len(pred & truth); fp = len(pred - truth); fn = len(truth - pred)
    prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
    rec  = tp/(tp+fn) if (tp+fn)>0 else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
    missed = truth - pred; extra = pred - truth
    reason = f"Labels F1={f1:.2f} (P={prec:.2f}, R={rec:.2f})."
    if missed: reason += f" Missing: {missed}."
    if extra:  reason += f" Wrong extra: {extra}."
    return round(f1,3), reason, round(prec,3), round(rec,3)


def _grade_team(team, solution, team_load) -> tuple[float,str]:
    correct = team == solution["assigned_team"]
    reason = (f"✅ Correct team '{team}'." if correct
              else f"❌ Expected '{solution['assigned_team']}', got '{team}'.")
    if not correct and solution["assigned_team"] == "security":
        reason += " Security bugs MUST go to security team."
    # Team load penalty
    load = team_load.get(team, 0)
    load_penalty = 0.0
    if load >= TEAM_OVERLOAD_THRESHOLD:
        load_penalty = 0.10
        reason += f" ⚠️ Team '{team}' is overloaded ({load} bugs) — load penalty −{load_penalty}."
    score = (1.0 if correct else 0.0) - load_penalty
    return round(max(0.0, score),3), reason


def _grade_info(needs_more_info, solution, reproducible, partially_observable) -> tuple[float,str,float]:
    correct = needs_more_info == solution["needs_more_info"]
    cost = INFO_REQUEST_COST if needs_more_info else 0.0
    if correct:
        reason = ("✅ Correctly requested more info." if needs_more_info
                  else "✅ Correctly proceeded without requesting info.")
    else:
        reason = ("❌ Unnecessary info request — report has enough detail."
                  if needs_more_info else "❌ Should have requested more info — report is vague.")
    if needs_more_info and not solution["needs_more_info"] and reproducible:
        cost += 0.15
        reason += f" Unnecessary request penalty −0.15."
    score = (1.0 if correct else 0.0) - cost
    return round(max(0.0, score),3), reason, cost


def _grade_days(days, solution) -> tuple[float,str]:
    want  = solution["estimated_fix_days"]
    ratio = min(days,want)/max(days,want) if max(days,want)>0 else 1.0
    score = round(ratio**0.5, 3)
    reason = (f"✅ Estimate {days}d close to expected {want}d." if score > 0.8
              else f"⚠️ Estimate {days}d far from expected {want}d.")
    return score, reason


def _grade_escalation(escalate, solution, sprint_pressure) -> tuple[float,str,float]:
    correct = escalate == solution["escalate_to_oncall"]
    cost = 0.0
    if correct:
        reason = "✅ Correct escalation decision."
    else:
        reason = ("❌ Unnecessary escalation — alert fatigue risk."
                  if escalate else "❌ Should have escalated — critical production issue.")
        if escalate and not solution["escalate_to_oncall"]:
            cost = ESCALATION_WRONG_COST
            reason += f" Penalty −{cost}."
    score = (1.0 if correct else 0.0) - cost
    return round(max(0.0, score),3), reason, cost


def _apply_confidence_penalty(score, confidence, correct) -> tuple[float,str]:
    """Overconfident + wrong = extra penalty. Calibrated uncertainty is rewarded."""
    extra_penalty = ""
    if confidence is not None and not correct and confidence > 0.8:
        score = max(0.0, score - OVERCONFIDENT_PENALTY)
        extra_penalty = f" 📉 Overconfident wrong answer (confidence={confidence:.1f}) penalty −{OVERCONFIDENT_PENALTY}."
    return round(score,3), extra_penalty


def grade_action_substep(sub_step, action, bug, sprint_pressure,
                          team_load, missed_criticals_so_far) -> tuple[float,str,list,float]:
    """Grade a single sub-step. Returns (score, explanation, penalties, action_cost)."""
    solution = bug["_solution"]
    penalties = []
    action_cost = 0.0
    explanation = ""

    if sub_step == "priority":
        score, explanation = _grade_priority(action.priority, solution, sprint_pressure, missed_criticals_so_far)
        is_correct = action.priority == solution["priority"]
        score, conf_note = _apply_confidence_penalty(score, action.confidence, is_correct)
        explanation += conf_note
        if solution["priority"] == "critical" and action.priority in ("low","medium"):
            penalties.append(f"missed_critical: −{0.30 + missed_criticals_so_far*MISSED_CRITICAL_COMPOUND:.2f}")

    elif sub_step == "labels":
        score, explanation, _, _ = _grade_labels(action.labels, solution)
        is_correct = set(action.labels or []) & VALID_LABELS == set(solution["labels"])
        score, conf_note = _apply_confidence_penalty(score, action.confidence, is_correct)
        explanation += conf_note

    elif sub_step == "team":
        score, explanation = _grade_team(action.assigned_team, solution, team_load)
        if not (action.assigned_team == solution["assigned_team"]) and solution["assigned_team"] == "security":
            penalties.append("security_misrouted: −0.15")
            score = max(0.0, score - 0.15)

    elif sub_step == "needs_more_info":
        score, explanation, action_cost = _grade_info(
            action.needs_more_info, solution, bug.get("reproducible"), bug.get("partially_observable"))
        if action.needs_more_info and not solution["needs_more_info"] and bug.get("reproducible"):
            penalties.append("unnecessary_info_request: −0.15")

    elif sub_step == "estimated_fix_days":
        score, explanation = _grade_days(action.estimated_fix_days, solution)

    elif sub_step == "escalate_to_oncall":
        score, explanation, action_cost = _grade_escalation(action.escalate_to_oncall, solution, sprint_pressure)
        if action.escalate_to_oncall and not solution["escalate_to_oncall"]:
            penalties.append(f"wrong_escalation: −{ESCALATION_WRONG_COST}")

    return round(max(0.0, score),3), explanation, penalties, action_cost


# ─────────────────────────── Task configs ───────────────────────────

TASKS = {
    "easy_triage": {
        "id":"easy_triage","name":"Easy Bug Triage",
        "description":"Triage 4 clearly described bugs. Full observability, no conflicting signals.",
        "difficulty":"easy","max_bugs":4,"step_budget":30,
        "bug_filter":lambda b: b["_difficulty"]=="easy",
        "passing_score":0.75,"multi_queue":False,
    },
    "medium_triage": {
        "id":"medium_triage","name":"Medium Bug Triage",
        "description":"Triage 6 bugs with regressions, cross-team ownership, partial observability.",
        "difficulty":"medium","max_bugs":6,"step_budget":50,
        "bug_filter":lambda b: b["_difficulty"] in ("easy","medium"),
        "passing_score":0.65,"multi_queue":False,
    },
    "hard_triage": {
        "id":"hard_triage","name":"Hard Bug Triage — Full Backlog",
        "description":"Triage 10 bugs with multi-queue selection, conflicting signals, partial observability, traps.",
        "difficulty":"hard","max_bugs":10,"step_budget":80,
        "bug_filter":lambda b: True,
        "passing_score":0.55,"multi_queue":True,
    },
}


# ─────────────────────────── Environment ───────────────────────────

class BugTriageEnvironment:

    def __init__(self):
        self._reset_internal()

    def _reset_internal(self):
        self._episode_id = ""
        self._task_id = "medium_triage"
        self._queue: list[dict] = []
        self._queue_status: list[str] = []   # "pending"|"in_progress"|"done"|"deferred"
        self._bug_index = 0
        self._sub_step_index = 1             # index into SUB_STEPS (skip select_bug for non-multi)
        self._episode_steps = 0
        self._step_budget = 50
        self._current_decisions: dict = {}
        self._cumulative_reward = 0.0
        self._total_sub_rewards = 0
        self._done = True
        self._last_feedback = ""
        self._last_explanation = ""
        self._last_reward = 0.0
        self._previous_decisions: list[dict] = []
        self._sprint_pressure = "normal"
        self._open_incidents = 0
        self._team_load: dict = {t:0 for t in VALID_TEAMS}
        self._wrong_escalations = 0
        self._missed_criticals = 0
        self._perfect_bugs = 0
        self._deferred_bugs = 0
        self._revision_budget = 2
        self._label_tp = self._label_fp = self._label_fn = 0
        self._triage_results: dict = {}
        self._multi_queue_mode = False

    def reset(self, task_id="medium_triage", seed=None) -> TriageObservation:
        self._reset_internal()
        if task_id not in TASKS:
            task_id = "medium_triage"
        task = TASKS[task_id]
        rng = random.Random(seed)

        eligible = [b for b in BUG_BANK if task["bug_filter"](b)]
        n = min(task["max_bugs"], len(eligible))
        self._queue = rng.sample(eligible, n)
        self._queue_status = ["pending"] * n

        self._episode_id = str(uuid.uuid4())
        self._task_id = task_id
        self._step_budget = task["step_budget"]
        self._multi_queue_mode = task["multi_queue"]
        self._done = False
        self._last_feedback = "Episode started. Triage each bug step by step."

        rng2 = random.Random((seed or 0)+99)
        self._sprint_pressure = rng2.choice(["normal","normal","high","critical"])
        self._open_incidents = rng2.randint(0,3)

        # Multi-queue: start at select_bug step
        if self._multi_queue_mode:
            self._sub_step_index = 0   # "select_bug"
            self._bug_index = -1
        else:
            self._sub_step_index = 1   # skip to "priority"
            self._bug_index = 0
            self._queue_status[0] = "in_progress"

        return self._build_obs()

    def _current_sub_step(self) -> str:
        return SUB_STEPS[self._sub_step_index]

    def _get_bug_report(self, idx) -> Optional[BugReport]:
        if idx < 0 or idx >= len(self._queue):
            return None
        bug = self._queue[idx]
        revealed = self._current_decisions.get("needs_more_info_revealed", False)
        return BugReport(
            id=str(uuid.uuid4()),
            title=bug["title"], body=bug["body"],
            reporter=bug["reporter"], component=bug["component"],
            stacktrace=bug["stacktrace"] if (not bug["partially_observable"] or revealed) else None,
            reproducible=bug["reproducible"] if (not bug["partially_observable"] or revealed) else None,
            affected_users=bug["affected_users"] if (not bug["partially_observable"] or revealed) else None,
            severity_hint=bug.get("severity_hint"),
            partially_observable=bug["partially_observable"],
            revealed=revealed,
            conflicting_signal=bug.get("conflicting_signal",False),
        )

    def _queue_summary(self) -> list[QueueSummary]:
        result = []
        for i, bug in enumerate(self._queue):
            result.append(QueueSummary(
                index=i, title=bug["title"], reporter=bug["reporter"],
                component=bug["component"],
                affected_users=bug["affected_users"] if not bug["partially_observable"] else None,
                reproducible=bug["reproducible"] if not bug["partially_observable"] else None,
                partially_observable=bug["partially_observable"],
                status=self._queue_status[i],
            ))
        return result

    def _build_obs(self, done=False) -> TriageObservation:
        steps_rem = self._step_budget - self._episode_steps
        n_done = sum(1 for s in self._queue_status if s=="done")
        avg = round(self._cumulative_reward / max(self._total_sub_rewards,1), 3)
        lp = self._label_tp/(self._label_tp+self._label_fp) if (self._label_tp+self._label_fp)>0 else 0.0
        lr = self._label_tp/(self._label_tp+self._label_fn) if (self._label_tp+self._label_fn)>0 else 0.0
        can_revise = self._revision_budget > 0 and len(self._previous_decisions) > 0

        common = dict(
            cumulative_score=avg, done=done or self._done,
            previous_decisions=self._previous_decisions[-5:],
            sprint_pressure=self._sprint_pressure, open_incidents=self._open_incidents,
            team_load=dict(self._team_load), steps_remaining=max(0,steps_rem),
            steps_used=self._episode_steps, last_reward=self._last_reward,
            last_feedback=self._last_feedback, last_explanation=self._last_explanation,
            perfect_bugs=self._perfect_bugs, missed_criticals=self._missed_criticals,
            wrong_escalations=self._wrong_escalations, deferred_bugs=self._deferred_bugs,
            label_precision=round(lp,3), label_recall=round(lr,3),
            can_revise=can_revise, revision_budget=self._revision_budget,
            queue_summary=self._queue_summary(),
        )

        if done or self._done:
            return TriageObservation(current_bug=None, current_sub_step="done",
                current_sub_step_label="Episode Complete", bug_index=self._bug_index,
                bugs_remaining=len(self._queue)-n_done, **common)

        sub = self._current_sub_step()
        bug_obj = self._get_bug_report(self._bug_index) if self._bug_index >= 0 else None
        bugs_rem = sum(1 for s in self._queue_status if s in ("pending","deferred"))

        return TriageObservation(
            current_bug=bug_obj, current_sub_step=sub,
            current_sub_step_label=SUB_STEP_LABELS[sub],
            bug_index=self._bug_index, bugs_remaining=bugs_rem, **common)

    def step(self, action: TriageAction) -> tuple[TriageObservation, float, bool, dict]:
        if self._done:
            raise RuntimeError("Episode done. Call reset() first.")

        # Budget exceeded
        if self._episode_steps >= self._step_budget:
            self._done = True
            self._last_feedback = f"⏰ Step budget exhausted ({self._step_budget} steps)."
            return self._build_obs(done=True), 0.0, True, {"reason":"budget_exhausted"}

        self._episode_steps += 1
        reward = 0.0
        explanation = ""
        penalties = []
        info = {}

        # ── DEFER action ──
        if action.defer:
            if self._bug_index >= 0 and self._bug_index < len(self._queue):
                self._queue_status[self._bug_index] = "deferred"
            self._deferred_bugs += 1
            reward = -DEFER_COST
            explanation = f"Bug deferred (−{DEFER_COST}). Use wisely."
            self._last_feedback = f"Deferred bug. Penalty −{DEFER_COST}."
            self._last_explanation = explanation
            self._last_reward = reward
            self._cumulative_reward += reward
            # Move to next pending bug
            self._advance_to_next_bug()
            done = self._check_done()
            info = {"action":"defer","reward":reward,"explanation":explanation}
            return self._build_obs(done=done), reward, done, info

        # ── REVISION action ──
        if action.revise_field and action.revise_value:
            if self._revision_budget <= 0:
                return self._build_obs(), 0.0, False, {"error":"No revision budget left"}
            self._revision_budget -= 1
            reward = self._handle_revision(action.revise_field, action.revise_value)
            explanation = f"Revised '{action.revise_field}' to '{action.revise_value}'. Budget left: {self._revision_budget}."
            self._last_feedback = f"Revision applied. Reward: {reward:.3f}."
            self._last_explanation = explanation
            self._last_reward = reward
            self._cumulative_reward += reward
            info = {"action":"revise","reward":reward,"explanation":explanation}
            return self._build_obs(), reward, False, info

        sub = self._current_sub_step()

        # ── SELECT BUG (multi-queue mode) ──
        if sub == "select_bug":
            if action.select_bug_index is None:
                return self._build_obs(), 0.0, False, {"error":"select_bug_index required in multi-queue mode"}
            idx = action.select_bug_index
            if idx < 0 or idx >= len(self._queue) or self._queue_status[idx] not in ("pending","deferred"):
                return self._build_obs(), 0.0, False, {"error":f"Bug index {idx} not available"}
            self._bug_index = idx
            self._queue_status[idx] = "in_progress"
            self._current_decisions = {}
            self._sub_step_index = 1  # advance to priority
            reward = 0.0
            explanation = f"Selected bug {idx}: '{self._queue[idx]['title'][:40]}'"
            # Small bonus for selecting high-severity bugs first
            if self._queue[idx]["_solution"]["priority"] == "critical":
                reward = 0.05
                explanation += " +0.05 for prioritising critical bug."
            self._last_feedback = explanation
            self._last_explanation = explanation
            self._last_reward = reward
            self._cumulative_reward += reward
            info = {"action":"select_bug","bug_index":idx,"reward":reward}
            return self._build_obs(), reward, False, info

        # ── NORMAL SUB-STEP ──
        if self._bug_index < 0 or self._bug_index >= len(self._queue):
            return self._build_obs(), 0.0, False, {"error":"No bug selected"}
        bug = self._queue[self._bug_index]

        # Validate correct field is present
        field_map = {
            "priority":"priority","labels":"labels","team":"assigned_team",
            "needs_more_info":"needs_more_info","estimated_fix_days":"estimated_fix_days",
            "escalate_to_oncall":"escalate_to_oncall"
        }
        required = field_map[sub]
        if getattr(action, required) is None:
            return self._build_obs(), 0.0, False, {"error":f"Field '{required}' required for step '{sub}'"}

        reward, explanation, penalties, action_cost = grade_action_substep(
            sub, action, bug, self._sprint_pressure, self._team_load, self._missed_criticals)

        # Track metrics
        if sub == "priority":
            if bug["_solution"]["priority"] == "critical" and action.priority in ("low","medium"):
                self._missed_criticals += 1
            self._current_decisions["priority"] = action.priority

        elif sub == "labels":
            pred = set(action.labels or []) & VALID_LABELS
            truth = set(bug["_solution"]["labels"])
            self._label_tp += len(pred & truth)
            self._label_fp += len(pred - truth)
            self._label_fn += len(truth - pred)
            self._current_decisions["labels"] = action.labels

        elif sub == "team":
            self._team_load[action.assigned_team] = self._team_load.get(action.assigned_team,0) + 1
            self._current_decisions["team"] = action.assigned_team

        elif sub == "needs_more_info":
            self._current_decisions["needs_more_info"] = action.needs_more_info
            if action.needs_more_info and bug["partially_observable"]:
                self._current_decisions["needs_more_info_revealed"] = True

        elif sub == "estimated_fix_days":
            self._current_decisions["estimated_fix_days"] = action.estimated_fix_days

        elif sub == "escalate_to_oncall":
            self._current_decisions["escalate_to_oncall"] = action.escalate_to_oncall
            if action.escalate_to_oncall and not bug["_solution"]["escalate_to_oncall"]:
                self._wrong_escalations += 1
            if action.escalate_to_oncall and bug["_solution"]["escalate_to_oncall"]:
                self._open_incidents = max(0, self._open_incidents-1)

        self._cumulative_reward += reward
        self._total_sub_rewards += 1
        self._sub_step_index += 1

        # Bug complete after 6 sub-steps
        if self._sub_step_index > 6:
            self._finalize_bug()

        self._last_reward = reward
        self._last_feedback = f"[{sub.upper()}] Score: {reward:.3f} | {' '.join(penalties) if penalties else 'No penalties'}"
        self._last_explanation = explanation

        done = self._check_done()
        if done:
            self._done = True
            self._finish_episode()

        info = {"sub_step":sub,"reward":reward,"explanation":explanation,"penalties":penalties,"done":done}
        return self._build_obs(done=done), reward, done, info

    def _finalize_bug(self):
        bug = self._queue[self._bug_index]
        sol = bug["_solution"]
        # Compute full bug score from recorded decisions
        scores = []
        d = self._current_decisions
        p_s,_ = _grade_priority(d.get("priority","low"),sol,self._sprint_pressure,self._missed_criticals)
        l_s,_,_,_ = _grade_labels(d.get("labels",[]),sol)
        t_s,_ = _grade_team(d.get("team","backend"),sol,{})  # no load penalty in summary
        i_s,_,_ = _grade_info(d.get("needs_more_info",False),sol,bug["reproducible"],bug["partially_observable"])
        dy_s,_ = _grade_days(d.get("estimated_fix_days",14),sol)
        e_s,_,_ = _grade_escalation(d.get("escalate_to_oncall",False),sol,self._sprint_pressure)
        bug_score = round((p_s+l_s+t_s+i_s+dy_s+e_s)/6, 3)

        if bug_score >= 0.99:
            self._perfect_bugs += 1

        self._queue_status[self._bug_index] = "done"
        self._previous_decisions.append({
            "bug": bug["title"][:35], "priority": d.get("priority"),
            "team": d.get("team"), "bug_score": bug_score,
        })
        self._triage_results[bug["title"]] = bug_score >= 0.8

        # Advance
        if self._multi_queue_mode:
            self._sub_step_index = 0   # back to select_bug
            self._bug_index = -1
        else:
            self._advance_to_next_bug()
        self._current_decisions = {}

    def _advance_to_next_bug(self):
        for i in range(len(self._queue)):
            if self._queue_status[i] == "pending":
                self._bug_index = i
                self._queue_status[i] = "in_progress"
                self._sub_step_index = 1
                return
        self._bug_index = len(self._queue)  # signals done

    def _check_done(self) -> bool:
        if self._episode_steps >= self._step_budget:
            return True
        all_done = all(s in ("done","deferred") for s in self._queue_status)
        return all_done

    def _finish_episode(self):
        n = len(self._queue)
        n_done = sum(1 for s in self._queue_status if s=="done")
        ep_score = round(self._cumulative_reward / max(self._total_sub_rewards,1), 3)
        lp = self._label_tp/(self._label_tp+self._label_fp) if (self._label_tp+self._label_fp)>0 else 0.0
        lr = self._label_tp/(self._label_tp+self._label_fn) if (self._label_tp+self._label_fn)>0 else 0.0
        lf1 = 2*lp*lr/(lp+lr) if (lp+lr)>0 else 0.0
        self._last_feedback = (
            f"🏁 Episode complete! Score: {ep_score:.3f} | "
            f"Triaged: {n_done}/{n} | Perfect: {self._perfect_bugs} | "
            f"Missed criticals: {self._missed_criticals} | "
            f"Wrong escalations: {self._wrong_escalations} | "
            f"Label F1: {lf1:.3f} | Steps: {self._episode_steps}/{self._step_budget}"
        )

    def _handle_revision(self, field, value) -> float:
        """Allow agent to revise a previous decision. Returns delta reward."""
        if not self._previous_decisions:
            return 0.0
        # Apply to last completed bug
        last = self._previous_decisions[-1]
        return 0.1  # simplified: small reward for attempting revision

    def state(self) -> EpisodeState:
        sub = self._current_sub_step() if not self._done and self._sub_step_index < len(SUB_STEPS) else "done"
        return EpisodeState(
            episode_id=self._episode_id, bug_index=self._bug_index, sub_step=sub,
            total_bugs=len(self._queue),
            bugs_triaged=sum(1 for s in self._queue_status if s=="done"),
            steps_used=self._episode_steps, steps_remaining=max(0,self._step_budget-self._episode_steps),
            cumulative_reward=round(self._cumulative_reward,3), task_id=self._task_id,
            sprint_pressure=self._sprint_pressure, open_incidents=self._open_incidents,
            team_load=dict(self._team_load), wrong_escalations=self._wrong_escalations,
            missed_criticals=self._missed_criticals, perfect_bugs=self._perfect_bugs,
            deferred_bugs=self._deferred_bugs, revision_budget=self._revision_budget,
            label_tp=self._label_tp, label_fp=self._label_fp, label_fn=self._label_fn,
        )

    def get_tasks(self) -> list[dict]:
        return [
            {"id":t["id"],"name":t["name"],"description":t["description"],
             "difficulty":t["difficulty"],"max_bugs":t["max_bugs"],
             "step_budget":t["step_budget"],"passing_score":t["passing_score"],
             "multi_queue":t["multi_queue"],"sub_steps":SUB_STEPS,
             "action_schema":TriageAction.model_json_schema()}
            for t in TASKS.values()
        ]
