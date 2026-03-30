"""Gradio UI v4 — multi-queue, conflicting signals, revision, confidence, defer."""
import gradio as gr
from environment import BugTriageEnvironment, TriageAction, TASKS

_env = BugTriageEnvironment()

def _fmt_bug(bug) -> str:
    if bug is None:
        return "## 🎉 Episode Complete!"
    lines = [f"### 🐛 {bug.title}"]
    if bug.conflicting_signal:
        lines.append("⚠️ **CONFLICTING SIGNAL** — read the body carefully, title may mislead!")
    lines.append(
        f"**Reporter:** `{bug.reporter}` | **Component:** `{bug.component}` | "
        f"**Affected:** {bug.affected_users if bug.affected_users is not None else '❓ hidden'} | "
        f"**Reproducible:** {'✅' if bug.reproducible else ('❌' if bug.reproducible is False else '❓ hidden')}"
    )
    lines.append(f"\n{bug.body}")
    if bug.stacktrace:
        lines.append(f"\n```\n{bug.stacktrace}\n```")
    if bug.severity_hint:
        lines.append(f"\n⚠️ **Hint:** {bug.severity_hint}")
    if bug.partially_observable and not bug.revealed:
        lines.append("\n🔒 *Some fields hidden. Check 'Needs more info' to reveal.*")
    return "\n".join(lines)

def _fmt_queue(queue_summary) -> str:
    if not queue_summary:
        return "_No queue._"
    rows = ["| # | Title | Component | Status |", "|---|-------|-----------|--------|"]
    for q in queue_summary:
        icon = {"pending":"⏳","in_progress":"🔄","done":"✅","deferred":"⏸️"}.get(q.status,"❓")
        rows.append(f"| {q.index} | {q.title[:40]} | {q.component} | {icon} {q.status} |")
    return "\n".join(rows)

def _fmt_history(decisions) -> str:
    if not decisions:
        return "_No decisions yet._"
    rows = ["| Bug | Priority | Team | Score |", "|-----|----------|------|-------|"]
    for d in decisions:
        rows.append(f"| {d['bug']} | {d.get('priority','?')} | {d.get('team','?')} | {d.get('bug_score','?')} |")
    return "\n".join(rows)

def start_episode(task_id, seed_str, state):
    seed = int(seed_str) if str(seed_str).strip().isdigit() else 42
    obs = _env.reset(task_id=task_id, seed=seed)
    state["obs"] = obs
    total = len(_env._queue)
    return (
        _fmt_bug(obs.current_bug),
        _fmt_queue(obs.queue_summary),
        f"🎯 **{TASKS[task_id]['name']}** | Bugs: {total} | Budget: {obs.steps_remaining} steps",
        f"🔥 Sprint: `{obs.sprint_pressure}` | 🚨 Incidents: `{obs.open_incidents}` | 🔄 Revision budget: `{obs.revision_budget}`",
        f"**{obs.current_sub_step_label}**",
        "", "0.000", f"Step 0 | Budget: {obs.steps_remaining}",
        _fmt_history([]), state,
    )

def submit_step(priority, labels, team, needs_info, days, escalate,
                confidence, defer, revise_field, revise_value, select_idx, state):
    obs = state.get("obs")
    if obs is None or obs.done:
        return (_fmt_bug(None), "", "⚠️ Start an episode first.", "—", "", "", "0.000", "—", "—", state)

    sub = obs.current_sub_step

    # Build action
    if defer:
        action = TriageAction(defer=True)
    elif revise_field and revise_value:
        action = TriageAction(revise_field=revise_field, revise_value=revise_value)
    elif sub == "select_bug":
        try:
            idx = int(select_idx)
        except:
            idx = 0
        action = TriageAction(select_bug_index=idx)
    else:
        kwargs = {k:None for k in ["priority","labels","assigned_team","needs_more_info","estimated_fix_days","escalate_to_oncall"]}
        conf = float(confidence) if confidence else None
        if sub == "priority":           kwargs["priority"] = priority; kwargs["confidence"] = conf
        elif sub == "labels":           kwargs["labels"] = labels
        elif sub == "team":             kwargs["assigned_team"] = team
        elif sub == "needs_more_info":  kwargs["needs_more_info"] = needs_info
        elif sub == "estimated_fix_days": kwargs["estimated_fix_days"] = int(days)
        elif sub == "escalate_to_oncall": kwargs["escalate_to_oncall"] = escalate
        try:
            action = TriageAction(**kwargs)
        except Exception as e:
            return (_fmt_bug(obs.current_bug), _fmt_queue(obs.queue_summary),
                    f"❌ {e}", obs.current_sub_step_label, "", "", str(obs.cumulative_score), "—",
                    _fmt_history(obs.previous_decisions), state)

    new_obs, reward, done, info = _env.step(action)
    state["obs"] = new_obs

    total = len(_env._queue)
    progress = f"Step {_env._episode_steps}/{total*6} | Budget left: {new_obs.steps_remaining}"

    feedback_lines = [f"**{info.get('sub_step', info.get('action','?')).upper()} → {reward:.3f}**"]
    if info.get("explanation"): feedback_lines.append(f"💡 _{info['explanation']}_")
    if info.get("penalties"):   feedback_lines.append(f"⚠️ {', '.join(info['penalties'])}")

    if done:
        s = _env.state()
        ep_score = round(_env._cumulative_reward / max(_env._total_sub_rewards,1), 3)
        lp = s.label_tp/(s.label_tp+s.label_fp) if (s.label_tp+s.label_fp)>0 else 0.0
        lr = s.label_tp/(s.label_tp+s.label_fn) if (s.label_tp+s.label_fn)>0 else 0.0
        lf1 = 2*lp*lr/(lp+lr) if (lp+lr)>0 else 0.0
        feedback_lines = [
            f"## 🏁 Final Score: **{ep_score:.3f}**\n",
            f"| Metric | Value |", f"|--------|-------|",
            f"| Perfect bugs | {s.perfect_bugs}/{total} |",
            f"| Missed criticals | {s.missed_criticals} |",
            f"| Wrong escalations | {s.wrong_escalations} |",
            f"| Deferred bugs | {s.deferred_bugs} |",
            f"| Label Precision | {lp:.3f} |",
            f"| Label Recall | {lr:.3f} |",
            f"| Label F1 | {lf1:.3f} |",
            f"| Steps used | {s.steps_used}/{s.steps_used+new_obs.steps_remaining} |",
        ]
        score_str = f"{ep_score:.3f}"
        progress = f"Done — {sum(1 for s in _env._queue_status if s=='done')}/{total} triaged"
    else:
        score_str = str(new_obs.cumulative_score)

    ctx = (f"🔥 Sprint: `{new_obs.sprint_pressure}` | 🚨 Incidents: `{new_obs.open_incidents}` | "
           f"🔄 Revision: `{new_obs.revision_budget}` | 💰 Budget: `{new_obs.steps_remaining}`")

    return (
        _fmt_bug(new_obs.current_bug),
        _fmt_queue(new_obs.queue_summary),
        "\n".join(feedback_lines),
        f"**{new_obs.current_sub_step_label}**" if not done else "✅ Done",
        ctx, "",
        score_str, progress,
        _fmt_history(new_obs.previous_decisions),
        state,
    )


def build_ui():
    with gr.Blocks(title="Bug Triage OpenEnv v4", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
# 🐛 Bug Triage OpenEnv v4
**Advanced Sequential RL** — multi-queue selection, conflicting signals, partial observability,
action costs, revision steps, confidence scoring, team load awareness, step budget.
[OpenEnv](https://github.com/meta-pytorch/OpenEnv)-compliant.
""")
        state = gr.State({})

        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### ⚙️ Setup")
                task_dd = gr.Dropdown(choices=list(TASKS.keys()), value="easy_triage", label="Task")
                seed_box = gr.Textbox(value="42", label="Seed")
                start_btn = gr.Button("▶ Start Episode", variant="primary")
                status_md = gr.Markdown("_Not started._")
                context_md = gr.Markdown("")
                gr.Markdown("### 🗂️ Bug Queue")
                queue_md = gr.Markdown("_Start to see queue._")
                gr.Markdown("### 📜 Decision History")
                history_md = gr.Markdown("_No decisions yet._")

            with gr.Column(scale=3):
                gr.Markdown("### 📋 Current Bug Report")
                bug_md = gr.Markdown("_Start an episode._")

        gr.Markdown("---")
        step_label = gr.Markdown("### 🎯 Current Decision")

        with gr.Row():
            with gr.Column():
                priority_dd = gr.Dropdown(["critical","high","medium","low"], value="medium", label="1️⃣ Priority")
                confidence_sl = gr.Slider(0.0, 1.0, value=0.8, step=0.05, label="🎯 Confidence (for priority)")
                team_dd = gr.Dropdown(["backend","frontend","infra","security","mobile","data","qa"], value="backend", label="3️⃣ Assigned Team")
                days_sl = gr.Slider(1, 90, value=7, step=1, label="5️⃣ Estimated Fix (days)")
            with gr.Column():
                labels_cb = gr.CheckboxGroup(
                    ["bug","performance","security","ux","crash","data-loss","regression","flaky","user-error","not-a-bug"],
                    value=["bug"], label="2️⃣ Labels")
                needs_cb = gr.Checkbox(label="4️⃣ 🔍 Needs more info", value=False)
                escalate_cb = gr.Checkbox(label="6️⃣ 🚨 Escalate to on-call", value=False)
            with gr.Column():
                gr.Markdown("#### 🔀 Advanced Actions")
                defer_cb = gr.Checkbox(label="⏸️ Defer this bug (−0.08)", value=False)
                select_idx = gr.Number(value=0, label="🗂️ Select bug index (multi-queue)", precision=0)
                revise_field = gr.Dropdown(
                    ["","priority","labels","team","escalate_to_oncall"],
                    value="", label="✏️ Revise field")
                revise_value = gr.Textbox(value="", label="✏️ Revise new value")

        submit_btn = gr.Button("✅ Submit Decision", variant="primary", size="lg")

        gr.Markdown("---")
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📊 Feedback & Explanation")
                feedback_md = gr.Markdown("_Submit a decision to see feedback._")
            with gr.Column():
                gr.Markdown("### 🏆 Score")
                score_box = gr.Textbox(label="Cumulative Score", value="0.000", interactive=False)
                progress_box = gr.Textbox(label="Progress", value="—", interactive=False)

        gr.Markdown("---")
        gr.Markdown("""### 📡 API
```
POST /reset    {"task_id":"hard_triage","seed":42}
POST /step     {"select_bug_index":2}                    ← multi-queue
POST /step     {"priority":"critical","confidence":0.9}  ← with confidence
POST /step     {"defer":true}                            ← defer bug
POST /step     {"revise_field":"priority","revise_value":"critical"}  ← revision
GET  /state    GET /tasks    GET /baseline    POST /grader
```""")

        inputs = [priority_dd, labels_cb, team_dd, needs_cb, days_sl, escalate_cb,
                  confidence_sl, defer_cb, revise_field, revise_value, select_idx, state]
        outputs = [bug_md, queue_md, feedback_md, step_label, context_md,
                   gr.Textbox(visible=False), score_box, progress_box, history_md, state]

        start_btn.click(start_episode,
            inputs=[task_dd, seed_box, state],
            outputs=[bug_md, queue_md, status_md, context_md, step_label,
                     feedback_md, score_box, progress_box, history_md, state])
        submit_btn.click(submit_step, inputs=inputs, outputs=outputs)

    return demo
