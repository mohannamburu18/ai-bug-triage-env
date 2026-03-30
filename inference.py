"""
Inference script for Bug Triage OpenEnv v4.
Uses Hugging Face Inference API to run LLM agent against the environment.

Required environment variables:
- API_BASE_URL: The API endpoint (default: https://router.huggingface.co/hf)
- MODEL_NAME: The model identifier to use for inference
- HF_TOKEN: Your Hugging Face API token
"""

import os
import sys
import json
import requests

# Import environment
from environment import (
    BugTriageEnvironment, TriageAction, TASKS, SUB_STEPS
)

# ─────────────────────────── Configuration ───────────────────────────

# NEW Hugging Face Router API
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/hf")
MODEL_NAME = os.environ.get("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.3")
HF_TOKEN = os.environ.get("HF_TOKEN")

# Fallback to OPENAI_API_KEY if HF_TOKEN not set
if not HF_TOKEN:
    HF_TOKEN = os.environ.get("OPENAI_API_KEY")

if not HF_TOKEN:
    print("=" * 60)
    print("ERROR: HF_TOKEN environment variable is required")
    print("=" * 60)
    print("\nSet it using:")
    print("  export HF_TOKEN=hf_your_token_here")
    print("\nOr on Windows:")
    print("  set HF_TOKEN=hf_your_token_here")
    print("=" * 60)
    sys.exit(1)

print(f"Using API: {API_BASE_URL}")
print(f"Using Model: {MODEL_NAME}")

# ─────────────────────────── System Prompt ───────────────────────────

SYSTEM_PROMPT = """You are an expert software engineering triage assistant.
You will be given a bug report and a specific decision to make.
Respond ONLY with a valid JSON object containing exactly one field for the decision.

Decisions:
- priority: {"priority":"critical"|"high"|"medium"|"low","confidence":0.0-1.0}
- labels: {"labels":[...]} from [bug,performance,security,ux,crash,data-loss,regression,flaky,user-error,not-a-bug]
- team: {"assigned_team":"backend"|"frontend"|"infra"|"security"|"mobile"|"data"|"qa"}
- needs_more_info: {"needs_more_info":true|false}
- estimated_fix_days: {"estimated_fix_days":1-90}
- escalate_to_oncall: {"escalate_to_oncall":true|false}
- select_bug: {"select_bug_index":N}

CRITICAL RULES:
- Read the BODY carefully, not just the title — titles can be misleading
- Staging-only issues → low priority
- Feature requests → low + not-a-bug label
- SQL injection/auth bypass → critical + security + escalate
- Vague/non-reproducible → needs_more_info=true

Respond with ONLY valid JSON, no markdown, no explanation."""


# ─────────────────────────── LLM Call Function ───────────────────────────

def call_llm(user_message: str) -> dict:
    """Call the Hugging Face Router API and parse JSON response."""
    try:
        url = f"{API_BASE_URL}/{MODEL_NAME}"
        
        headers = {
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "application/json"
        }
        
        # Format prompt for instruct model
        full_prompt = f"""<s>[INST] {SYSTEM_PROMPT}

{user_message} [/INST]"""
        
        payload = {
            "inputs": full_prompt,
            "parameters": {
                "temperature": 0.1,
                "max_new_tokens": 200,
                "return_full_text": False
            }
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        
        if response.status_code != 200:
            print(f"  API Error: {response.status_code} - {response.text[:100]}")
            return None
        
        result = response.json()
        
        # Extract generated text
        if isinstance(result, list) and len(result) > 0:
            raw = result[0].get("generated_text", "")
        elif isinstance(result, dict):
            raw = result.get("generated_text", str(result))
        else:
            raw = str(result)
        
        raw = raw.strip()
        
        # Clean up response - remove markdown code blocks if present
        raw = raw.replace("```json", "").replace("```", "").strip()
        
        # Extract JSON from response
        if "{" in raw and "}" in raw:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            raw = raw[start:end]
        
        return json.loads(raw)
        
    except json.JSONDecodeError as e:
        print(f"  JSON Parse Error: {e}")
        return None
    except requests.exceptions.Timeout:
        print(f"  Request Timeout")
        return None
    except Exception as e:
        print(f"  LLM Error: {e}")
        return None


# ─────────────────────────── Helper Functions ───────────────────────────

def build_prompt(obs, sub_step: str) -> str:
    """Build the prompt for the current sub-step."""
    if sub_step == "select_bug":
        queue_info = "\n".join([
            f"[{q.index}] {q.title} | Component: {q.component} | "
            f"Affected: {q.affected_users or '?'} | Status: {q.status}"
            for q in obs.queue_summary
        ])
        return f"""Bug Queue:
{queue_info}

Select the most critical pending bug to handle first.
Respond with ONLY: {{"select_bug_index": N}}"""
    
    bug = obs.current_bug
    bug_info = f"""Bug Report:
Title: {bug.title}
Body: {bug.body}
Component: {bug.component}
Reporter: {bug.reporter}
Reproducible: {bug.reproducible}
Affected Users: {bug.affected_users if bug.affected_users is not None else 'Unknown'}
Stacktrace: {bug.stacktrace or 'None'}
Severity Hint: {bug.severity_hint or 'None'}"""

    prompts = {
        "priority": f"""{bug_info}

Decide the priority level for this bug.
Respond with ONLY: {{"priority": "critical"|"high"|"medium"|"low", "confidence": 0.0-1.0}}""",
        
        "labels": f"""{bug_info}

Select all applicable labels for this bug.
Options: bug, performance, security, ux, crash, data-loss, regression, flaky, user-error, not-a-bug
Respond with ONLY: {{"labels": ["label1", "label2", ...]}}""",
        
        "team": f"""{bug_info}

Assign this bug to the appropriate team.
Options: backend, frontend, infra, security, mobile, data, qa
Respond with ONLY: {{"assigned_team": "team_name"}}""",
        
        "needs_more_info": f"""{bug_info}

Does this bug report need more information before it can be triaged?
Respond with ONLY: {{"needs_more_info": true}} or {{"needs_more_info": false}}""",
        
        "estimated_fix_days": f"""{bug_info}

Estimate days to fix. Guidelines: critical=1-2, high=3-7, medium=7-21, low=14-30
Respond with ONLY: {{"estimated_fix_days": N}}""",
        
        "escalate_to_oncall": f"""{bug_info}

Should this bug be escalated to on-call engineers?
Only for: critical security issues, production outages, data loss
Respond with ONLY: {{"escalate_to_oncall": true}} or {{"escalate_to_oncall": false}}"""
    }
    
    return prompts.get(sub_step, "")


def get_fallback_action(sub_step: str) -> TriageAction:
    """Return a safe fallback action if LLM fails."""
    fallbacks = {
        "select_bug": TriageAction(select_bug_index=0),
        "priority": TriageAction(priority="medium", confidence=0.5),
        "labels": TriageAction(labels=["bug"]),
        "team": TriageAction(assigned_team="backend"),
        "needs_more_info": TriageAction(needs_more_info=False),
        "estimated_fix_days": TriageAction(estimated_fix_days=14),
        "escalate_to_oncall": TriageAction(escalate_to_oncall=False),
    }
    return fallbacks.get(sub_step, TriageAction())


# ─────────────────────────── Main Inference ───────────────────────────

def run_inference(task_id: str, seed: int = 42, verbose: bool = True) -> dict:
    """Run LLM inference on a single task."""
    env = BugTriageEnvironment()
    obs = env.reset(task_id=task_id, seed=seed)
    
    total_reward = 0.0
    steps = 0
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Task: {task_id} | Seed: {seed}")
        print(f"{'='*60}")
    
    while not obs.done:
        sub_step = obs.current_sub_step
        
        # Build prompt and call LLM
        prompt = build_prompt(obs, sub_step)
        llm_response = call_llm(prompt)
        
        # Create action from response or use fallback
        if llm_response:
            try:
                action = TriageAction(**llm_response)
            except Exception as e:
                if verbose:
                    print(f"  Action parse error: {e}")
                action = get_fallback_action(sub_step)
        else:
            action = get_fallback_action(sub_step)
        
        # Take step
        obs, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        
        if verbose and sub_step != "select_bug":
            explanation = info.get('explanation', '')[:50] if info.get('explanation') else ''
            print(f"  [{sub_step}] reward={reward:.3f} | {explanation}")
        
        if done:
            break
    
    # Calculate final score
    episode_score = round(total_reward / max(env._total_sub_rewards, 1), 3)
    task = TASKS[task_id]
    passed = episode_score >= task["passing_score"]
    
    result = {
        "task_id": task_id,
        "difficulty": task["difficulty"],
        "episode_score": episode_score,
        "passing_threshold": task["passing_score"],
        "passed": passed,
        "steps_used": steps,
        "total_reward": round(total_reward, 3),
    }
    
    if verbose:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"\n{status} | Score: {episode_score:.3f} | Threshold: {task['passing_score']}")
    
    return result


def run_all_tasks(seed: int = 42, verbose: bool = True) -> list:
    """Run inference on all tasks."""
    results = []
    
    print("\n" + "="*65)
    print("BUG TRIAGE OPENENV v4 — LLM INFERENCE")
    print("="*65)
    print(f"API Base URL: {API_BASE_URL}")
    print(f"Model: {MODEL_NAME}")
    print("="*65)
    
    for task_id in ["easy_triage", "medium_triage", "hard_triage"]:
        result = run_inference(task_id, seed=seed, verbose=verbose)
        results.append(result)
    
    # Print summary
    print("\n" + "="*65)
    print("SUMMARY")
    print("="*65)
    
    all_passed = True
    for r in results:
        status = "✅ PASS" if r["passed"] else "❌ FAIL"
        print(f"{status} | {r['task_id']:15s} | score={r['episode_score']:.3f} | threshold={r['passing_threshold']}")
        if not r["passed"]:
            all_passed = False
    
    print("="*65)
    print(f"Overall: {'✅ ALL TASKS PASSED' if all_passed else '❌ SOME TASKS FAILED'}")
    print("="*65)
    
    # Save results
    with open("inference_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to inference_results.json")
    
    return results


# ─────────────────────────── Entry Point ───────────────────────────

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run LLM inference on Bug Triage OpenEnv")
    parser.add_argument("--task", type=str, default=None, 
                        help="Specific task to run (easy_triage, medium_triage, hard_triage)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    
    args = parser.parse_args()
    
    if args.task:
        if args.task not in TASKS:
            print(f"Error: Unknown task '{args.task}'")
            print(f"Available tasks: {list(TASKS.keys())}")
            sys.exit(1)
        run_inference(args.task, seed=args.seed, verbose=not args.quiet)
    else:
        run_all_tasks(seed=args.seed, verbose=not args.quiet)
