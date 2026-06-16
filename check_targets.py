#!/usr/bin/env python3
"""Local validation of the structured-target pipeline in train_critic.py.

Stubs torch/peft/transformers (not installed locally; training is server-only)
so the data-side functions can be imported and exercised against the real
data/{train,dev,test}.jsonl. Checks:
  1. build_target runs on every row of every split (both formats).
  2. Structured-target stats: problems per row, quote localization, cleaning hits.
  3. Round-trip: parse_generated_critique(build_target(...)) recovers the
     problems and assessment (the eval parser understands the train format).
  4. No filler ("na"/"Nil") or cross-response comparisons survive cleaning.
  5. Target length distribution (chars) for max_length budgeting.
"""
import json
import re
import sys
import types

# ---- stub heavy deps before importing train_critic -------------------------
def _stub(name):
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod

torch = _stub("torch")
torch.Tensor = object
torch.tensor = lambda *a, **k: None
torch.long = torch.float = None
torch.no_grad = lambda: (lambda f: f)  # used as @torch.no_grad() at class scope
_stub("torch.distributed")
nn = _stub("torch.nn")
nn.Module = object
nn.Linear = nn.ReLU = nn.Sequential = object
_stub("torch.nn.functional")
_stub("torch.nn.parallel").DistributedDataParallel = object
tud = _stub("torch.utils.data")
tud.DataLoader = object
tud.Dataset = object
_stub("torch.utils.data.distributed").DistributedSampler = object
_stub("torch.utils")
peft = _stub("peft")
peft.LoraConfig = peft.TaskType = peft.get_peft_model = object
tf = _stub("transformers")
tf.AutoModelForCausalLM = tf.AutoTokenizer = object
tf.get_linear_schedule_with_warmup = object

import train_critic as tc  # noqa: E402

assert tc.ACTIVE_TARGET_FORMAT == "structured"


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


CROSS = re.compile(tc._CROSS_REF_RE.pattern, re.I)
FILLER_TOKENS = re.compile(r"(?<![A-Za-z])(?:na|nil|n/a)(?![A-Za-z])", re.I)

grand_fail = 0
for split in ("train", "dev", "test"):
    rows = load_jsonl("data/%s.jsonl" % split)
    n = len(rows)
    stats = {"rows": n, "with_problems": 0, "problem_items": 0, "quoted": 0,
             "unlocalized": 0, "empty_assessment": 0, "roundtrip_fail": 0,
             "cross_ref_leak": 0, "target_chars": []}
    for ex in rows:
        gold_scores = tc.extract_gold_scores(ex)
        gold_rationale = tc.extract_gold_rationale(ex)
        # plain format must still build (legacy path)
        tc.ACTIVE_TARGET_FORMAT = "plain"
        _ = tc.build_target(ex, gold_rationale, gold_scores.get("overall"))
        tc.ACTIVE_TARGET_FORMAT = "structured"
        target = tc.build_target(ex, gold_rationale, gold_scores.get("overall"))
        stats["target_chars"].append(len(target))

        problems = tc.extract_problem_items(ex)
        if problems:
            stats["with_problems"] += 1
        stats["problem_items"] += len(problems)
        for it in problems:
            if it["quote"]:
                stats["quoted"] += 1
            else:
                stats["unlocalized"] += 1
        if tc.extract_overall_rationale(ex) == "":
            stats["empty_assessment"] += 1

        # cleaning checks on what actually landed in the target
        if CROSS.search(target):
            stats["cross_ref_leak"] += 1

        # round-trip: the eval parser must recover what the target encodes
        rat, parsed_problems, assessment = tc.parse_generated_critique(target)
        ok = len(parsed_problems) == len(problems)
        for a, b in zip(parsed_problems, problems):
            ok = ok and a["dim"] == b["dim"] and a["rationale"] == b["rationale"]
            ok = ok and (a["quote"] == b["quote"])
        exp_assessment = tc.extract_overall_rationale(ex) or (
            "See the problems above." if problems else tc.NULL_RATIONALE)
        ok = ok and assessment == exp_assessment
        if not ok:
            stats["roundtrip_fail"] += 1
            if stats["roundtrip_fail"] <= 2:
                print("ROUNDTRIP FAIL sample (%s):" % split)
                print(target[:600])
                print("parsed problems:", parsed_problems[:3], "expected:", problems[:3])

    chars = sorted(stats["target_chars"])
    grand_fail += stats["roundtrip_fail"] + stats["cross_ref_leak"]
    print("\n== %s ==" % split)
    print("  rows=%d  with_problems=%d (%.0f%%)  items=%d (quoted=%d, unlocalized=%d)"
          % (n, stats["with_problems"], 100.0 * stats["with_problems"] / n,
             stats["problem_items"], stats["quoted"], stats["unlocalized"]))
    print("  empty_assessment=%d  roundtrip_fail=%d  cross_ref_leak_in_target=%d"
          % (stats["empty_assessment"], stats["roundtrip_fail"], stats["cross_ref_leak"]))
    print("  target chars: p50=%d p90=%d max=%d (~tokens: p50=%d p90=%d max=%d)"
          % (chars[n // 2], chars[int(n * 0.9)], chars[-1],
             chars[n // 2] // 4, chars[int(n * 0.9)] // 4, chars[-1] // 4))

# ---- show samples -----------------------------------------------------------
print("\n" + "=" * 72)
for split, idx in (("train", 2), ("dev", 0), ("test", 0)):
    rows = load_jsonl("data/%s.jsonl" % split)
    # pick a row with problems if idx-th has none
    ex = rows[idx]
    if not tc.extract_problem_items(ex):
        ex = next((r for r in rows if tc.extract_problem_items(r)), ex)
    gs = tc.extract_gold_scores(ex)
    print("\n--- SAMPLE TARGET (%s) ---" % split)
    print(tc.build_target(ex, tc.extract_gold_rationale(ex), gs.get("overall"))[:900])

# ---- cleaning unit checks ----------------------------------------------------
print("\n" + "=" * 72)
cases = [
    ("na", ""), ("Nil", ""), ("N/A.", ""), ("-", ""),
    ("Short and sweet. Not as thorough as responses 1 and 2.", "Short and sweet."),
    ("I also like this response as well, similarly though, I feel it could use more empathy.",
     "I also like this response as well, similarly though, I feel it could use more empathy."),
    ("This response is good.", "This response is good."),
    ("Compared to the first response, this is weak. The tone is cold.", "The tone is cold."),
]
clean_fail = 0
for src, want in cases:
    got = tc.clean_rationale_text(src)
    flag = "OK " if got == want else "FAIL"
    if got != want:
        clean_fail += 1
    print("  [%s] %r -> %r" % (flag, src, got))

print("\nRESULT:", "ALL CHECKS PASSED" if (grand_fail + clean_fail) == 0
      else "FAILURES: roundtrip/leak=%d clean=%d" % (grand_fail, clean_fail))
