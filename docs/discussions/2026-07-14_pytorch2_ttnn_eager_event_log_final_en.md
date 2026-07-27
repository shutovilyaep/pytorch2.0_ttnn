# Event log: Eager-execution assignment, CI/build rescue `#1293`, review-gate silence, and deprecation

**Repo:** personal fork discussion mirror of public `tenstorrent/pytorch2.0_ttnn` history
**Style:** FRAME-2 - public GitHub facts + explicitly labeled first-party chronology. No private screenshots attached.

---

## Install (this fork)

**Linux x86_64, CPython 3.10 only.** No Tenstorrent hardware is required for import smoke tests.

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install 'torch-ttnn-shutov[pypi]'
python -c "import ttnn, torch_ttnn; from torch_ttnn.cpp_extension import ttnn_module; assert hasattr(ttnn_module,'as_torch_device'); print('ok')"
```

That pulls matching `ttnn-shutov==0.65.0.dev20251205` (metal pin `8dfb324099a`). Import names stay `torch_ttnn` / `ttnn`. OpenMPI ULFM is bundled in the wheel (no `LD_LIBRARY_PATH` / `TT_METAL_HOME`).

Packaging background (why upstream `pip install` paths do not work): [discussion #10](https://github.com/shutovilyaep/pytorch2.0_ttnn/discussions/10).

---

## 60-second summary

I was brought onto `pytorch2.0_ttnn` with an **Eager execution** focus (plus "learn the build system / start from the version-update CI job" as onboarding color - first-party). In practice the blocking work became a large **build/CI/packaging rescue** against a moving `tt-metal` dependency.

That rescue is public as [#1293](https://github.com/tenstorrent/pytorch2.0_ttnn/pull/1293) (via earlier fork [#1243](https://github.com/tenstorrent/pytorch2.0_ttnn/pull/1243)): about **41 commits / 54 files / +2980/-875**, last visible upstream iteration **2025-11-20**, review requested from `jmalone-tt`, `ayerofieiev-tt`, and `kevinwuTT`. On head commit `254f2642` the org check rollup is **SUCCESS** (three required checks green). The branch was **conflict-free vs `main` until 2025-12-08**; it still has **zero reviews** (0 approving reviews, 0 review comments) and was never merged.

In the same week, `jmalone-tt` closed the public Eager bounty [#1073](https://github.com/tenstorrent/pytorch2.0_ttnn/issues/1073) as **"Bounty not needed at this time"** (**2025-11-17**), while Eager op PRs continued. After roll-off, the written reason given was **"implementation speed on the PyTorch project and the tt-metal refactor"** - without naming a PR, metric, or decision owner.

---

## 1. Assignment framing vs delivered scope

**Public Eager demand existed:**

- Bounty issue [#1073](https://github.com/tenstorrent/pytorch2.0_ttnn/issues/1073) (Bert Large via `model.to("ttnn:0")`).
- Related public prioritization / instructions in project Discussions and issues (e.g. D1173, `#1215`).

**First-party onboarding color (not a staffing email export):** Eager ops were the sold focus; build/CI was framed as something to pick up along the way ("start with the job that bumps versions").

**Observed delivery:** the largest public PR under my handle in this repo is the CI/build modernization line `#1243` -> `#1293`, with Eager registration PRs (`#1225-1228`, `#1296`) stacked on the same fix branch. That is a **scope mismatch** between interview/onboarding framing and the merge-blocking work - not by itself proof of bad faith.

---

## 2. Why chasing only `tt-metal` would have drowned Eager

`#1293` is not a one-line CI tweak. Public changes span:

- packaging (`pyproject.toml` / scikit-build-core),
- CMake / C++ extension / RPATH / wheel verification,
- GitHub Actions,
- submodule / Docker / sfpi / git-lfs drift,
- API fixes chasing `tt-metal` / PyTorch headers.

**First-party:** a version-auto-update CI job kept moving pins while the project already failed to build cleanly - a moving swamp. Pushing Eager registration **on top of** a repaired build branch was the only path that did not postpone feature work forever behind dependency churn.

---

## 3. Timeline of the rescue PR (public)

| When | Event | Level |
| --- | --- | --- |
| 2025-10-20 | Fork PR `#1243` opened | fact |
| 2025-11-12 | `#1243` ready for review; visible rescue commits begin | fact |
| 2025-11-14 | `aliaksei-sala` review on `#1243`: `TT_METAL_HOME` is deprecated / error-prone; questions removing `add_subdirectory(tt-metal)` | recorded |
| 2025-11-17 | `jmalone-tt` closes `#1073`: "Bounty not needed at this time" | recorded |
| 2025-11-19 | `#1243` closed unmerged; org PR `#1293` opened with the same branch content; reviewers requested | fact |
| 2025-11-20 | Last visible upstream commit (`254f2642` @ 20:01 UTC); org check rollup **SUCCESS** by 20:19 UTC (`validate-pr`, `cpp-extension-tests`, `tests-passed`) | fact |
| 2025-11-21 | `#1296` unary Eager ops targeting `fix/tt_metal_bump` | fact |
| 2025-11-20 -> 2025-12-08 | Conflict-free window vs `main` | fact |
| 2025-12-08 | `#1310` README deprecate -> TT-Forge merged by `jmalone-tt` (first conflicting main commit vs `#1293` files) | fact |

### Precise wording discipline

- **Do claim:** substantial PR delivered; reviewers requested; org CI on head `254f2642` green (links below); still Open; conflict-free until `#1310`; reviews count = 0.
- **Do not claim as fact:** "on Nov 20 only Approve & Merge remained." Conflict-free + green CI != approved. Required reviews were never given.
- **Do not invent** an English Sala quote "can be done differently." Use the actual `#1243` review points above. First-party feeling that the feedback was diffuse/blocking is separate and labeled.

Org green check-runs on `254f2642` (verified 2026-07-27 via GitHub API):

| Check | Conclusion | Permalink |
| --- | --- | --- |
| `validate-pr` | SUCCESS | https://github.com/tenstorrent/pytorch2.0_ttnn/actions/runs/19549722596/job/55977769461 |
| `cpp-extension-tests` | SUCCESS | https://github.com/tenstorrent/pytorch2.0_ttnn/actions/runs/19549722522/job/55977765817 |
| `tests-passed` | SUCCESS | https://github.com/tenstorrent/pytorch2.0_ttnn/actions/runs/19549722522/job/55979333638 |

---

## 4. First-party communication chronology (labeled)

These items are **author recollection** unless a public GitHub event is cited. Calendar/Slack screenshots are **not** in this branch.

1. **Expectations conversation (Joe + Artem):** Joe wanted Eager merged by year-end / before holiday downtime, enough to run at least one model, plus some optimizations. I am **not** asserting a proven calendar date of "Nov 10" here.
2. **After the branch was in review shape:** recurring 2x/week meetings were emptied/cancelled from my side's calendar view; I joined Zoom alone; Slack pings got short reassurance then silence.
3. **Internal escalation plan:** report the block inside EPAM; plan a technical-debt / "blocker" discussion for non-urgent process debt.
4. **Deprecation signal:** on a call, Joe stated the project would be deprecated (Kevin present in recollection). Public repo signal: [#1310](https://github.com/tenstorrent/pytorch2.0_ttnn/pull/1310) on **2025-12-08**.
5. **After deprecate/roll-off:** bounty-oriented model PRs still merged in Feb 2026 (e.g. `#1335`, `#1280`) - public fact that activity did not universally freeze.

---

## 5. Roll-off reason vs public gate

Around **2026-02-03** I left the project chat (roll-off signal; export not attached). On **2026-02-06**, Joe's written reason (verbatim; screenshots excluded) was:

> Ultimately, it came down to implementation speed on the PyTorch project and the tt-metal refactor.

The same message also appreciated the work. No PR, failing check, metric, reviewer, or decision date was named.

**Why "slow implementation" is incomplete as an engineering RCA against the public record:**

- `#1293` merge/review cadence sat with requested maintainers and never received an approving review (reviews = 0) while head CI was green.
- Eager feature PRs depended on that unmerged fix branch.
- Closing `#1073` as "not needed" while Eager work continued creates a demand/label contradiction that needs a written explanation.
- A separate public record in `tt-metal` shows multiple merged infrastructure migrations under my handle (see the companion Discussion on `shutovilyaep/tt-metal`).

**H0 (open):** bandwidth / risk aversion on a large CI PR / project freeze - possible without proving intent.

---

## 6. Requests for engineering review

1. Which PR, check-run, or metric supported the "implementation speed" assessment for `pytorch2.0_ttnn`?
2. Who owned merge/review on `#1293` after 2025-11-20, and why was there no approving review while org CI on `254f2642` was green?
3. How does closing `#1073` as "not needed" on 2025-11-17 relate to the continued Eager PR line and the later speed label?
4. When was scope formally changed from Eager feature delivery to build/CI rescue, and who accepted that change?

---

## Public links

| Item | URL |
| --- | --- |
| `#1293` CI/build rescue | https://github.com/tenstorrent/pytorch2.0_ttnn/pull/1293 |
| `#1243` fork precursor | https://github.com/tenstorrent/pytorch2.0_ttnn/pull/1243 |
| `#1073` Eager bounty | https://github.com/tenstorrent/pytorch2.0_ttnn/issues/1073 |
| `#1296` unary on fix branch | https://github.com/tenstorrent/pytorch2.0_ttnn/pull/1296 |
| `#1310` deprecate README | https://github.com/tenstorrent/pytorch2.0_ttnn/pull/1310 |
| Packaging / pip install record | https://github.com/shutovilyaep/pytorch2.0_ttnn/discussions/10 |
| Evidence manifest (this package) | [evidence_manifest_eager_ci.md](../discussions/evidence_manifest_eager_ci.md) |

Companion `tt-metal` event log (op migrations, silent bundled revert of `#33725`, UB handling): published on `shutovilyaep/tt-metal` Discussions (see that repo after port).

---

## Explicitly not claimed here

- Maintainer sabotage / diversion as fact.
- Exact "Nov 10" expectations-call date.
- Mapping Joe's speed label to a named PR (he did not name one).
- Private Slack/calendar/LinkedIn screenshots.
- That green CI alone equals "ready to merge" without the required approving review.
