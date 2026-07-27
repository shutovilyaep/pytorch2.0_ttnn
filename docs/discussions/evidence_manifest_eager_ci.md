# Evidence manifest - pytorch2.0_ttnn Eager / CI event log

**Target Discussion:** `shutovilyaep/pytorch2.0_ttnn`
**FRAME-2:** public GitHub facts + labeled first-party chronology. No private screenshots.
**Verified:** 2026-07-27 via GitHub API under `shutovilyaep`.

## Claim levels

| Level | Meaning |
| --- | --- |
| **fact** | Public GitHub / PyPI / dated export |
| **recorded** | Verbatim public comment / DM wording allowed for publish |
| **first-party** | Author recollection without calendar/Slack export in this package |
| **open** | Hypothesis / incomplete gate |

## Public anchors

| Claim | Level | Anchor |
| --- | --- | --- |
| Contact / onboarding window ~2025-09 | fact (DM date) + first-party staffing | LinkedIn contact 2025-09-12 (wording not required here) |
| Public Eager instructions / bounty demand | fact | Discussion D1173; issue [#1073](https://github.com/tenstorrent/pytorch2.0_ttnn/issues/1073) |
| Joe closes `#1073` "Bounty not needed at this time" 2025-11-17 | recorded | `#1073` close comment by `jmalone-tt` |
| Fork PR `#1243` -> org PR `#1293` (same branch content) | fact | https://github.com/tenstorrent/pytorch2.0_ttnn/pull/1293 |
| `#1293` size: 41 commits, 54 files, +2980/-875 | fact | PR metadata |
| Last visible upstream commit 2025-11-20 `254f2642af1cfcf41d1d16675da49c0974c50e80` | fact | PR commits |
| Org check rollup SUCCESS on `254f2642` | fact | GraphQL `statusCheckRollup.state` = SUCCESS (verified 2026-07-27) |
| `validate-pr` SUCCESS | fact | https://github.com/tenstorrent/pytorch2.0_ttnn/actions/runs/19549722596/job/55977769461 |
| `cpp-extension-tests` SUCCESS | fact | https://github.com/tenstorrent/pytorch2.0_ttnn/actions/runs/19549722522/job/55977765817 |
| `tests-passed` SUCCESS | fact | https://github.com/tenstorrent/pytorch2.0_ttnn/actions/runs/19549722522/job/55979333638 |
| Review requested: `jmalone-tt`, `ayerofieiev-tt`, `kevinwuTT` | fact | PR metadata |
| No reviews on `#1293` (0 reviews, 0 review comments) | fact | empty reviews (verified 2026-07-27) |
| Clean mergeability window vs main: 2025-11-20 -> 2025-12-08 | fact | compare / conflict forensics; first conflict `#1310` |
| "Ready to merge = only button left" | open / partial | conflict-free + green CI != approved |
| Sala review on `#1243`: TT_METAL_HOME deprecated; why remove `add_subdirectory(tt-metal)` | recorded | `#1243` review thread |
| Author Discussion D1286 answering CMake approach | fact | archive / Discussion |
| Eager op PRs `#1225-1228`, `#1296` on fix branch | fact | GitHub (still Open as of 2026-07-27) |
| README deprecate `#1310` merged 2025-12-08 by `jmalone-tt` | fact | https://github.com/tenstorrent/pytorch2.0_ttnn/pull/1310 |
| Bounty model merges after roll-off (`#1335` 2026-02-20, `#1280` 2026-02-24) | fact | GitHub |
| Joe roll-off reason wording: "implementation speed on the PyTorch project and the tt-metal refactor" | recorded | 2026-02-06 (verbatim allowed; screenshots excluded) |
| Expectations call: Eager by New Year | first-party | no calendar export here; do **not** assert exact "Nov 10" |
| Cancelled / emptied recurring calls; Slack ignore | first-party | screenshots not in package |
| Internal "blocker" / Tech Debt Discussion plan | first-party | screenshots TODO / not attached |
| Deprecation announced on call with Self (+ Kevin) | first-party | public signal = `#1310` |
| Sabotage / diversion as fact | open - **do not publish as fact** |

## Deliberately excluded

- Slack/Teams/calendar screenshot binaries
- LinkedIn screenshot images / contact metadata beyond the allowed speed quote
- Exact "Nov 10" as proven date
- Invented English Sala quote "can be done differently" (use actual `#1243` wording)
- Evening ASR `20-22` / `0144`
