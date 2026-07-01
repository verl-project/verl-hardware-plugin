## Summary

<!-- Brief description of what this PR does. -->

## Motivation

<!-- Why is this change needed? Link related issues if applicable. -->

## Changes

<!-- List the key changes made in this PR. -->

-

## Testing

<!-- How was this tested? Include commands, test results, or environment info. -->

- [ ] `pytest tests/ -v` passes
- [ ] Manually verified on target hardware (if applicable)

## Acceptance Baseline (for new hardware adaptation PRs)

<!-- 
If this PR adds or modifies a hardware platform/engine, you MUST complete this section.
Skip this section for documentation-only or non-platform changes.
-->

- [ ] Ran `scripts/baseline_grpo_gsm8k.sh` on target hardware (8 devices)
- [ ] Training completed all epochs without error
- [ ] `critic/rewards/mean` shows clear upward trend in first 100 steps
- [ ] Curve is consistent with [NVIDIA reference](https://swanlab.cn/@heavyrain/verl_grpo_gsm8k_math/runs/8h196r8o/chart)

**SwanLab or training log link**: <!-- paste your run link here -->

**Reward curve comparison** (first 100 steps):
<!-- 
Attach a screenshot or overlay comparing your critic/rewards/mean curve 
against the NVIDIA reference. Paste image or link below.
-->

## Checklist

- [ ] Code follows the project's style and passes `pre-commit` checks
- [ ] Documentation updated (if applicable)
- [ ] No secrets or credentials included
