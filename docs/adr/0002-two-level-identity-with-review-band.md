# ADR 0002: Two-level identity with an explicit review band

Status: accepted

## Context

The business asks two different questions: "is this an employee or a customer?" and
"which employee is it?". Forcing a single classifier to answer both produces confident
wrong names, which is worse than no name for HR and legal reasons.

## Decision

1. Role first (employee vs customer), from cheap visual cues or a fine-tuned class.
2. Employees: appearance embedding matched against an authorised enrolment gallery keyed
   by `employee_id`, with two thresholds:
   - `sim >= match_threshold`: accept.
   - `review_threshold <= sim < match_threshold`: emit with `needs_review = true`.
   - below: `employee-unknown-*`, still tracked, never named.
3. Customers: matched only against an anonymous online gallery with a TTL. Never named.
4. Thresholds are derived from data (`rva reid-thresholds`) at a target false-accept rate.

## Consequences

- The system defaults to "unknown" instead of a wrong identity.
- The review band produces exactly the crops needed to retrain ReID (hard negatives).
- Employee ReID is a continuously trained system, not a one-shot model: enrolment and
  thresholds are re-run when uniforms, cameras or staff change.
