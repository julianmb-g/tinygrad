Lessons Learned for tinygrad:
- Maintain strict adherence to SPECS.md.

Lessons Learned:
- When sequencing test executions before prototype evaluations or coverage extractions in harness.yaml, always use ';' instead of '&&' to prevent test failures from short-circuiting coverage extraction.
- Prioritize resolving Tier 1 blockers in DESIGN.md to avoid halting the pipeline execution.
