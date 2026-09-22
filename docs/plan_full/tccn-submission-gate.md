# TCCN Submission Gate

Overall: **local-smoke-only**

## Notes

- F2-F5 remain local-smoke or missing; remote multi-seed evidence is not present.
- 6 smoke/mini/tiny table references still need final handling.

## Checks

| Check | Status | Return code | JSON |
|---|---|---:|---|
| `claim-lint` | pass | 0 | `paper/data/tccn_claim_lint.json` |
| `innovation-lint` | pass | 0 | `paper/data/tccn_innovation_lint.json` |
| `table-audit` | pass | 0 | `paper/data/tccn_table_audit.json` |
| `evidence-contract` | pass | 0 | `paper/data/tccn_evidence_contract.json` |
| `readiness-audit` | pass | 0 | `paper/data/tccn_readiness_audit.json` |

## Key Status

- claim_lint: `pass`
- innovation_lint: `pass`
- table_audit: `local-smoke-tables-present`
- table_replacements: `6`
- evidence_contract: `local-smoke-only`
- readiness_audit: `local-smoke-ready`
