# ax-code scan — gpt-oss

**Agent:** ax-code CLI · model `zai-coding-plan/glm-5.2[1m]`  
**Mode:** wide static scan (code-only)  
**Batch report:** `../wave1-4-axcode-batch.md`  
**Shared substrate:** `../W0-shared/axcode-scan.md`

## Coverage map

- Registry / convert: architecture_registry + convert/model_family for `gpt_oss`
- Graph: families/gpt_oss.rs MXFP4 packed
- Runner MTP: mtp_model_policy fail-closed defaults (shared Wave 0 PASS)
- Server/CLI: aliases not re-audited beyond SUPPORTED-MODELS consistency

## Findings

_no open P0/P1 from static scan for this family after Wave 0 shared fixes_

## Dead code candidates

_none high-confidence family-local_

## Completeness self-score

70 (code-only; weights smoke not run)

## Residual LIMIT

MXFP4 residency; 120B needs 128GB class
