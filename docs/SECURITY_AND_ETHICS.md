# Security, Ethics, and Responsible Disclosure

## Authorized scope

Use the system only on:

- models and infrastructure you own;
- models whose licenses explicitly permit the research;
- third-party APIs for which you have written authorization and terms permit automated evaluation;
- isolated red-team environments with defined scope.

Do not use it to bypass authentication, rate limits, output restrictions, contractual controls, or confidentiality obligations.

## Separate threat claims

The following are measured independently:

1. **Fingerprinting:** identify model/version/lineage.
2. **Property inference:** infer architecture or training properties.
3. **Functional extraction:** create a substitute with similar behavior.
4. **Parameter extraction:** infer internal parameter values up to symmetry.
5. **Training-data extraction:** recover memorized data.

Evidence for one is not evidence for the others. Training-data extraction is outside the default repository and requires a separate ethics and privacy review.

## Data minimization

- Never record secrets, personal data, proprietary production prompts, or user conversations in query banks or AIM.
- Use synthetic and public corpora by default.
- Redact credentials from environment snapshots.
- Store test traces and target checkpoints in access-controlled object storage.
- Use short-lived cloud credentials and instance roles, not static keys in configs.
- Encrypt object storage and scheduler logs at rest.

## AIM deployment

The provided compose deployment is for isolated research networks. Before internet exposure, add:

- authentication and authorization;
- TLS termination;
- network allowlists/security groups;
- encrypted storage and backups;
- audit logging;
- retention/deletion policy;
- protection against logging sensitive prompts or environment variables.

AIM is metadata infrastructure, not a secret manager.

## Remote model code

Hugging Face `trust_remote_code` is false by default. Enabling it executes repository-provided code. Pin a commit, inspect the code, run in a sandbox without credentials, and record approval in the manifest.

## Public reporting

Before publishing a result that materially improves unauthorized extraction:

1. reproduce under clean conditions;
2. quantify the exact output interface and query count;
3. test simple mitigations such as reduced logit precision/top-k access, rate limits, and response randomization;
4. contact affected model/API providers under coordinated disclosure when appropriate;
5. avoid publishing live credentials, exploit scripts against third-party endpoints, or recoverable proprietary parameters.

The repository supports scientific extraction experiments but is not configured to target commercial services.

## Risk/benefit review checklist

- Is the target authorized?
- Could the query corpus include sensitive information?
- Are outputs more revealing than necessary for the research question?
- Can the experiment be performed on synthetic/open models instead?
- Are researchers counting training-data leakage without privacy approval?
- Is the claimed vulnerability reproducible and bounded?
- Does the release enable harm disproportionate to interpretability benefits?
- Is there a responsible-disclosure plan?
