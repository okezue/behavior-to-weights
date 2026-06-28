# AWS Batch deployment

These files are credential-free templates. They assume an existing GPU-capable AWS Batch compute
environment, queue, ECR repository, S3 bucket, CloudWatch log group, and least-privilege IAM roles.

1. Build `Dockerfile`, tag it with the exact Git commit, and push it to ECR.
2. Replace placeholders in `job-definition.template.json`; register the job definition.
3. Keep manifests, query banks, configs, and outputs under immutable S3 prefixes.
4. Submit with `submitbatch.py`; use `--dry-run` to audit the request without AWS access.

```bash
python infra/aws/submitbatch.py --dry-run \
  --job-name b2w-tier2-zoo \
  --job-queue YOUR_QUEUE \
  --job-definition behavior-to-weights-gpu:1 \
  --command b2w zoo build-micro --config s3://.../zoo.yaml --output s3://.../zoo
```

The core CLI currently expects local paths. Production Batch jobs should stage pinned inputs from
S3 to instance storage, run the command, verify checksums, and upload outputs atomically. A wrapper
for that staging step is deliberately deployment-specific because encryption, VPC endpoints,
KMS keys, and bucket policies differ by organization.
