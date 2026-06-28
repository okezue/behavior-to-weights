# Self-hosted AIM 3.29.1

This deployment pins the latest AIM 3.x release used by the repository adapter. It provides a
remote tracking endpoint on port `53800` and the web UI on `43800`.

```bash
cd infra/aim
docker compose up -d --build
curl --fail http://localhost:43800 || true
export AIM_REPO=aim://localhost:53800
```

Configure a run with:

```yaml
tracking:
  backend: aim
  repo: aim://localhost:53800
  experiment: behavior-to-weights
```

The server and UI share a named Docker volume. Back up that volume before upgrades. For an
internet-facing deployment, put both ports behind authenticated TLS, expose only the UI to users,
and restrict the tracking RPC endpoint to the training network.

AIM 3.29.1 is intentionally isolated in Python 3.11. The main project remains usable without AIM
through its JSONL tracker, including on newer Python interpreters.
