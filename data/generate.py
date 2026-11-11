"""Generate a synthetic incident corpus for Incident Copilot.

Everything here is invented. No real company, customer or employer data.

Deterministic: a fixed seed means the same corpus every run, so evaluation
numbers are comparable across experiments.

Outputs (written next to this file):
  incidents.json  - 45 resolved past incidents (the corpus we index)
  runbooks.json   - 10 runbooks (also indexed)
  eval_set.json   - 15 held-out incidents with known labels (never indexed)

The held-out split matters: if we indexed the evaluation incidents, retrieval
would find the exact answer and every score would be meaningless.
"""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path

SEED = 20261110
TOTAL = 60
EVAL_SIZE = 15

SERVICES = [
    "checkout-api", "payments-worker", "auth-service", "search-indexer",
    "notification-dispatch", "user-profile-db", "cart-cache", "reporting-etl",
]

# Each root cause is a label. The agent's job is to predict it, and in week 3
# these labels become the ground truth for accuracy scoring.
CAUSES = {
    "connection_pool_exhaustion": {
        "runbook": "RB-01",
        "symptoms": [
            "requests queue then time out while CPU stays flat",
            "threads pile up waiting on the database, server load is normal",
            "p99 latency spikes to 30s but the database reports almost no activity",
            "intermittent timeouts under load that clear when traffic drops",
        ],
        "logs": [
            "HikariPool-1 - Connection is not available, request timed out after 30000ms",
            "WARN  pool size 20 reached, 147 threads awaiting connection",
            "java.sql.SQLTransientConnectionException: timeout acquiring connection",
            "ERROR  active=20 idle=0 waiting=203 pool=primary",
            "INFO  database CPU 6%, active queries 3",
        ],
        "fix": "raised pool size and added a 5s acquisition timeout with retry",
    },
    "memory_leak": {
        "runbook": "RB-02",
        "symptoms": [
            "latency climbs over hours and pods restart on OOM",
            "resident memory grows steadily until the container is killed",
            "the service degrades through the day and recovers after a restart",
            "heap usage never returns to baseline after traffic peaks",
        ],
        "logs": [
            "Killed process 1 (python) total-vm:8216404kB, anon-rss:7901232kB",
            "WARN  GC overhead limit approaching: 94% heap after full GC",
            "kubelet  Container app OOMKilled, restarting (restartCount=7)",
            "INFO  rss 1.2GB -> 6.8GB over 5h, request rate unchanged",
            "ERROR  MemoryError during response serialisation",
        ],
        "fix": "fixed an unbounded in-memory cache that never evicted entries",
    },
    "bad_deploy": {
        "runbook": "RB-03",
        "symptoms": [
            "error rate jumps within minutes of a release",
            "5xx responses begin immediately after a rollout completes",
            "a new exception appears in volume with no change in traffic",
            "failures start at the exact timestamp of a version bump",
        ],
        "logs": [
            "ERROR  AttributeError: 'NoneType' object has no attribute 'tenant_id'",
            "INFO   rollout 4f2a1c completed, 100% traffic shifted",
            "ERROR  5xx rate 0.4% -> 31.8% over 4 minutes",
            "ERROR  TypeError: missing 1 required positional argument: 'region'",
            "INFO   image tag app:2026.11.3 replaced app:2026.11.2",
        ],
        "fix": "rolled back the release and added a canary gate on 5xx rate",
    },
    "expired_certificate": {
        "runbook": "RB-04",
        "symptoms": [
            "all outbound TLS calls fail at once with no code change",
            "every dependency becomes unreachable simultaneously overnight",
            "handshakes are rejected although the network path is healthy",
            "a service that was untouched for weeks stops authenticating",
        ],
        "logs": [
            "ERROR  javax.net.ssl.SSLHandshakeException: PKIX path validation failed",
            "ERROR  certificate expired on 2026-03-14T00:00:00Z",
            "WARN   retry 3/3 failed, circuit opened for upstream billing-gw",
            "ERROR  x509: certificate has expired or is not yet valid",
            "INFO   tcp connect to billing-gw:443 succeeded, tls negotiation failed",
        ],
        "fix": "rotated the certificate and added a 30-day expiry alert",
    },
    "db_lock_contention": {
        "runbook": "RB-05",
        "symptoms": [
            "writes stall while reads stay healthy",
            "update statements hang for a minute then fail, selects are instant",
            "a queue of blocked transactions builds up behind one long query",
            "order submissions time out but the catalogue loads fine",
        ],
        "logs": [
            "ERROR  deadlock detected, process 4471 waits for ShareLock on transaction",
            "WARN   lock_waits 312 in last minute, longest 48s",
            "INFO   autovacuum skipped: relation orders is locked",
            "ERROR  canceling statement due to lock timeout",
            "INFO   blocking pid 8812 idle in transaction for 4m12s",
        ],
        "fix": "shortened the transaction and reordered writes to a consistent key order",
    },
    "upstream_timeout": {
        "runbook": "RB-06",
        "symptoms": [
            "our service is healthy but a dependency is slow",
            "requests that call one partner API hang, everything else is fast",
            "our own metrics look normal while end-to-end latency is terrible",
            "thread pools fill up waiting on a third-party response",
        ],
        "logs": [
            "ERROR  upstream inventory-svc responded in 9821ms, exceeds 3000ms budget",
            "WARN   bulkhead rejected 214 calls, queue full",
            "INFO   our own p50 unchanged at 42ms",
            "ERROR  read timed out after 10000ms calling inventory-svc/v2/stock",
            "WARN   dependency error budget consumed 140% this hour",
        ],
        "fix": "added a timeout, a circuit breaker and a cached fallback response",
    },
    "disk_full": {
        "runbook": "RB-07",
        "symptoms": [
            "writes fail and the process cannot log",
            "the service refuses new data and health checks go silent",
            "file operations fail with no space errors across one node",
            "a node stops accepting writes while its peers are fine",
        ],
        "logs": [
            "ERROR  OSError: [Errno 28] No space left on device",
            "WARN   /var/log at 100% (48G/48G), oldest file 94 days",
            "ERROR  failed to flush WAL segment, shutting down",
            "ERROR  could not write to temporary file: No space left on device",
            "WARN   inode usage 99.4% on /dev/nvme0n1p1",
        ],
        "fix": "added log rotation with a retention limit and a disk-usage alert",
    },
    "cache_stampede": {
        "runbook": "RB-08",
        "symptoms": [
            "a cache expiry causes a burst of identical database queries",
            "thousands of requests recompute the same value at the same moment",
            "a traffic cliff in the cache turns into a flood on the database",
            "periodic spikes every time a batch of keys ages out together",
        ],
        "logs": [
            "WARN   cache miss rate 2% -> 98% in 10s for key prefix catalog:",
            "ERROR  database connections 18 -> 400, rejecting new queries",
            "INFO   TTL batch expiry: 240000 keys evicted simultaneously",
            "WARN   identical query executed 3812 times in 5s",
            "ERROR  redis GET catalog:home returned nil for 94% of requests",
        ],
        "fix": "added jittered TTLs and a single-flight lock on cache refill",
    },
    "config_drift": {
        "runbook": "RB-09",
        "symptoms": [
            "one environment misbehaves while identical code works elsewhere",
            "the same build passes in staging and fails in production",
            "behaviour differs per region with no corresponding code change",
            "a feature behaves as though it were disabled in only one cluster",
        ],
        "logs": [
            "ERROR  feature flag async_settlement=false in prod, true in staging",
            "WARN   config checksum mismatch: expected 9ab12f, got 41cc08",
            "ERROR  unknown region value 'us-east-1a' for parameter queue_region",
            "WARN   env var SETTLEMENT_MODE unset, falling back to 'legacy'",
            "INFO   effective config differs from repository at 3 keys",
        ],
        "fix": "moved the value into version-controlled config with a drift check in CI",
    },
    "rate_limit_exceeded": {
        "runbook": "RB-10",
        "symptoms": [
            "a third-party API starts rejecting our calls",
            "a provider returns refusals once our call volume rises",
            "requests succeed early in the hour and fail near the end of it",
            "our retries make a partner outage worse instead of better",
        ],
        "logs": [
            "ERROR  429 Too Many Requests from provider api.payments-gw.test",
            "WARN   quota 10000/hour exhausted at 14:22, resets 15:00",
            "ERROR  retry storm detected: 3400 retries in 60s",
            "WARN   provider header x-ratelimit-remaining: 0",
            "ERROR  backoff disabled, 12 immediate retries per failed call",
        ],
        "fix": "added token-bucket throttling and exponential backoff with jitter",
    },
}

RUNBOOK_TEXT = {
    "RB-01": ("Connection pool exhaustion",
              "Check pool metrics for threads awaiting connection. Confirm the "
              "database itself is healthy and accepting connections. If the pool "
              "is saturated but the database is idle, the leak is in the "
              "application: look for connections acquired outside a context "
              "manager. Raise the pool only after the leak is ruled out."),
    "RB-02": ("Memory leak and OOMKill",
              "Compare heap or RSS growth against request volume. A flat request "
              "rate with climbing memory points at retained objects, usually an "
              "unbounded cache or an accumulating list. Capture a heap dump "
              "before the restart, since the evidence is lost afterwards."),
    "RB-03": ("Bad deploy",
              "Correlate the error-rate change with the rollout timestamp. If "
              "they line up within minutes, roll back first and investigate "
              "afterwards. Do not attempt a forward fix during the incident "
              "unless rollback is impossible."),
    "RB-04": ("Expired TLS certificate",
              "A sudden total failure of outbound calls with no deployment is "
              "almost always credentials or certificates. Check the expiry date "
              "on the presented chain. Rotation is the fix; the follow-up is an "
              "expiry alert at 30 days."),
    "RB-05": ("Database lock contention",
              "Inspect current locks and the longest waiting transaction. Writes "
              "stalling while reads are fine indicates row or table locks. Look "
              "for long transactions holding locks across external calls."),
    "RB-06": ("Upstream timeout",
              "Compare our own latency with the dependency's. If ours is normal "
              "and theirs is slow, the fix is isolation, not optimisation: "
              "timeout, circuit breaker, and a degraded fallback."),
    "RB-07": ("Disk full",
              "Identify the largest directories and the oldest files. Logs and "
              "temporary files are the usual cause. Free space first to restore "
              "service, then add rotation and retention so it cannot recur."),
    "RB-08": ("Cache stampede",
              "A miss-rate spike followed by a database connection spike for one "
              "key prefix indicates simultaneous expiry. Fix with jittered TTLs "
              "and single-flight refill so only one request repopulates a key."),
    "RB-09": ("Configuration drift",
              "When identical code behaves differently per environment, diff the "
              "effective configuration rather than reading the code. Treat the "
              "repository as the source of truth and detect drift in CI."),
    "RB-10": ("Third-party rate limit",
              "A 429 response means our call rate, not our correctness, is the "
              "problem. Check quota windows and whether retries are amplifying "
              "load. Throttle on our side and back off with jitter."),
}

IMPACTS = [
    "checkout failures for a subset of customers",
    "delayed order confirmations",
    "elevated error rate on the mobile app",
    "partial outage for one region",
    "background jobs backing up",
    "slow page loads on the account dashboard",
]


def build(rng):
    """Build TOTAL incidents, cycling through causes so every label is covered."""
    cause_names = list(CAUSES)
    incidents = []
    start = datetime(2026, 1, 6, 9, 0)

    for i in range(TOTAL):
        cause = cause_names[i % len(cause_names)]
        spec = CAUSES[cause]
        service = rng.choice(SERVICES)
        severity = rng.choice(["P1", "P2", "P2", "P3", "P3", "P4"])
        opened = start + timedelta(days=i * 4, hours=rng.randint(0, 10))
        minutes = {"P1": 95, "P2": 140, "P3": 260, "P4": 480}[severity]

        # Each incident gets its own phrasing and its own subset of log lines,
        # so two incidents with the same root cause are not near-duplicates.
        symptom = rng.choice(spec["symptoms"])
        logs = rng.sample(spec["logs"], 3)

        incidents.append({
            "id": f"INC-{1001 + i}",
            "service": service,
            "severity": severity,
            "opened_at": opened.isoformat(timespec="seconds"),
            "title": f"{service}: {symptom}",
            "impact": rng.choice(IMPACTS),
            "symptoms": symptom,
            "log_excerpt": "\n".join(logs),
            "root_cause_category": cause,
            "resolution": spec["fix"],
            "runbook_id": spec["runbook"],
            "time_to_resolve_minutes": minutes + rng.randint(-25, 60),
        })
    return incidents


def main():
    rng = random.Random(SEED)
    incidents = build(rng)
    rng.shuffle(incidents)

    # Held-out split. Evaluation incidents keep their labels but are never
    # indexed, so retrieval has to generalise from *other* incidents.
    eval_set = incidents[:EVAL_SIZE]
    corpus = incidents[EVAL_SIZE:]

    runbooks = [
        {
            "id": rb_id,
            "title": title,
            "root_cause_category": cause,
            "body": body,
        }
        for cause, spec in CAUSES.items()
        for rb_id, (title, body) in [(spec["runbook"], RUNBOOK_TEXT[spec["runbook"]])]
    ]

    out = Path(__file__).parent
    for name, payload in [
        ("incidents.json", corpus),
        ("runbooks.json", runbooks),
        ("eval_set.json", eval_set),
    ]:
        (out / name).write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {name}: {len(payload)} records")

    labels = sorted({i["root_cause_category"] for i in corpus})
    print(f"\ncorpus labels ({len(labels)}): {', '.join(labels)}")
    print(f"held-out eval incidents: {[i['id'] for i in eval_set]}")


if __name__ == "__main__":
    main()
