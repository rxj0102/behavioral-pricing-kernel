# CBOE snapshot collector

Status: written and tested with an injected fetch function. NOT enabled, NOT scheduled. It has never contacted CBOE, because the build sandbox could not reach the host.

## What it does

scripts/collect_cboe_snapshot.py takes one snapshot of the free CBOE delayed SPX chain and stores the raw JSON in a private directory outside the repository, with a manifest line per snapshot. Parsing into the panel schema is a separate later step, so the raw record is never altered.

## Safety rails (all tested)

- Off by default: it refuses to run unless BPK_CBOE_TERMS_ACK equals the acknowledgement string, which you set only after reading the CBOE terms of use.
- Raw path: BPK_CBOE_RAW_DIR must be an absolute path outside the repository and outside any git work tree. The directory is created with mode 0700 and files with 0600. A directory open to group or others is rejected.
- Manifest: raw_dir/manifest.jsonl has file, UTC time, bytes, SHA-256, option count and schema hash for each snapshot.
- Loud failure on schema change: if the snapshot fails the checker (src/bpk/data/cboe_schema.py) or differs from the frozen reference, the snapshot goes to raw_dir/quarantine, an ALERT_SCHEMA_CHANGE file is written, failures.jsonl gets a record, the script prints to stderr and exits with code 2. Other exit codes: 3 not enabled, 4 fetch failure, 5 configuration, 1 other. Nothing is parsed or appended to the manifest as successful.
- The frozen reference (tests/fixtures/cboe_schema.json) is produced by scripts/freeze_cboe_fixture.py from one snapshot that a person has inspected. Without it the collector refuses to run, so it cannot be enabled before the layout was checked by a human.

## Enabling (all steps are yours; I will not do them)

1. Open the network domains cdn.cboe.com and www.cboe.com for the environment where the collector runs.
2. Read the CBOE terms of use (python scripts/freeze_cboe_fixture.py --terms saves the legal pages locally to read). Decide whether collection and any use in the paper is allowed. Record your finding in docs/data.md.
3. Run python scripts/freeze_cboe_fixture.py and inspect tests/fixtures/cboe_schema.json against the raw snapshot.
4. Choose a private raw directory outside the repo and set BPK_CBOE_RAW_DIR and BPK_CBOE_TERMS_ACK in the scheduler environment.
5. Install one of the examples in deploy/ (cron or systemd user timer). They are inert examples until you copy and edit them. The schedule is weekdays 16:30 New York time.
6. Check raw_dir for ALERT_SCHEMA_CHANGE and failures.jsonl regularly, or alert on a nonzero exit of the job.

A scheduler on a normal always-on machine is needed: ephemeral cloud sessions cannot collect daily, and a public GitHub Actions run would write the raw data to a place that is not private, which conflicts with the rules above.

## Limits

- The sample starts the day collection starts. It cannot be back-filled and it will not contain 2008 or 2020.
- The endpoint is undocumented and may change or disappear without notice; the schema check exists for that reason.
