# Backup recovery runbook — synthetic evaluation fixture
Database snapshots are taken every six hours. Snapshot retention is 35 days.
Restore requests require approval from the on-call database operator.
The recovery time objective is four hours; it is an objective rather than a contractual guarantee.
Restore drills run quarterly in an isolated staging account. Backups are encrypted with AES-256.
