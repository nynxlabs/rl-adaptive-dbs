"""Allow ``python -m rl_adaptive_dbs.tui`` (used by ``--dev`` child processes)."""

from rl_adaptive_dbs.tui import main

raise SystemExit(main())
