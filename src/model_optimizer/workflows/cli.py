"""Workflow CLI."""

from __future__ import annotations

import argparse

from .manifest import WorkflowManifest
from .runner import WorkflowRunner


def workflow_cli(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a workflow manifest")
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="Execute a workflow manifest")
    run_p.add_argument("--manifest", required=True, help="Workflow manifest path (.json/.yaml)")
    run_p.add_argument("--dry-run", action="store_true", help="Plan only, do not execute steps")

    plan_p = sub.add_parser("plan", help="Print commands without executing")
    plan_p.add_argument("--manifest", required=True, help="Workflow manifest path (.json/.yaml)")

    parsed = parser.parse_args(args or [])
    if parsed.command is None:
        parser.print_help()
        return

    manifest = WorkflowManifest.load(parsed.manifest)
    if parsed.command == "plan" or getattr(parsed, "dry_run", False):
        manifest.dry_run = True

    runner = WorkflowRunner(manifest)
    result = runner.run() if parsed.command == "run" else runner.plan()
    for cmd in result.commands:
        print(f"[workflow] {cmd.stage}.{cmd.action}: {' '.join(cmd.argv[1:])}")
