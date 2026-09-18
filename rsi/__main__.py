"""python -m rsi {init,status,dry-run,run,stop,confirm-plan}."""

import argparse
import json
import signal
from pathlib import Path

from rsi.core import atomic
from rsi.runner import Runner, dry_run, lock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["init", "status", "dry-run", "run", "stop", "confirm-plan"]
    )
    parser.add_argument("--state", type=Path, default=Path(".rsi"))
    parser.add_argument("--config", type=Path, default=Path("rsi/config.json"))
    parser.add_argument("--synthetic", "--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prior-study", type=Path, help="completed v2/v3 history for fresh init")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    if args.command == "dry-run" or (args.command == "run" and args.synthetic):
        print(json.dumps(dry_run(repo), indent=2))
        return
    runner = Runner(repo, args.state, args.synthetic)
    if args.command == "stop":
        atomic(args.state / "STOP", {"reason": "operator requested stop"})
    elif args.command == "status":
        print(json.dumps(runner.status(), indent=2))
    elif args.command == "confirm-plan":
        from rsi.confirmation import plan

        print(json.dumps(plan(runner), indent=2))
    else:
        with lock(runner.state):
            if args.command == "init":
                runner.initialize(args.config, args.prior_study)
            else:

                def request_stop(signum, frame):
                    atomic(runner.stop, {"signal": signum})

                signal.signal(signal.SIGTERM, request_stop)
                signal.signal(signal.SIGINT, request_stop)
                try:
                    runner.run(args.resume)
                except InterruptedError as exc:
                    print(str(exc))
        print(json.dumps(runner.status(), indent=2))


if __name__ == "__main__":
    main()
