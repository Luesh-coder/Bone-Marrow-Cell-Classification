"""
Process exit for the command-line scripts.

With torch 2.14 + Python 3.15 on Windows, a process that has run CUDA
convolutions (with or without cuDNN) segfaults during normal interpreter
shutdown: exit code 139 after all work and files are done. Verified with a
3-line script; it is a teardown bug in that torch build, not in this project.

run_and_exit(main) runs main(), flushes output, and ends the process with
os._exit, skipping the crashing teardown. Exit codes are meaningful again:
0 = success, 1 = error (traceback printed), 130 = Ctrl+C, or SystemExit's code.
Everything the scripts write (checkpoints, CSV/JSON, plots, Optuna's SQLite
rows) is written before main() returns, so nothing is lost.

Light module (no torch): imported at the top of entry scripts.
"""

import os
import sys
import traceback


def run_and_exit(main):
    code = 0
    try:
        main()
    except SystemExit as exc:
        if exc.code is None or isinstance(exc.code, int):
            code = exc.code or 0
        else:                       # SystemExit("message"): print it, exit 1 (like Python)
            print(exc.code, file=sys.stderr)
            code = 1
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        code = 130
    except BaseException:
        traceback.print_exc()
        code = 1
    finally:
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except Exception:
                pass
    os._exit(code)
