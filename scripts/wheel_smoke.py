"""Exercise every installed CLI from an empty directory, without source paths.

Run using the Python interpreter of an environment containing the built wheel
and its extras. Each command is isolated, preventing earlier imports from hiding
an accidental dependency on a script directory.
"""

import subprocess
import sys
import tempfile

from iwac_pipeline.cli import ANALYSIS_MODULES, PROCESS_MODULES, UPLOAD_MODULES


def main():
    commands = [("upload_main", [name, "--help"]) for name in UPLOAD_MODULES]
    commands += [("process_main", [name, "--help"]) for name in PROCESS_MODULES]
    commands += [("analyze_main", [name, "--help"]) for name in ANALYSIS_MODULES]
    commands += [("mirror_main", ["--help"]), ("publish_public_main", ["--help"])]
    with tempfile.TemporaryDirectory(prefix="iwac-wheel-") as directory:
        for function, arguments in commands:
            code = f"from iwac_pipeline.cli import {function}; raise SystemExit({function}({arguments!r}))"
            result = subprocess.run(
                [sys.executable, "-I", "-c", code], cwd=directory,
                capture_output=True, text=True, timeout=90,
            )
            if result.returncode:
                raise SystemExit(f"{function} {arguments}:\n{result.stdout}\n{result.stderr}")
            print(f"PASS {function} {' '.join(arguments)}")
    print(f"Validated {len(commands)} installed command routes")


if __name__ == "__main__":
    main()
