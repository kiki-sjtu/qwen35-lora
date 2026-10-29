"""Use the existing project Python when a training entry is run from an IDE/CLI."""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def ensure_project_python(script_file):
    environment = PROJECT_ROOT / '.venv'
    interpreter = environment / ('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    if Path(sys.prefix).resolve()==environment.resolve():
        return
    if not interpreter.is_file():
        raise RuntimeError('Project environment missing. First run: python code/setup_env.py')
    os.execv(str(interpreter),[str(interpreter),str(Path(script_file).resolve()),*sys.argv[1:]])
