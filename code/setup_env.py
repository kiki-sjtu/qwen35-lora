"""Prepare or check the project environment using Python only."""
import argparse
import os
import subprocess
import sys
import venv
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_ROOT.parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only',action='store_true',help='Only check existing environment; do not install')
    args=parser.parse_args()
    environment=PROJECT_ROOT/'.venv'
    interpreter=environment/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    if not interpreter.exists():
        if args.check_only: raise FileNotFoundError(interpreter)
        subprocess.run([sys.executable,'-c','import torch; assert torch.cuda.is_available(), "CUDA PyTorch is required in the base Python"'],check=True)
        venv.EnvBuilder(with_pip=True,system_site_packages=True).create(environment)
    if not args.check_only:
        subprocess.run([str(interpreter),'-m','pip','install','-r',str(CODE_ROOT/'requirements.txt')],check=True)
    subprocess.run([str(interpreter),'-X','utf8',str(CODE_ROOT/'check_env.py')],check=True)


if __name__=='__main__': main()
