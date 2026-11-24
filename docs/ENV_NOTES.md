# Environment Notes — VIST-NRF

## Colab / Drive topology
- VIST_NRF_DRIVE defaults to `/content/drive/MyDrive/VIST-NRF`
- VIST_NRF_DATA defaults to `$VIST_NRF_DRIVE/data`
- VIST_NRF_RESULTS defaults to `$VIST_NRF_DRIVE/results`
- Code: cloned via Git into Colab; install `pip install -e <repo>`.
- Persistent artifacts (raw data, checkpoints, results, figures): Google Drive.
- Never embed GitHub tokens in repo URLs; use Colab Secrets `GITHUB_TOKEN` + `GIT_ASKPASS`.

## Colab bootstrap (skeleton — set OWNER before use)

```python
import os, stat, subprocess
from pathlib import Path
from google.colab import drive, userdata

drive.mount('/content/drive')
os.environ['VIST_NRF_DRIVE'] = '/content/drive/MyDrive/VIST-NRF'
os.environ['VIST_NRF_DATA'] = str(Path(os.environ['VIST_NRF_DRIVE']) / 'data')
os.environ['VIST_NRF_RESULTS'] = str(Path(os.environ['VIST_NRF_RESULTS']) / 'results')
for key in ('VIST_NRF_DATA', 'VIST_NRF_RESULTS'):
    Path(os.environ[key]).mkdir(parents=True, exist_ok=True)

repo_url = 'https://github.com/<OWNER>/VIST-NRF.git'  # replace OWNER
repo_dir = '/content/VIST-NRF'
token = userdata.get('GITHUB_TOKEN')
if not token:
    raise RuntimeError('Add GITHUB_TOKEN to Colab Secrets.')

askpass = Path('/tmp/vist_nrf_askpass.sh')
askpass.write_text('#!/bin/sh\ncase "$1" in\n*Username*) echo x-access-token ;;\n*Password*) printf "%s" "$GITHUB_TOKEN" ;;\nesac\n')
askpass.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
env = os.environ.copy()
env.update(GIT_ASKPASS=str(askpass), GIT_TERMINAL_PROMPT='0', GITHUB_TOKEN=token)
try:
    if not Path(repo_dir).exists():
        subprocess.run(['git', 'clone', repo_url, repo_dir], check=True, env=env)
    else:
        subprocess.run(['git', '-C', repo_dir, 'pull', '--ff-only'], check=True, env=env)
finally:
    askpass.unlink(missing_ok=True)
    env.pop('GITHUB_TOKEN', None)
    del token

subprocess.run(['python', '-m', 'pip', 'install', '-q', '-e', repo_dir], check=True)
import torch, vist_nrf
commit = subprocess.check_output(['git', '-C', repo_dir, 'rev-parse', '--short', 'HEAD'], text=True).strip()
print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available(), '| commit:', commit)
```

## Local (this machine)
- Python 3.14.8, torch CPU available; xgboost not installed locally (optional).
- Editable install: `pip install -e .` then `pip install -e ".[dev]"`.

## Status
UNVERIFIED — Colab bootstrap not yet executed on this project.
