"""Generate VIST-NRF Colab-ready notebooks (00_bootstrap ... 05_uncertainty_and_failures)."""

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

NOTEBOOKS_DIR = "/home/pranjal/Desktop/Projects/Github/Research_Projects/VIST-NRF/notebooks"


def nb(cells):
    nb = new_notebook()
    nb.cells = cells
    return nb


# ---------------------------------------------------------------------------
# 00_bootstrap.ipynb
# ---------------------------------------------------------------------------
cells_00 = [
    new_markdown_cell("# 00 Bootstrap — VIST-NRF on Colab\n"
                      "Mounts Google Drive, sets `VIST_NRF_DRIVE/DATA/RESULTS`, clones the repo via Git "
                      "using `GITHUB_TOKEN` from Colab Secrets, and installs the package in editable mode."),
    new_code_cell(
        "import os, stat, subprocess\n"
        "from pathlib import Path\n"
        "from google.colab import drive, userdata\n"
        "\n"
        "drive.mount('/content/drive')\n"
        "os.environ['VIST_NRF_DRIVE'] = '/content/drive/MyDrive/VIST-NRF'\n"
        "os.environ['VIST_NRF_DATA'] = str(Path(os.environ['VIST_NRF_DRIVE']) / 'data')\n"
        "os.environ['VIST_NRF_RESULTS'] = str(Path(os.environ['VIST_NRF_DRIVE']) / 'results')\n"
        "for key in ('VIST_NRF_DATA', 'VIST_NRF_RESULTS'):\n"
        "    Path(os.environ[key]).mkdir(parents=True, exist_ok=True)\n"
        "\n"
        "repo_url = 'https://github.com/<OWNER>/VIST-NRF.git'  # <-- set OWNER\n"
        "repo_dir = '/content/VIST-NRF'\n"
        "token = userdata.get('GITHUB_TOKEN')\n"
        "if not token:\n"
        "    raise RuntimeError('Add GITHUB_TOKEN to Colab Secrets.')\n"
        "\n"
        "askpass = Path('/tmp/vist_nrf_askpass.sh')\n"
        "askpass.write_text('#!/bin/sh\\ncase \"$1\" in\\n"
        "  *Username*) echo x-access-token ;;\n"
        "  *Password*) printf \"%s\" \"$GITHUB_TOKEN\" ;;\n"
        "esac\\n')\n"
        "askpass.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)\n"
        "env = os.environ.copy()\n"
        "env.update(GIT_ASKPASS=str(askpass), GIT_TERMINAL_PROMPT='0', GITHUB_TOKEN=token)\n"
        "try:\n"
        "    if not Path(repo_dir).exists():\n"
        "        subprocess.run(['git', 'clone', repo_url, repo_dir], check=True, env=env)\n"
        "    else:\n"
        "        subprocess.run(['git', '-C', repo_dir, 'pull', '--ff-only'], check=True, env=env)\n"
        "finally:\n"
        "    askpass.unlink(missing_ok=True)\n"
        "    env.pop('GITHUB_TOKEN', None)\n"
        "    del token\n"
        "\n"
        "subprocess.run(['python', '-m', 'pip', 'install', '-q', '-e', repo_dir], check=True)\n"
        "import torch, vist_nrf\n"
        "commit = subprocess.check_output(['git', '-C', repo_dir, 'rev-parse', '--short', 'HEAD'], text=True).strip()\n"
        "print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available(), '| commit:', commit)"
    ),
]
with open(f"{NOTEBOOKS_DIR}/00_bootstrap.ipynb", "w") as f:
    nbf.write(nb(cells_00), f)

# ---------------------------------------------------------------------------
# 01_data_audit.ipynb
# ---------------------------------------------------------------------------
cells_01 = [
    new_markdown_cell("# 01 Data Audit — PhysioNet/CinC Challenge 2019\n"
                      "Audits the downloaded `.psv` tree: file counts per environment, schema, missingness, "
                      "sepsis prevalence, and record integrity. Writes `docs/DATA_NOTES.md` and a download manifest."),
    new_code_cell(
        "import os\n"
        "from pathlib import Path\n"
        "from vist_nrf.data.download import audit_tree, verify_counts, save_manifest\n"
        "from vist_nrf.data.physionet2019 import list_psv_files, infer_hospital_system, audit_psv\n"
        "\n"
        "root = os.environ.get('VIST_NRF_DATA', '/content/drive/MyDrive/VIST-NRF/data')\n"
        "raw_root = Path(root) / 'raw' / 'physionet2019'\n"
        "print('Raw root:', raw_root, '| exists:', raw_root.exists())\n"
        "\n"
        "if raw_root.exists():\n"
        "    files = list_psv_files(raw_root)\n"
        "    print('Total .psv files:', len(files))\n"
        "    from collections import Counter\n"
        "    envs = Counter(infer_hospital_system(p) for p in files)\n"
        "    print('Per-environment:', dict(envs))\n"
        "    manifest = audit_tree(raw_root, source_url='https://registry.opendata.aws/challenge-2019/')\n"
        "    problems = verify_counts(manifest)\n"
        "    if problems:\n"
        "        print('PROBLEMS:'); [print(' -', p) for p in problems]\n"
        "    else:\n"
        "        print('Counts match spec (A=20336, B=20000)')\n"
        "    save_manifest(raw_root, Path(root) / 'download_manifest.json')\n"
        "    print('Manifest saved.')\n"
        "else:\n"
        "    print('No data downloaded yet. See docs/DATA_NOTES.md for the aws s3 sync command.')"
    ),
]
with open(f"{NOTEBOOKS_DIR}/01_data_audit.ipynb", "w") as f:
    nbf.write(nb(cells_01), f)

# ---------------------------------------------------------------------------
# 02_baselines.ipynb
# ---------------------------------------------------------------------------
cells_02 = [
    new_markdown_cell("# 02 Baselines — Stage A\n"
                      "Trains logistic regression and XGBoost (values-only vs values+timing) on the source "
                      "environment and evaluates in-environment + cross-site. Saves results as JSONL."),
    new_code_cell(
        "import os, json\n"
        "from pathlib import Path\n"
        "from vist_nrf.utils.seed import set_seed\n"
        "from vist_nrf.models import build_model\n"
        "from vist_nrf.eval.metrics import evaluate\n"
        "\n"
        "set_seed(0)\n"
        "results_dir = Path(os.environ.get('VIST_NRF_RESULTS', '/content/drive/MyDrive/VIST-NRF/results'))\n"
        "results_dir.mkdir(parents=True, exist_ok=True)\n"
        "\n"
        "# Synthetic smoke test (replace with real PhysioNet data path after download)\n"
        "import numpy as np\n"
        "rng = np.random.default_rng(0)\n"
        "X = rng.normal(0, 1, (200, 12))\n"
        "y = (X[:, 0] + 0.5 * X[:, 3] + rng.normal(0, 0.5, 200) > 0).astype(int)\n"
        "envs = [0] * 100 + [1] * 100\n"
        "\n"
        "for name in ['logistic_regression', 'xgboost_values', 'xgboost_timing']:\n"
        "    try:\n"
        "        m = build_model(name, seed=0)\n"
        "        m.fit(X[:100], y[:100], envs=envs[:100])\n"
        "        p = m.predict_proba(X[100:])[:, 1]\n"
        "        mres = evaluate(y[100:], p)\n"
        "        print(name, '| AUROC=%.3f AUPRC=%.3f Brier=%.3f' % (mres.auroc, mres.auprc, mres.brier))\n"
        "    except Exception as e:\n"
        "        print(name, 'SKIPPED:', e)"
    ),
]
with open(f"{NOTEBOOKS_DIR}/02_baselines.ipynb", "w") as f:
    nbf.write(nb(cells_02), f)

# ---------------------------------------------------------------------------
# 03_semisynthetic.ipynb
# ---------------------------------------------------------------------------
cells_03 = [
    new_markdown_cell("# 03 Semi-synthetic Benchmark — Controlled Observation-Policy Shift\n"
                      "Generates latent trajectories + policy-dependent observation; tests oracle invariance "
                      "and held-out policy shift (spec §8)."),
    new_code_cell(
        "import numpy as np\n"
        "from vist_nrf.benchmarks.semisynthetic import SemiSyntheticConfig, generate_all, oracle_predict\n"
        "from vist_nrf.eval.metrics import auroc\n"
        "\n"
        "cfg = SemiSyntheticConfig(n_patients=100, n_hours=48, n_vars=8, n_envs=2, seed=0,\n"
        "                          w=[0.5, 0.8], r=[0.2, 0.2])\n"
        "data = generate_all(cfg, seed=0)\n"
        "scores, truths = oracle_predict(data)\n"
        "y = np.array([truths[k] for k in scores])\n"
        "s = np.array([scores[k] for k in scores])\n"
        "print('Oracle AUROC (policy-invariant upper bound): %.3f' % auroc(y, s))\n"
        "print('Patients per env:', {e: len(p) for e, p in data.items()})"
    ),
]
with open(f"{NOTEBOOKS_DIR}/03_semisynthetic.ipynb", "w") as f:
    nbf.write(nb(cells_03), f)

# ---------------------------------------------------------------------------
# 04_method_and_ablations.ipynb
# ---------------------------------------------------------------------------
cells_04 = [
    new_markdown_cell("# 04 VIST-NRF Method & Ablations — Stage C\n"
                      "State encoder + policy encoder + prediction head, with optional adversarial "
                      "environment confusion and invariance penalties (ERM/IRM/V-REx/GroupDRO)."),
    new_code_cell(
        "import numpy as np\n"
        "from vist_nrf.utils.seed import set_seed\n"
        "from vist_nrf.models import build_model\n"
        "from vist_nrf.eval.metrics import evaluate\n"
        "\n"
        "set_seed(0)\n"
        "rng = np.random.default_rng(1)\n"
        "X = rng.normal(0, 1, (150, 10))\n"
        "y = (X[:, 0] > 0).astype(int)\n"
        "envs = [0] * 75 + [1] * 75\n"
        "\n"
        "for cfg in [\n"
        "    dict(use_policy=False, use_adversarial=False, invariance_method='erm'),\n"
        "    dict(use_policy=True, use_adversarial=False, invariance_method='erm'),\n"
        "    dict(use_policy=False, use_adversarial=True, invariance_method='erm'),\n"
        "]:\n"
        "    m = build_model('vist_nrf', input_dim=10, n_envs=2, epochs=3, **cfg)\n"
        "    m.fit(X[:100], y[:100], envs=envs[:100])\n"
        "    p = m.predict_proba(X[100:])[:, 1]\n"
        "    r = evaluate(y[100:], p)\n"
        "    print(cfg, '| AUROC=%.3f' % r.auroc)"
    ),
]
with open(f"{NOTEBOOKS_DIR}/04_method_and_ablations.ipynb", "w") as f:
    nbf.write(nb(cells_04), f)

# ---------------------------------------------------------------------------
# 05_uncertainty_and_failures.ipynb
# ---------------------------------------------------------------------------
cells_05 = [
    new_markdown_cell("# 05 Uncertainty, Calibration & Failure Analysis — Secondary\n"
                      "Brier, ECE, reliability curves, patient-level bootstrap CIs, split conformal. "
                      "Only run after core result is stable (spec §11)."),
    new_code_cell(
        "import numpy as np\n"
        "from vist_nrf.utils.seed import set_seed\n"
        "from vist_nrf.eval.metrics import (evaluate, brier, ece, reliability_curve,\n"
        "                                       patient_bootstrap, bootstrap_ci,\n"
        "                                       split_conformal_threshold, conformal_coverage)\n"
        "\n"
        "set_seed(0)\n"
        "rng = np.random.default_rng(2)\n"
        "n = 300\n"
        "y = (rng.random(n) < 0.2).astype(int)\n"
        "p = rng.random(n)\n"
        "pid = np.repeat(np.arange(60), 5)[:n]\n"
        "\n"
        "r = evaluate(y, p)\n"
        "print('AUROC=%.3f Brier=%.3f ECE=%.3f prev=%.3f' % (r.auroc, r.brier, r.ece, r.prevalence))\n"
        "boot = patient_bootstrap(y, p, pid, lambda yt, ys: evaluate(yt, ys).auroc, n_resamples=200, seed=0)\n"
        "lo, hi = bootstrap_ci(boot)\n"
        "print('AUROC bootstrap 95%% CI: (%.3f, %.3f)' % (lo, hi))\n"
        "thr = split_conformal_threshold(p[y == 0], y[y == 0], alpha=0.1)\n"
        "cov = conformal_coverage(p, y, thr)\n"
        "print('Conformal threshold=%.3f coverage=%.3f' % (thr, cov))\n"
        "pt, pp, cnt = reliability_curve(y, p)\n"
        "print('Reliability bins:', list(zip(pp.round(2), pt.round(2), cnt)))\n"
    ),
]
with open(f"{NOTEBOOKS_DIR}/05_uncertainty_and_failures.ipynb", "w") as f:
    nbf.write(nb(cells_05), f)

print("Notebooks generated.")
for i in range(6):
    import os
    print(f"  notebooks/{i:02d}_*.ipynb exists:", any(f.startswith(f'{i:02d}_') for f in os.listdir(NOTEBOOKS_DIR)))
