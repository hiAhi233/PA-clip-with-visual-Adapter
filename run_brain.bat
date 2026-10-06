@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

REM ================================================================
REM  Brain MRI few-shot: one-click run (Windows + conda torch-gpu1)
REM
REM  IMPORTANT: keep this file ASCII-only AND CRLF line endings.
REM    - Chinese text in a .bat is parsed with the system ANSI
REM      codepage (GBK) and cmd mangles keywords (setlocal -> 'local').
REM    - LF-only endings cause the same breakage ("%PY%" -> 'ython.exe"').
REM    Chinese log output from python is fine (console is chcp 65001).
REM
REM  Steps: [1] check env -> [2] build brain_mri/ data dirs
REM         -> [3] regression tests -> [4] CUDA smoke -> [5] K-shot
REM         train + test eval + heatmaps -> print result summary
REM
REM  Runtime on RTX 5060 Laptop 8G: tests ~10s, smoke 1-2 min,
REM  K=5 / 2700 steps / TVJ about 40-60 min.
REM ================================================================

REM ---------------- knobs ----------------
set "PY=C:\Users\HP\.conda\envs\torch-gpu1\python.exe"
set "BRAIN_SRC=D:\图神经网络\小样本原型学习\AA-CLIP\data\MedAD\Brain_AD"

set "K=5"
set "SEED=0"
set "GROUP=TVJ"
set "STEPS=2700"
set "BATCH=16"
set "VAL_CASES=2"

REM STRICT=1: drop cases that also appear in test and enable the case-id
REM           regex (case-level clean, but the abnormal pool becomes the
REM           4 slices of patient 00803 only).
REM STRICT=0: follow the original BMAD split; results record
REM           case_disjoint_verified=false.
set "STRICT=0"

REM FRESH=1: rebuild data dirs and add --fresh (new artifact dir, never
REM          overwrites an existing checkpoint).
set "FRESH=0"

REM SKIP_TESTS=1: skip step 3.
set "SKIP_TESTS=0"
REM ----------------------------------------

set "HF_HUB_OFFLINE=1"
set "PYTHONUNBUFFERED=1"
REM Do NOT set PYTHONUTF8=1 here: UTF-8 mode makes site.py decode .pth
REM files as UTF-8 and this conda env has a non-UTF-8 .pth -> crash.
set "PYTHONIOENCODING=utf-8"

set "FRESH_ARG="
if "%FRESH%"=="1" set "FRESH_ARG=--fresh"

echo ================================================================
echo  Brain MRI few-shot: K=%K% seed=%SEED% group=%GROUP% steps=%STEPS% batch=%BATCH%
echo  Project dir: %CD%
echo ================================================================

echo.
echo [1/5] Check python and deps
if not exist "%PY%" (
    echo   [ERROR] not found: %PY%
    echo   Run "where python" inside conda env torch-gpu1 and update PY in this file.
    goto :fail
)
"%PY%" -c "import torch, open_clip; print('  torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
if errorlevel 1 goto :fail

echo.
echo [2/5] Prepare brain_mri/ data dirs
if "%FRESH%"=="0" if exist "brain_mri\train\normal" if exist "brain_mri\test\abnormal" if exist "brain_mri_masks\test\abnormal" (
    echo   data dirs already exist, skipping copy ^(set FRESH=1 to rebuild^)
    goto :after_prep
)
if not exist "%BRAIN_SRC%" (
    echo   [ERROR] data source not found: %BRAIN_SRC%
    goto :fail
)
if "%STRICT%"=="1" (
    "%PY%" prepare_brain_mri_slices.py --src "%BRAIN_SRC%" --val-cases %VAL_CASES% --strict-cases
) else (
    "%PY%" prepare_brain_mri_slices.py --src "%BRAIN_SRC%" --val-cases %VAL_CASES%
)
if errorlevel 1 goto :fail
:after_prep

echo.
echo [3/5] Regression tests (logic only, no training)
if "%SKIP_TESTS%"=="1" (
    echo   skipped by SKIP_TESTS=1
    goto :after_tests
)
"%PY%" -m unittest tests.test_brain_fewshot
if errorlevel 1 goto :fail
:after_tests

echo.
echo [4/5] CUDA smoke: real BiomedCLIP, synthetic images, 2 steps
"%PY%" _smoke_brain_fewshot.py
if errorlevel 1 goto :fail

echo.
echo [5/5] Few-shot train + test eval (K=%K%, %STEPS% steps)
echo   support pool: brain_mri\train / eval: brain_mri\test ^(640 normal + 3075 abnormal^)
echo   artifacts  : runs\brain_mri\
if "%STRICT%"=="1" (
    "%PY%" brain_fewshot_run.py --data-root brain_mri/train --mask-root brain_mri_masks/train --val-data-root brain_mri/val --val-mask-root brain_mri_masks/val --eval-data-root brain_mri/test --eval-mask-root brain_mri_masks/test --case-id-regex "([0-9]+)_" --data-format slice --ks %K% --seeds %SEED% --groups %GROUP% --steps %STEPS% --batch-size %BATCH% --prompt-set brain_mri_sentence --out-root runs/brain_mri --results runs/brain_mri/results.jsonl %FRESH_ARG%
) else (
    "%PY%" brain_fewshot_run.py --data-root brain_mri/train --mask-root brain_mri_masks/train --val-data-root brain_mri/val --val-mask-root brain_mri_masks/val --eval-data-root brain_mri/test --eval-mask-root brain_mri_masks/test --data-format slice --ks %K% --seeds %SEED% --groups %GROUP% --steps %STEPS% --batch-size %BATCH% --prompt-set brain_mri_sentence --out-root runs/brain_mri --results runs/brain_mri/results.jsonl %FRESH_ARG%
)
if errorlevel 1 goto :fail

echo.
echo ================================================================
echo  Done. Latest record:
"%PY%" -c "import json,os;rows=[json.loads(l) for l in open(os.path.join('runs','brain_mri','results.jsonl'),encoding='utf-8') if l.strip()];r=rows[-1];print('  group',r['group'],'K',r['k'],'seed',r['seed'],'stage',r['train_stage']);print('  support normal=%s abnormal=%s'%(r['normal_count'],r['abnormal_count']));print('  image AUROC=%s AP=%s F1=%s'%(r['auroc'],r['ap'],r['f1']));print('  pixel Dice=%s IoU=%s'%(r['dice'],r['iou']));print('  eval split',r['evaluation_split'],'status',r['status']);print('  ckpt',r['checkpoint'])"
echo   metrics : runs\brain_mri\^<group^>_k%K%_s%SEED%_*\metrics.json
echo   heatmaps: same run dir under heatmaps\ ^(up to 32 png + raw npy^)
echo ================================================================
endlocal
exit /b 0

:fail
echo.
echo ================================================================
echo  [FAILED] last step returned non-zero. See output above.
echo  Tips: if the data dirs look wrong, delete brain_mri\ and
echo        brain_mri_masks\ and rerun; after an interruption just
echo        rerun this script, finished runs are reused automatically.
echo ================================================================
endlocal
exit /b 1
