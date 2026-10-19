$ErrorActionPreference = "Stop"

$python = Join-Path $env:LOCALAPPDATA "Python\pythoncore-3.12-64\python.exe"
$environment = Join-Path $PSScriptRoot ".venv"
$environmentPython = Join-Path $environment "Scripts\python.exe"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Python 3.12 x64 was not found at $python. Install it before running this script."
}

if (-not (Test-Path -LiteralPath $environmentPython)) {
    & $python -m venv $environment
    if ($LASTEXITCODE -ne 0) {
        throw "Could not create the virtual environment."
    }
}

& $environmentPython -c "import struct, sys; raise SystemExit(0 if struct.calcsize('P') == 8 and sys.version_info[:2] == (3, 12) else 1)"
if ($LASTEXITCODE -ne 0) {
    throw "The project .venv must use 64-bit Python 3.12."
}

& $environmentPython -m pip install numpy pandas scikit-learn matplotlib seaborn xgboost ipykernel notebook
if ($LASTEXITCODE -ne 0) {
    throw "Could not install the notebook dependencies."
}

& $environmentPython -m ipykernel install --user --name roc-auc-tutorial-312-x64 --display-name "ROC AUC Tutorial (Python 3.12 x64)"
if ($LASTEXITCODE -ne 0) {
    throw "Could not register the Jupyter kernel."
}

Write-Host "Setup complete. Select 'ROC AUC Tutorial (Python 3.12 x64)' in the notebook kernel picker."