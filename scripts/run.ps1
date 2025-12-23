param(
    [Parameter(Mandatory=$true)][string]$Target,
    [string]$EnvName = 'trading',
    [int]$Epochs = 3,
    [string]$Ticker = 'AAPL'
)

switch ($Target) {
    'setup-windows' {
        Write-Host "Running Windows setup script..."
        & .\scripts\setup_windows_env.ps1 -EnvName $EnvName -PythonVersion 3.11
        break
    }
    'install-reqs' {
        Write-Host "Installing requirements into $EnvName..."
        conda run -n $EnvName python -m pip install -r requirements.txt
        break
    }
    'tf-test' {
        Write-Host "Running TensorFlow import test in $EnvName..."
        conda run -n $EnvName python scripts/tf_import_test.py
        break
    }
    'quick-train' {
        Write-Host "Running quick synthetic training..."
        conda run -n $EnvName python scripts/run_quick_train.py
        break
    }
    'quick-test' {
        Write-Host "Running quick synthetic prediction test..."
        conda run -n $EnvName python scripts/run_quick_test.py
        break
    }
    'real-train' {
        Write-Host "Running short real training: epochs=$Epochs ticker=$Ticker"
        conda run -n $EnvName python scripts/run_real_train.py $Epochs $Ticker
        break
    }
    'btc-test' {
        Write-Host "Running BTC experiment: epochs=$Epochs days=$Ticker" # using $Ticker param slot to pass DAYS
        conda run -n $EnvName python scripts/run_btc_experiment.py $Epochs $Ticker
        break
    }
    'orchestrator-train' {
        Write-Host "Running full orchestrator train (orquestratorTrain.py)"
        conda run -n $EnvName python .\orquestratorTrain.py
        break
    }
    'orchestrator-test' {
        Write-Host "Running full orchestrator test (orquestadorTest.py)"
        conda run -n $EnvName python .\orquestadorTest.py
        break
    }
    default {
        Write-Host "Unknown target: $Target"
        Write-Host "Available targets: setup-windows, install-reqs, tf-test, quick-train, quick-test, real-train"
        exit 1
    }
}
