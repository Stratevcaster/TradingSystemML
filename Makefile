# Makefile for TradingSystemML - convenience targets

CONDA_ENV ?= trading
PY_VERSION ?= 3.11
EPOCHS ?= 3
TICKER ?= AAPL

.PHONY: help setup-conda setup-windows install-reqs tf-test quick-train quick-test real-train run-orchestrator-train run-orchestrator-test clean setup-all

help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@echo "  setup-conda        - Run the Bash setup script (Miniforge + env + packages)"
	@echo "  setup-windows      - Run the PowerShell setup (Miniforge, VC redist, env, packages)"
	@echo "  install-reqs       - Install requirements into $(CONDA_ENV) via pip"
	@echo "  tf-test            - Run TensorFlow import test inside $(CONDA_ENV)"
	@echo "  quick-train        - Run quick synthetic 1-epoch training (no network)"
	@echo "  quick-test         - Run quick prediction test (no network)"
	@echo "  real-train         - Run short real training: make real-train EPOCHS=3 TICKER=AAPL"
	@echo "  btc-test           - Run a BTC experiment (train + test): make btc-test EPOCHS=3 DAYS=10"
	@echo "  run-orchestrator-train - Run orquestratorTrain.py (long-running)"
	@echo "  run-orchestrator-test  - Run orquestadorTest.py"
	@echo "  clean              - Remove generated artifacts (results, caches)"
	@echo "  setup-all          - Run setup-conda then install-reqs and tf-test"

setup-conda:
	@echo "Running Bash setup (Miniforge + conda env)"
	@bash ./scripts/setup_conda_env.sh $(CONDA_ENV) $(PY_VERSION)

setup-windows:
	@echo "Running Windows PowerShell setup (Miniforge, VC redist, conda env)"
	@powershell -NoProfile -ExecutionPolicy Bypass -File ./scripts/setup_windows_env.ps1 -EnvName $(CONDA_ENV) -PythonVersion $(PY_VERSION)

install-reqs:
	@echo "Installing requirements into $(CONDA_ENV)"
	@conda run -n $(CONDA_ENV) python -m pip install -r requirements.txt

tf-test:
	@echo "Running TensorFlow import test"
	@conda run -n $(CONDA_ENV) python scripts/tf_import_test.py

quick-train:
	@echo "Running quick synthetic training"
	@conda run -n $(CONDA_ENV) python scripts/run_quick_train.py

quick-test:
	@echo "Running quick synthetic prediction test"
	@conda run -n $(CONDA_ENV) python scripts/run_quick_test.py

real-train:
	@echo "Running short real training: epochs=$(EPOCHS) ticker=$(TICKER)"
	@conda run -n $(CONDA_ENV) python scripts/run_real_train.py $(EPOCHS) $(TICKER)


btc-test:
	@echo "Running BTC short experiment: epochs=$(EPOCHS) days=$(DAYS)"
	@conda run -n $(CONDA_ENV) python scripts/run_btc_experiment.py $(EPOCHS) $(DAYS)

btc-10day:
	@echo "Running BTC short training + 10-day test: epochs=$(EPOCHS)"
	@conda run -n $(CONDA_ENV) python scripts/run_btc_10day.py $(EPOCHS)

run-orchestrator-train:
	@echo "Running orchestrator train (may be long)..."
	@conda run -n $(CONDA_ENV) python orquestratorTrain.py

run-orchestrator-test:
	@echo "Running orchestrator test"
	@conda run -n $(CONDA_ENV) python orquestadorTest.py

clean:
	@echo "Cleaning generated files..."
	@if [ -d "results" ]; then rm -rf results/* || powershell -Command "Remove-Item -Recurse -Force results\\*"; fi
	@if [ -d "__pycache__" ]; then rm -rf __pycache__ || powershell -Command "Remove-Item -Recurse -Force __pycache__"; fi

setup-all: setup-conda install-reqs tf-test
