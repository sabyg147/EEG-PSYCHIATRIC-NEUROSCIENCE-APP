#!/usr/bin/env bash
# exit on error
set -o errexit

echo "Upgrading core Python build tools..."
python -m pip install --upgrade pip
python -m pip install --upgrade setuptools wheel

echo "Installing project requirements..."
pip install -r requirements.txt
