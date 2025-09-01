#!/bin/bash

# Upgrade pip and setuptools first
pip install --upgrade pip setuptools wheel

# Install dependencies
pip install -r requirements.txt
