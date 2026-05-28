# Contributing

Thanks for wanting to improve KitchenAI. This guide is short and beginner friendly.

## Quick start

1. Fork the repo and clone your fork.
2. Create a branch: `feature/<short-name>` or `fix/<short-name>`.
3. Make changes and test locally.
4. Open a pull request with a clear summary.

## Development setup

```powershell
cd C:\Users\Asus\Desktop\kitchyen\backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## What to test

There are no formal tests yet. Run this before a PR:

```powershell
python -m compileall backend/app
```

## Coding guidelines

- Keep functions small and focused.
- Avoid committing large datasets or model weights.
- Prefer descriptive names over abbreviations.

## Reporting issues

When filing a bug, include:
- What you were trying to do
- The exact error message or traceback
- Your OS and Python version

## Feature requests

Describe the feature, why it helps, and a rough idea of the implementation.