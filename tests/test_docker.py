"""Unit tests to validate Dockerfile configuration (Stage 3)."""

from pathlib import Path
import pytest


def test_dockerfile_structure():
    """Verify Dockerfile base image, dependencies ordering, port exposure, and entrypoint."""
    dockerfile_path = Path("Dockerfile")
    assert dockerfile_path.exists(), "Dockerfile must exist in repo root"

    content = dockerfile_path.read_text(encoding="utf-8")
    lines = [line.strip() for line in content.splitlines() if line.strip() and not line.startswith("#")]

    # Must use python:3.12-slim
    assert any("FROM python:3.12-slim" in line for line in lines)

    # Must copy requirements and pip install before copying source code
    req_copy_idx = next(i for i, line in enumerate(lines) if "COPY requirements.txt" in line)
    pip_install_idx = next(i for i, line in enumerate(lines) if "pip install" in line)
    code_copy_idx = next(i for i, line in enumerate(lines) if "COPY src/" in line or "COPY ." in line)

    assert req_copy_idx < pip_install_idx < code_copy_idx, (
        "requirements.txt must be copied and installed before copying source code for layer caching"
    )

    # Must expose port 8000
    assert any("EXPOSE 8000" in line for line in lines)

    # Must bind uvicorn to 0.0.0.0 and port 8000
    assert any("0.0.0.0" in line and "8000" in line and "uvicorn" in line for line in lines)
