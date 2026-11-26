"""Unit tests to validate Kubernetes deployment and service manifests (Stage 3)."""

from pathlib import Path
import pytest
import yaml


@pytest.fixture
def k8s_manifests():
    """Load and parse k8s deployment and service yaml files."""
    dep_path = Path("k8s/deployment.yaml")
    svc_path = Path("k8s/service.yaml")
    assert dep_path.exists(), "k8s/deployment.yaml must exist"
    assert svc_path.exists(), "k8s/service.yaml must exist"

    with open(dep_path, "r", encoding="utf-8") as f:
        dep = yaml.safe_load(f)
    with open(svc_path, "r", encoding="utf-8") as f:
        svc = yaml.safe_load(f)
    return dep, svc


def test_deployment_spec(k8s_manifests):
    """Verify deployment name, replica count, image tag, container port, and selectors."""
    dep, _ = k8s_manifests
    assert dep["metadata"]["name"] == "mp055-api"
    assert dep["spec"]["replicas"] == 4

    container = dep["spec"]["template"]["spec"]["containers"][0]
    assert container["image"] == "mp055-api:1.0"
    assert container["ports"][0]["containerPort"] == 8000


def test_service_spec(k8s_manifests):
    """Verify service name, ports, and selector matching."""
    dep, svc = k8s_manifests
    assert svc["metadata"]["name"] == "mp055-svc"

    # Selector must match pod labels
    dep_labels = dep["spec"]["template"]["metadata"]["labels"]
    svc_selector = svc["spec"]["selector"]
    assert svc_selector["app"] == dep_labels["app"]
    assert svc_selector["app"] == "mp055-api"

    # Ports: port 80 forwarding to 8000
    port_spec = svc["spec"]["ports"][0]
    assert port_spec["port"] == 80
    assert port_spec["targetPort"] == 8000


def test_probes_configuration(k8s_manifests):
    """Verify readiness and liveness probes; liveness must wait longer than readiness."""
    dep, _ = k8s_manifests
    container = dep["spec"]["template"]["spec"]["containers"][0]

    assert "readinessProbe" in container
    assert "livenessProbe" in container

    readiness = container["readinessProbe"]
    liveness = container["livenessProbe"]

    assert readiness["httpGet"]["path"] == "/health"
    assert liveness["httpGet"]["path"] == "/health"
    assert readiness["httpGet"]["port"] == 8000
    assert liveness["httpGet"]["port"] == 8000

    # Liveness must wait longer to allow application to initialize
    assert liveness["initialDelaySeconds"] > readiness["initialDelaySeconds"]
