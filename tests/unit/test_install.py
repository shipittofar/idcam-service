"""Installation layout, systemd unit, and first-boot config assumptions."""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
from pathlib import Path

import pytest
import yaml

from idcam.config.loader import load_config
from idcam.config.schema import AppConfig
from idcam.exceptions import ConfigurationError

ROOT = Path(__file__).resolve().parents[2]
INSTALL_SH = ROOT / "scripts" / "install.sh"
UNINSTALL_SH = ROOT / "scripts" / "uninstall.sh"
UNIT_FILE = ROOT / "deploy" / "idcam.service"
TEMPLATE = ROOT / "deploy" / "config.yaml.template"
ENV_TEMPLATE = ROOT / "deploy" / "environment"
EXAMPLE = ROOT / "config.example.yaml"


def test_install_script_exists_and_requires_root_or_staging():
    text = INSTALL_SH.read_text(encoding="utf-8")
    assert "EUID" in text or "idcam" in text
    assert "Run as root" in text or "sudo ./scripts/install.sh" in text
    assert "--staging" in text
    assert "groupadd" in text
    assert "useradd" in text
    assert "RuntimeDirectory=idcam" in text
    assert "/etc/idcam" in text
    assert "/var/lib/idcam" in text
    assert "/var/log/idcam" in text
    assert "IDCAM_AUTH_TOKEN" in text
    assert "never overwritten" in text.lower() or "Keeping existing" in text


def test_installer_does_not_mkdir_run_idcam():
    text = INSTALL_SH.read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        assert "mkdir" not in stripped or "/run/idcam" not in stripped
        assert "install -d" not in stripped or "/run/idcam" not in stripped


def test_installer_does_not_use_world_writable_modes():
    text = INSTALL_SH.read_text(encoding="utf-8")
    assert "0777" not in text
    assert "0666" not in text
    assert "chmod 777" not in text
    assert "chmod 666" not in text


def test_installer_does_not_install_example_as_production_config():
    text = INSTALL_SH.read_text(encoding="utf-8")
    assert "config.yaml.template" in text
    assert "config.example.yaml" not in text


def test_systemd_unit_runs_as_idcam_with_runtime_directory():
    text = UNIT_FILE.read_text(encoding="utf-8")
    assert "User=idcam" in text
    assert "Group=idcam" in text
    assert "RuntimeDirectory=idcam" in text
    assert "RuntimeDirectoryMode=0755" in text
    assert "StateDirectory=idcam" in text
    assert "LogsDirectory=idcam" in text
    assert "EnvironmentFile=-/etc/idcam/environment" in text
    assert "Environment=IDCAM_CONFIG=/etc/idcam/config.yaml" in text
    assert "Environment=IDCAM_ENV=production" in text
    assert "ExecStart=/opt/idcam/venv/bin/idcam" in text
    assert "Restart=on-failure" in text
    assert "RestartPreventExitStatus=2" in text
    assert "NoNewPrivileges=true" in text
    assert "ProtectSystem=strict" in text
    assert "User=root" not in text


def test_production_template_is_not_silently_example_com():
    text = TEMPLATE.read_text(encoding="utf-8")
    assert "CHANGE_ME" in text
    assert "unconfigured.invalid" in text
    assert "api.example.com" not in text
    data = yaml.safe_load(text)
    with pytest.raises(Exception):
        AppConfig.model_validate(data)


def test_production_template_file_fails_loader(tmp_path, monkeypatch):
    monkeypatch.delenv("IDCAM_ENV", raising=False)
    monkeypatch.delenv("IDCAM_DEVICE_ID", raising=False)
    monkeypatch.delenv("IDCAM_SERVER_URL", raising=False)
    dest = tmp_path / "config.yaml"
    dest.write_text(TEMPLATE.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(ConfigurationError):
        load_config(dest)


def test_example_yaml_not_used_as_production_default():
    example = EXAMPLE.read_text(encoding="utf-8")
    assert "api.example.com" in example
    unit = UNIT_FILE.read_text(encoding="utf-8")
    assert "IDCAM_ENV=production" in unit


def test_environment_template_has_no_secrets():
    text = ENV_TEMPLATE.read_text(encoding="utf-8")
    assert "IDCAM_AUTH_TOKEN" in text
    for line in text.splitlines():
        if line.startswith("IDCAM_AUTH_TOKEN="):
            assert line.strip() in {"IDCAM_AUTH_TOKEN=", "IDCAM_AUTH_TOKEN=''"}


def test_uninstall_does_not_purge_without_confirmation():
    text = UNINSTALL_SH.read_text(encoding="utf-8")
    assert "--purge" in text
    assert "DELETE" in text
    assert "/etc/idcam" in text
    assert "Preserved" in text or "keeps" in text.lower()


def _bash() -> str | None:
    """Prefer Git Bash. Ignore the Windows Store/WSL stub at System32\\bash.exe."""
    candidates: list[str] = []
    for name in ("bash", "bash.exe"):
        found = shutil.which(name)
        if found:
            candidates.append(found)
    for extra in (
        r"C:\Program Files\Git\bin\bash.exe",
        r"C:\Program Files\Git\usr\bin\bash.exe",
    ):
        if Path(extra).is_file():
            candidates.append(extra)
    for path in candidates:
        normalized = path.replace("/", "\\").lower()
        if "system32" in normalized or "windowsapps" in normalized:
            continue
        return path
    return None


def _posix(path: Path) -> str:
    text = str(path)
    if os.name == "nt":
        text = text.replace("\\", "/")
        if len(text) >= 2 and text[1] == ":":
            text = f"/{text[0].lower()}{text[2:]}"
    return text


# Packages Debian Trixie / Raspberry Pi OS Trixie do not ship. The installer
# must not require these — numpy/opencv come from PyPI wheels, not ATLAS.
TRIXIE_UNAVAILABLE_APT = frozenset(
    {
        "libatlas-base-dev",
        "libatlas3-base",
        "libatlas-base",
        "libopenblas-base",  # Ubuntu name; not a Pi OS / Trixie package
    }
)

REQUIRED_RUNTIME_APT = frozenset(
    {
        "python3",
        "python3-pip",
        "python3-venv",
        "python3-dev",
        "python3-libcamera",
        "python3-picamera2",
        "v4l-utils",
        "wget",
        "curl",
    }
)

RECOMMENDED_RUNTIME_APT = frozenset(
    {
        "python3-kms++",
        "libcamera-dev",
    }
)


def _parse_bash_array(text: str, name: str) -> list[str]:
    match = re.search(rf"{re.escape(name)}=\((.*?)\)", text, re.S)
    if not match:
        raise AssertionError(f"bash array {name} not found in install.sh")
    tokens: list[str] = []
    for line in match.group(1).splitlines():
        stripped = line.split("#", 1)[0].strip()
        if stripped:
            tokens.extend(stripped.split())
    return tokens


def test_installer_does_not_require_libatlas_on_debian_trixie():
    text = INSTALL_SH.read_text(encoding="utf-8")
    required = _parse_bash_array(text, "IDC_APT_REQUIRED")
    recommended = _parse_bash_array(text, "IDC_APT_RECOMMENDED")
    camera_cli = _parse_bash_array(text, "IDC_APT_CAMERA_CLI")
    all_pkgs = set(required) | set(recommended) | set(camera_cli)

    assert TRIXIE_UNAVAILABLE_APT.isdisjoint(all_pkgs)
    assert REQUIRED_RUNTIME_APT == set(required)
    assert RECOMMENDED_RUNTIME_APT == set(recommended)
    assert "libcamera-apps" not in required
    assert "libatlas-base-dev" not in required
    assert "rpicam-apps" in camera_cli
    assert "libcamera-apps" in camera_cli
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if "apt-get install" in stripped or "IDC_APT_" in stripped:
            assert "libatlas-base-dev" not in stripped
            assert "libatlas" not in stripped

    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "numpy>=" in pyproject
    assert "opencv-python-headless>=" in pyproject
    assert "atlas" not in pyproject.lower()
    assert "libatlas" not in pyproject.lower()
    assert "--system-site-packages" in text
    assert "python3-picamera2" in required
    assert "python3-libcamera" in required


def test_deploy_wrapper_does_not_add_apt_packages():
    text = (ROOT / "deploy" / "install.sh").read_text(encoding="utf-8")
    assert "libatlas-base-dev" not in text
    assert "scripts/install.sh" in text


def test_installer_does_not_rewrite_operator_apt_sources():
    text = INSTALL_SH.read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        assert not stripped.startswith("sed ") or "sources.list" not in stripped
        assert "tee " not in stripped or "sources.list" not in stripped
        assert "rm " not in stripped or "sources.list" not in stripped
    assert "will not modify" in text
    assert "configured multiple times" in text
    assert "leaving existing sources unchanged" not in text or "will not modify" in text


def test_required_apt_failures_are_not_swallowed():
    text = INSTALL_SH.read_text(encoding="utf-8")
    assert "install_required_apt_packages" in text
    assert "require_apt_candidates" in text
    in_required = False
    for line in text.splitlines():
        if line.startswith("install_required_apt_packages()"):
            in_required = True
            continue
        if in_required:
            if line.startswith("}") and "||" not in line:
                break
            if "apt-get install" in line:
                assert "|| true" not in line
                assert "||true" not in line
            if line.startswith("}"):
                break
    assert "This is a hard failure" in text
    assert "have no installation candidate" in text


@pytest.mark.skipif(_bash() is None, reason="bash is required to expand installer apt arrays")
def test_apt_package_arrays_expand_without_trixie_removed_packages():
    bash = _bash()
    script = f"""
set -euo pipefail
eval "$(sed -n '/^IDC_APT_REQUIRED=(/,/^)/p; /^IDC_APT_RECOMMENDED=(/,/^)/p; /^IDC_APT_CAMERA_CLI=(/,/^)/p' '{_posix(INSTALL_SH)}')"
printf 'REQ:%s\\n' "${{IDC_APT_REQUIRED[@]}}"
printf 'REC:%s\\n' "${{IDC_APT_RECOMMENDED[@]}}"
printf 'CLI:%s\\n' "${{IDC_APT_CAMERA_CLI[@]}}"
"""
    result = subprocess.run(
        [bash, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    required = {
        line.split(":", 1)[1]
        for line in result.stdout.splitlines()
        if line.startswith("REQ:")
    }
    camera_cli = {
        line.split(":", 1)[1]
        for line in result.stdout.splitlines()
        if line.startswith("CLI:")
    }
    recommended = {
        line.split(":", 1)[1]
        for line in result.stdout.splitlines()
        if line.startswith("REC:")
    }
    assert required
    assert camera_cli
    assert recommended
    assert TRIXIE_UNAVAILABLE_APT.isdisjoint(required | recommended | camera_cli)
    assert REQUIRED_RUNTIME_APT == required
    assert RECOMMENDED_RUNTIME_APT == recommended


@pytest.mark.skipif(_bash() is None, reason="bash is required for staging installer test")
def test_staging_install_is_idempotent_and_preserves_config(tmp_path):
    bash = _bash()
    staging = tmp_path / "stage"
    cmd = [
        bash,
        _posix(INSTALL_SH),
        "--staging",
        _posix(staging),
        "--skip-python",
    ]
    env = os.environ.copy()
    env.pop("IDC_AUTH_TOKEN", None)
    first = subprocess.run(cmd, check=True, env=env, capture_output=True, text=True)
    assert first.returncode == 0
    combined = f"{first.stdout}\n{first.stderr}"
    assert "Skipping apt package installation" in combined
    assert "libatlas-base-dev" not in combined
    for n in range(1, 9):
        assert f"[{n}/8]" in combined
    assert "Python/camera import checks skipped" in combined
    assert "Validation passed" in combined

    config = staging / "etc" / "idcam" / "config.yaml"
    env_file = staging / "etc" / "idcam" / "environment"
    unit = staging / "etc" / "systemd" / "system" / "idcam.service"
    state = staging / "var" / "lib" / "idcam"
    tmp_dir = state / "tmp"
    logs = staging / "var" / "log" / "idcam"
    run_dir = staging / "run" / "idcam"

    assert config.is_file()
    assert env_file.is_file()
    assert unit.is_file()
    assert tmp_dir.is_dir()
    assert logs.is_dir()
    assert not run_dir.exists()
    unit_text = unit.read_text(encoding="utf-8")
    assert "RuntimeDirectory=idcam" in unit_text
    assert "User=idcam" in unit_text
    assert "Group=idcam" in unit_text
    assert "User=root" not in unit_text
    assert "CHANGE_ME" in config.read_text(encoding="utf-8")
    assert "unconfigured.invalid" in config.read_text(encoding="utf-8")
    assert "api.example.com" not in config.read_text(encoding="utf-8")

    marker = "device_id: preserved-device\n"
    config.write_text(config.read_text(encoding="utf-8") + "\n# preserved\n", encoding="utf-8")
    env_file.write_text(env_file.read_text(encoding="utf-8") + "\nIDCAM_AUTH_TOKEN=keep-me\n", encoding="utf-8")
    (tmp_dir / "capture-keep.jpg").write_bytes(b"keep")
    (logs / "idcam.log").write_text("old-log\n", encoding="utf-8")

    second = subprocess.run(cmd, check=True, env=env, capture_output=True, text=True)
    assert second.returncode == 0
    assert "# preserved" in config.read_text(encoding="utf-8")
    assert "IDCAM_AUTH_TOKEN=keep-me" in env_file.read_text(encoding="utf-8")
    assert (tmp_dir / "capture-keep.jpg").read_bytes() == b"keep"
    assert "old-log" in (logs / "idcam.log").read_text(encoding="utf-8")
    assert not run_dir.exists()

    if os.name != "nt":
        for directory in (staging / "etc" / "idcam", state, tmp_dir, logs):
            mode = stat.S_IMODE(directory.stat().st_mode)
            assert mode & stat.S_IWOTH == 0, f"{directory} is world-writable"


@pytest.mark.skipif(_bash() is None, reason="bash is required to source installer functions")
def test_sourcing_installer_does_not_mutate_the_system(tmp_path):
    bash = _bash()
    script = f"""
source '{_posix(INSTALL_SH)}'
echo SOURCED_OK
"""
    result = subprocess.run(
        [bash, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
    )
    combined = f"{result.stdout}\n{result.stderr}"
    assert "SOURCED_OK" in result.stdout
    assert "[1/8]" not in combined
    assert not (tmp_path / "opt").exists()
    assert not (tmp_path / "etc").exists()


@pytest.mark.skipif(_bash() is None, reason="bash is required to test apt candidate failure")
def test_missing_required_package_fails_cleanly_with_package_name():
    bash = _bash()
    script = f"""
source '{_posix(INSTALL_SH)}'
apt_has_candidate() {{ return 1; }}
require_apt_candidates definitely-not-a-real-idcam-package
echo SHOULD_NOT_REACH
"""
    result = subprocess.run([bash, "-c", script], capture_output=True, text=True)
    combined = f"{result.stdout}\n{result.stderr}"
    assert result.returncode != 0
    assert "definitely-not-a-real-idcam-package" in combined
    assert "no installation candidate" in combined
    assert "SHOULD_NOT_REACH" not in combined
    assert "[4/8]" not in combined
    assert "Creating idcam service user" not in combined


class TestLocalAutostart:
    """setup_local.sh installs a boot service so idCam starts after reboot / first install."""

    TEMPLATE = ROOT / "deploy" / "idcam-local.service.template"
    SCRIPT = ROOT / "scripts" / "setup_local.sh"

    def test_unit_template_restarts_forever_and_starts_at_boot(self):
        text = self.TEMPLATE.read_text(encoding="utf-8")
        for needle in (
            "WantedBy=multi-user.target",
            "Restart=always",
            "StartLimitIntervalSec=0",
            "User=@USER@",
            "WorkingDirectory=@ROOT@",
            "ExecStart=@ROOT@/run.sh",
            "After=network-online.target",
        ):
            assert needle in text, needle

    def test_unit_does_not_depend_on_optional_groups_existing(self):
        assert "SupplementaryGroups=" not in "\n".join(
            line for line in self.TEMPLATE.read_text(encoding="utf-8").splitlines() if not line.startswith("#")
        )

    def test_setup_script_enables_service_and_disables_the_old_one(self):
        text = self.SCRIPT.read_text(encoding="utf-8")
        assert "systemctl enable idcam-local" in text
        assert "systemctl restart idcam-local" in text
        assert "systemctl disable --now idcam " in text  # old /opt/idcam service: same camera
        assert "--no-autostart" in text and "--remove-autostart" in text
        assert "idcam-local.service.template" in text

    def test_setup_script_is_valid_bash(self):
        import shutil
        import subprocess

        bash = shutil.which("bash")
        if bash is None:
            pytest.skip("bash not available")
        result = subprocess.run([bash, "-n", str(self.SCRIPT)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
