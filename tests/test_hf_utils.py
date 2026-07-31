"""resolve_model_path 의 모드별 분기 검증 — 네트워크 없이 확인한다.

다운로드가 실제로 일어나는 경로는 snapshot_download 를 모킹해서 검증하므로
Hub 접근이 필요 없다.
"""
import pytest

from embedded.hf_utils import _REQUIRED_FILE, resolve_model_path


def make_model_dir(path, complete: bool = True):
    """모델 폴더를 만든다. complete=False 면 핵심 파일 없이 폴더만."""
    path.mkdir(parents=True, exist_ok=True)
    if complete:
        (path / _REQUIRED_FILE).write_text("{}", encoding="utf-8")
    return str(path)


# ---- allow_download=False (기본): 로컬 경로만 --------------------------------

def test_existing_complete_dir_is_returned(tmp_path):
    d = make_model_dir(tmp_path / "model")
    assert resolve_model_path(d) == d


def test_missing_path_raises_without_touching_hub(tmp_path):
    missing = str(tmp_path / "nope")
    with pytest.raises(FileNotFoundError) as e:
        resolve_model_path(missing)
    assert "allow_download" in str(e.value)      # 해결 방법을 안내해야 한다


def test_repo_id_without_download_raises(tmp_path):
    """레포 ID 를 줘도 Hub 로 폴백하지 않고 실패해야 한다."""
    with pytest.raises(FileNotFoundError):
        resolve_model_path("BAAI/bge-m3")


def test_incomplete_dir_raises(tmp_path):
    """다운로드가 중단돼 폴더만 남은 경우를 걸러낸다."""
    d = make_model_dir(tmp_path / "half", complete=False)
    with pytest.raises(FileNotFoundError) as e:
        resolve_model_path(d)
    assert _REQUIRED_FILE in str(e.value)


def test_local_dir_rejected_when_download_disabled(tmp_path):
    """local_dir 은 다운로드 전용 인자 — 기본 모드에서 주면 거부한다.
    (허용하면 model_name 과 어긋날 때 조용히 한쪽이 무시된다)"""
    d = make_model_dir(tmp_path / "model")
    with pytest.raises(ValueError) as e:
        resolve_model_path("BAAI/bge-m3", local_dir=d)
    assert "local_dir" in str(e.value)


# ---- allow_download=True: Hub 레포 ID + local_dir 필수 ----------------------

def test_download_requires_local_dir():
    with pytest.raises(ValueError) as e:
        resolve_model_path("BAAI/bge-m3", allow_download=True)
    assert "local_dir" in str(e.value)


def test_download_called_with_expected_args(tmp_path, monkeypatch):
    captured = {}

    def fake_snapshot_download(**kwargs):
        captured.update(kwargs)
        return kwargs["local_dir"]

    monkeypatch.setattr(
        "embedded.hf_utils.snapshot_download", fake_snapshot_download
    )
    target = str(tmp_path / "dl")
    got = resolve_model_path(
        "BAAI/bge-m3",
        local_dir=target,
        allow_download=True,
        revision="main",
        token="tok",
        ignore_patterns=["onnx/*"],
        max_workers=2,
        local_files_only=False,
    )

    assert got == target
    assert captured["repo_id"] == "BAAI/bge-m3"       # model_name 은 레포 ID 로 해석
    assert captured["local_dir"] == target
    assert captured["revision"] == "main"
    assert captured["token"] == "tok"
    assert captured["ignore_patterns"] == ["onnx/*"]
    assert captured["max_workers"] == 2


def test_download_always_called_even_if_dir_exists(tmp_path, monkeypatch):
    """존재 검사를 두지 않는다 — snapshot_download 가 멱등하고 이어받기를
    지원하므로, 중단된 다운로드가 영구히 깨진 채 남지 않게 항상 호출한다."""
    calls = []
    monkeypatch.setattr(
        "embedded.hf_utils.snapshot_download",
        lambda **kw: calls.append(kw) or kw["local_dir"],
    )
    existing = make_model_dir(tmp_path / "already", complete=False)  # 반쪽 폴더
    resolve_model_path("BAAI/bge-m3", local_dir=existing, allow_download=True)
    assert len(calls) == 1
