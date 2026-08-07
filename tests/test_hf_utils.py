"""hf_utils 의 두 함수 검증 — 네트워크 없이 확인한다.

  ensure_model_dir()  로컬 폴더 검증 (모델 클래스가 사용)
  download_model()    Hub 다운로드 (snapshot_download 를 모킹)
"""
import pytest

from embedded.hf_utils import REQUIRED_FILE, download_model, ensure_model_dir


def make_model_dir(path, complete: bool = True):
    """모델 폴더를 만든다. complete=False 면 핵심 파일 없이 폴더만."""
    path.mkdir(parents=True, exist_ok=True)
    if complete:
        (path / REQUIRED_FILE).write_text("{}", encoding="utf-8")
    return str(path)


# ---- ensure_model_dir: 검증만, 네트워크 접근 없음 ---------------------------

def test_complete_dir_is_returned_as_is(tmp_path):
    d = make_model_dir(tmp_path / "model")
    assert ensure_model_dir(d) == d


def test_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError) as e:
        ensure_model_dir(str(tmp_path / "nope"))
    assert "download_model" in str(e.value)      # 해결 방법을 안내해야 한다


def test_repo_id_is_not_treated_as_path():
    """레포 ID 를 줘도 Hub 로 폴백하지 않고 실패해야 한다.
    (백엔드 라이브러리는 없는 경로를 레포 ID 로 오인해 조용히 받아온다)"""
    with pytest.raises(FileNotFoundError):
        ensure_model_dir("BAAI/bge-m3")


def test_incomplete_dir_raises(tmp_path):
    """다운로드가 중단돼 폴더만 남은 경우를 걸러낸다."""
    d = make_model_dir(tmp_path / "half", complete=False)
    with pytest.raises(FileNotFoundError) as e:
        ensure_model_dir(d)
    assert REQUIRED_FILE in str(e.value)


def test_file_instead_of_dir_raises(tmp_path):
    f = tmp_path / "not_a_dir.txt"
    f.write_text("x", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        ensure_model_dir(str(f))


# ---- download_model: 개발 PC 전용 ------------------------------------------

def test_download_passes_expected_args(tmp_path, monkeypatch):
    captured = {}

    def fake_snapshot_download(**kwargs):
        captured.update(kwargs)
        return kwargs["local_dir"]

    monkeypatch.setattr(
        "huggingface_hub.snapshot_download", fake_snapshot_download
    )
    target = str(tmp_path / "dl")
    got = download_model(
        "BAAI/bge-m3",
        target,
        revision="main",
        token="tok",
        ignore_patterns=["onnx/*"],
        max_workers=2,
    )

    assert got == target
    assert captured["repo_id"] == "BAAI/bge-m3"
    assert captured["local_dir"] == target
    assert captured["revision"] == "main"
    assert captured["token"] == "tok"
    assert captured["ignore_patterns"] == ["onnx/*"]
    assert captured["max_workers"] == 2


def test_download_called_even_if_dir_exists(tmp_path, monkeypatch):
    """존재 검사를 두지 않는다 — snapshot_download 가 멱등하고 이어받기를
    지원하므로, 중단된 다운로드가 영구히 깨진 채 남지 않게 항상 호출한다."""
    calls = []
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda **kw: calls.append(kw) or kw["local_dir"],
    )
    existing = make_model_dir(tmp_path / "already", complete=False)  # 반쪽 폴더
    download_model("BAAI/bge-m3", existing)
    assert len(calls) == 1


def test_download_requires_local_dir_positionally():
    """local_dir 은 필수 인자다 — 빠뜨리면 TypeError."""
    with pytest.raises(TypeError):
        download_model("BAAI/bge-m3")


def test_download_options_are_keyword_only(tmp_path, monkeypatch):
    """revision 이후는 키워드 전용 — 위치로 넘기면 TypeError."""
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda **kw: kw["local_dir"])
    with pytest.raises(TypeError):
        download_model("BAAI/bge-m3", str(tmp_path / "d"), "main")
