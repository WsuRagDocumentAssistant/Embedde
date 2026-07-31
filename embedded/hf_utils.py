import os
from typing import List, Optional

from huggingface_hub import snapshot_download

#: 로드 가능한 모델 폴더라면 반드시 있어야 하는 파일.
#: 폴더 존재만으로는 중단된 다운로드(빈 폴더)를 걸러낼 수 없어 이 파일로 판단한다.
_REQUIRED_FILE = "config.json"


def _is_complete_model_dir(path: str) -> bool:
    """모델 폴더가 로드 가능한 상태인지(핵심 파일이 있는지) 확인한다."""
    return os.path.isfile(os.path.join(path, _REQUIRED_FILE))


def resolve_model_path(
    model_name: str,
    local_dir: Optional[str] = None,
    allow_download: bool = False,
    revision: Optional[str] = None,
    token: Optional[str] = None,
    ignore_patterns: Optional[List[str]] = None,
    max_workers: int = 4,
    local_files_only: bool = False,
) -> str:
    """모델 식별자를 실제 로드 가능한 경로로 변환한다.

    "모델 파일을 어디서 가져오느냐"는 인코딩 방식과 무관하므로
    상속 계층이 아니라 유틸 함수로 둔다. 각 구현체가 __init__에서 호출.

    다운로드는 allow_download=True 로 명시적으로 켤 때만 일어난다.
    기본값(False)에서는 이미 존재하는 로컬 경로만 사용하며, 경로가 없으면
    조용히 Hub를 조회하는 대신 FileNotFoundError 를 던진다. 오프라인 서버
    배포에서 "받아온 폴더를 쓰는" 것이 정상 경로이고, 의도치 않은 네트워크
    접근은 사고이기 때문.

    인자의 역할이 모드에 따라 갈린다 — 겹치지 않게 분리해 둔다:
      - allow_download=False (기본): model_name = 로드할 로컬 폴더 경로.
        local_dir 은 쓰지 않는다(지정하면 ValueError).
      - allow_download=True: model_name = Hub 레포 ID, local_dir = 받을 폴더(필수).

    Args:
        model_name: 로컬 폴더 경로, 또는 (allow_download=True 일 때) Hub 레포 ID
        local_dir: 다운로드 대상 폴더. allow_download=True 일 때만 유효하며
            그때는 필수다. 이 폴더에 레포 파일 구조가 그대로 놓인다.
        allow_download: True면 Hub 접근/다운로드를 허용한다. 개발 PC에서
            가중치를 받아올 때만 켜고, 운영 서버에서는 끈 채로 둔다.
        revision: 브랜치/태그/커밋 해시. None이면 main 브랜치 최신 커밋
        token: private 레포 인증 토큰
        ignore_patterns: 제외할 파일 글롭 패턴 (예: ["*.bin", "onnx/*"])
        max_workers: 병렬 다운로드 스레드 수
        local_files_only: True면 네트워크 요청 없이 로컬 파일만 사용

    Returns:
        로드에 사용할 실제 경로. 호출부는 이 반환값을 실제 로드 경로로 기록할 것
        (인자로 받은 model_name 이 아니라).

    Raises:
        FileNotFoundError: 로컬 경로가 없거나, 있어도 모델 폴더로서 불완전할 때.
        ValueError: 모드에 맞지 않는 인자 조합일 때.
    """
    if not allow_download:
        # 다운로드 금지 모드: model_name 만이 로드 경로다.
        # local_dir(다운로드 대상)은 이 모드에서 의미가 없으므로 허용하지 않는다.
        # (허용하면 두 인자가 어긋날 때 조용히 한쪽이 무시된다)
        if local_dir is not None:
            raise ValueError(
                "local_dir 은 allow_download=True 일 때만 사용합니다. "
                f"로드할 폴더는 model_name 에 지정하세요 (local_dir={local_dir!r})."
            )
        if not os.path.exists(model_name):
            raise FileNotFoundError(
                f"로컬 모델 경로를 찾을 수 없습니다: {model_name!r}\n"
                "다운로드가 꺼져 있어(allow_download=False) Hub를 조회하지 않습니다. "
                "이미 받아둔 폴더 경로를 지정하거나, 개발 환경에서 새로 받으려면 "
                "allow_download=True 와 local_dir 을 함께 지정하세요."
            )
        if not _is_complete_model_dir(model_name):
            raise FileNotFoundError(
                f"모델 폴더가 불완전합니다(‘{_REQUIRED_FILE}’ 없음): {model_name!r}\n"
                "다운로드가 중단된 폴더이거나 반입이 누락됐을 수 있습니다. "
                "폴더를 통째로 다시 복사하세요."
            )
        return model_name

    # 다운로드 허용 모드: model_name 은 Hub 레포 ID 로 해석된다.
    if local_dir is None:
        # 받는 위치를 항상 명시하도록 강제한다. 지정하지 않으면 파일이
        # 전역 HF 캐시에만 남아 USB 반입용 폴더를 따로 만들 수 없다.
        raise ValueError(
            "allow_download=True 이면 local_dir 을 반드시 지정하세요. "
            "받은 파일을 그대로 옮길 수 있는 폴더가 필요합니다."
        )

    # 존재 검사를 두지 않는다 — snapshot_download 자체가 멱등하고 이어받기를
    # 지원하므로, 이미 받은 파일은 건너뛰고 누락/중단된 것만 다시 받는다.
    # (폴더 존재만 확인하면 중단된 다운로드가 영구히 깨진 채로 남는다)
    snapshot_download(
        repo_id=model_name,
        local_dir=local_dir,
        revision=revision,
        token=token,
        ignore_patterns=ignore_patterns,
        max_workers=max_workers,
        local_files_only=local_files_only,
    )
    return local_dir
