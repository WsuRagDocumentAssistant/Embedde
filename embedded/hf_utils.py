"""모델 파일을 다루는 유틸 — 검증과 다운로드를 역할별로 분리해 둔다.

  ensure_model_dir()  로컬 폴더가 로드 가능한지 검증 (모델 클래스가 사용)
  download_model()    Hub 에서 가중치를 받아온다 (개발 PC 에서만 사용)

두 함수를 나눠 둔 이유: 모델 클래스는 "이미 있는 폴더를 로드"만 하고,
다운로드는 개발 PC 에서 한 번 하는 별개 작업이다. 플래그 하나로 모드를
가르는 대신 함수를 갈라 두면 의도가 시그니처에 드러난다.
"""
import os
from typing import List, Optional

#: 로드 가능한 모델 폴더라면 반드시 있어야 하는 파일.
#: 폴더 존재만으로는 중단된 다운로드(빈 폴더)를 걸러낼 수 없어 이 파일로 판단한다.
REQUIRED_FILE = "config.json"


def ensure_model_dir(model_path: str) -> str:
    """로컬 모델 폴더가 로드 가능한 상태인지 검증하고 그대로 반환한다.

    백엔드 라이브러리들은 경로가 없으면 그것을 Hub 레포 ID 로 오인해 조용히
    다운로드를 시도한다(FlagEmbedding 은 인자로 끌 수도 없다). 그래서 라이브러리에
    넘기기 전에 여기서 막고, 원인이 분명한 에러를 던진다.

    Args:
        model_path: 이미 받아둔 모델 폴더 경로.

    Returns:
        검증을 통과한 model_path (그대로).

    Raises:
        FileNotFoundError: 폴더가 없거나, 있어도 모델 폴더로서 불완전할 때.
    """
    if not os.path.isdir(model_path):
        raise FileNotFoundError(
            f"모델 폴더를 찾을 수 없습니다: {model_path!r}\n"
            "이미 받아둔 폴더 경로를 지정하세요. 새로 받으려면 개발 환경에서 "
            "download_model() 을 사용합니다(운영 서버에서는 받지 않습니다)."
        )
    if not os.path.isfile(os.path.join(model_path, REQUIRED_FILE)):
        raise FileNotFoundError(
            f"모델 폴더가 불완전합니다(‘{REQUIRED_FILE}’ 없음): {model_path!r}\n"
            "다운로드가 중단된 폴더이거나 반입이 누락됐을 수 있습니다. "
            "폴더를 통째로 다시 복사하세요."
        )
    return model_path


def download_model(
    repo_id: str,
    local_dir: str,
    *,
    revision: Optional[str] = None,
    token: Optional[str] = None,
    ignore_patterns: Optional[List[str]] = None,
    max_workers: int = 4,
) -> str:
    """Hub 에서 가중치를 받아 local_dir 에 놓는다. 개발 PC 에서만 쓴다.

    받은 폴더를 통째로 USB 등으로 서버에 옮기는 것이 이 함수의 용도다.
    운영 서버(외부 통신 차단)에서는 호출하지 않는다.

    Args:
        repo_id: Hub 레포 ID (예: "BAAI/bge-m3").
        local_dir: 받을 폴더. 레포 파일 구조가 이 안에 그대로 놓인다.
        revision: 브랜치/태그/커밋 해시. None이면 main 브랜치 최신 커밋.
        token: private/gated 레포 인증 토큰.
        ignore_patterns: 제외할 파일 글롭 패턴 (예: ["onnx/*"]).
        max_workers: 병렬 다운로드 스레드 수.

    Returns:
        받은 폴더 경로(local_dir).
    """
    from huggingface_hub import snapshot_download  # 지연 import (운영 경로에선 불필요)

    # 존재 검사를 두지 않는다 — snapshot_download 자체가 멱등하고 이어받기를
    # 지원하므로, 이미 받은 파일은 건너뛰고 누락/중단된 것만 다시 받는다.
    # (폴더 존재만 확인하면 중단된 다운로드가 영구히 깨진 채로 남는다)
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        revision=revision,
        token=token,
        ignore_patterns=ignore_patterns,
        max_workers=max_workers,
    )
    return local_dir
