import os
from typing import List, Optional

from huggingface_hub import snapshot_download


def resolve_model_path(
    model_name: str,
    local_dir: Optional[str] = None,
    revision: Optional[str] = None,
    token: Optional[str] = None,
    ignore_patterns: Optional[List[str]] = None,
    max_workers: int = 4,
    local_files_only: bool = False,
) -> str:
    """HF Hub 레포 ID를 실제 로드 가능한 경로로 변환한다.

    "모델 파일을 어디서 가져오느냐"는 인코딩 방식과 무관하므로
    상속 계층이 아니라 유틸 함수로 둔다. 각 구현체가 __init__에서 호출.

    Args:
        model_name: Hub 레포 ID (예: "BAAI/bge-m3")
        local_dir: 지정 시 이 경로에 실제 파일로 다운로드. None이면 HF 기본 캐시 사용
        revision: 브랜치/태그/커밋 해시. None이면 main 브랜치 최신 커밋
        token: private 레포 인증 토큰
        ignore_patterns: 제외할 파일 글롭 패턴 (예: ["*.bin", "onnx/*"])
        max_workers: 병렬 다운로드 스레드 수
        local_files_only: True면 네트워크 요청 없이 로컬 파일만 사용
    """
    if local_dir is None:
        return model_name

    if not os.path.exists(local_dir):
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
