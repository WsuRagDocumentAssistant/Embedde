from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np


class BaseEmbeddedModel(ABC):
    """모든 임베딩 모델의 최소 공통분모.

    dense 벡터 하나는 반드시 뽑을 수 있어야 한다는 것만 강제한다.
    배치 분할, GPU->CPU 변환, 정규화는 여기서 공통 처리하고,
    서브클래스는 _encode_raw()만 구현하면 된다.
    """

    #: 검색용 인코딩 시 텍스트 앞에 붙는 접두사.
    #: e5 계열은 "query: "/"passage: ", bge 계열은 instruction 등
    #: 모델별로 클래스 속성 또는 __init__에서 오버라이드한다.
    query_prefix: str = ""
    passage_prefix: str = ""

    def __init__(self, normalize: bool = True, batch_size: int = 12):
        self.normalize = normalize
        self.batch_size = batch_size

    @property
    @abstractmethod
    def model_name(self) -> str:
        """사람이 읽을 수 있는 모델 식별자 (레포 ID 등)."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """dense 벡터 차원 수. 벡터 DB 컬렉션 생성 시 필요하다."""

    def encode(self, texts: List[str], batch_size: Optional[int] = None) -> np.ndarray:
        """배치별로 _encode_raw()를 호출해 결과를 모으고,
        GPU -> CPU 변환은 전체 배치가 끝난 뒤 딱 한 번만 수행한다."""
        bs = batch_size or self.batch_size
        raw_batches = [
            self._encode_raw(texts[i : i + bs]) for i in range(0, len(texts), bs)
        ]
        merged = self._concat_raw(raw_batches)
        vecs = self._to_numpy(merged).astype(np.float32)
        if self.normalize:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vecs = vecs / norms
        return vecs

    def encode_queries(self, texts: List[str], **kwargs) -> np.ndarray:
        """검색 쿼리 인코딩. 모델별 쿼리 접두사를 붙인 뒤 encode()."""
        return self.encode([self.query_prefix + t for t in texts], **kwargs)

    def encode_documents(self, texts: List[str], **kwargs) -> np.ndarray:
        """색인 대상 문서 인코딩. 모델별 문서 접두사를 붙인 뒤 encode()."""
        return self.encode([self.passage_prefix + t for t in texts], **kwargs)

    @abstractmethod
    def _encode_raw(self, texts: List[str]) -> Any:
        """서브클래스가 구현: 배치 하나 -> 모델 고유의 원본 출력.
        GPU 텐서를 그대로 반환해도 됨 — 여기서 CPU로 내리지 말 것.
        (변환 타이밍은 base의 encode()가 마지막에 한 번만 처리)"""
        raise NotImplementedError

    @staticmethod
    def _concat_raw(batches: list) -> Any:
        """배치별 raw 결과를 이어붙인다. torch.Tensor면 GPU 위에서 concat."""
        first = batches[0]
        if hasattr(first, "detach"):          # torch.Tensor
            import torch
            return torch.cat(batches, dim=0)  # 아직 GPU, 동기화 없음
        return np.concatenate([np.asarray(b) for b in batches], axis=0)

    @staticmethod
    def _to_numpy(x) -> np.ndarray:
        if isinstance(x, np.ndarray):
            return x
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()   # 동기화는 여기 딱 한 번만 발생
        return np.array(x)


class SparseCapable(ABC):
    """sparse(lexical) 임베딩을 지원하는 모델만 추가로 구현하는 인터페이스.

    BGE-M3, SPLADE 등 일부 모델만 상속받는다.
    호출부에서는 isinstance(model, SparseCapable)로 지원 여부를 분기한다.
    """

    @abstractmethod
    def encode_sparse(self, texts: List[str]) -> List[Dict[int, float]]:
        """텍스트 리스트 -> 문장마다 {token_id: weight} 딕셔너리."""
        raise NotImplementedError
