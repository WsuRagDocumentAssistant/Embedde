from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np


class BaseEmbeddedModel(ABC):
    """모든 임베딩 모델의 공통 뿌리 — 특정 출력(dense/sparse)을 강제하지 않는다.

    "임베딩 모델이라면 이름이 있고 배치 크기 개념이 있다" 정도만 공통으로 두고,
    실제 능력(dense 벡터 / sparse 가중치)은 DenseCapable / SparseCapable 로 분리한다.
    구체 모델은 자기가 가진 능력만 골라 다중 상속으로 조립한다.

        class BGEM3Model(BaseEmbeddedModel, DenseCapable, SparseCapable): ...  # 둘 다
        class STModel(BaseEmbeddedModel, DenseCapable): ...                    # dense만
        class SpladeModel(BaseEmbeddedModel, SparseCapable): ...               # sparse만

    능력 믹스인(DenseCapable/SparseCapable)은 BaseEmbeddedModel 을 상속하지 않는다.
    → 다중 상속 시 다이아몬드가 생기지 않아 MRO/super().__init__ 이 단순해진다.
    """

    batch_size: int = 12  # 클래스 기본값 (encode/encode_sparse가 참조)

    def __init__(self, batch_size: int = 12):
        self.batch_size = batch_size

    @property
    @abstractmethod
    def model_name(self) -> str:
        """사람이 읽을 수 있는 모델 식별자 (레포 ID 등). 모든 모델의 공통 속성."""


class DenseCapable(ABC):
    """dense 벡터를 뽑을 수 있는 능력.

    배치 분할, GPU->CPU 변환 타이밍, 정규화, 접두사 처리를 여기서 공통 완성한다.
    구체 모델은 _encode_raw() 와 dimension 만 채우면 된다.
    (dense 전용 유틸 _concat_raw/_to_numpy 도 이 능력 안에 자족적으로 둔다.)
    """

    #: 검색용 인코딩 시 텍스트 앞에 붙는 접두사 (e5 계열 등). 필요한 모델만 오버라이드.
    query_prefix: str = ""
    passage_prefix: str = ""
    #: L2 정규화 여부. 구체 모델의 __init__ 에서 인스턴스 값으로 덮어쓸 수 있다.
    normalize: bool = True

    @property
    @abstractmethod
    def dimension(self) -> int:
        """dense 벡터 차원 수. 벡터 DB 컬렉션 생성 시 필요하다."""

    @abstractmethod
    def _encode_raw(self, texts: List[str]) -> Any:
        """서브클래스가 구현: 배치 하나 -> 모델 고유의 원본 출력.
        GPU 텐서를 그대로 반환해도 됨 — 여기서 CPU로 내리지 말 것.
        (변환 타이밍은 encode()가 마지막에 한 번만 처리)"""
        raise NotImplementedError

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
    """sparse(lexical) 임베딩을 뽑을 수 있는 능력.

    BGE-M3, SPLADE 등만 가진다. dense 없이 sparse만 하는 모델도 표현 가능.
    호출부에서는 isinstance(model, SparseCapable)로 지원 여부를 분기한다.
    """

    @abstractmethod
    def encode_sparse(self, texts: List[str]) -> List[Dict[int, float]]:
        """텍스트 리스트 -> 문장마다 {token_id: weight} 딕셔너리."""
        raise NotImplementedError
