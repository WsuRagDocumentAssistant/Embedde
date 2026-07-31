import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

#: 배치 크기 기본값. 능력 믹스인과 공통 뿌리가 같은 값을 쓴다.
DEFAULT_BATCH_SIZE = 12


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
    믹스인이 참조하는 batch_size 는 각 믹스인에도 클래스 기본값으로 선언돼 있어,
    Base 없이 믹스인만 상속해도 AttributeError 가 나지 않는다.
    """

    batch_size: int = DEFAULT_BATCH_SIZE

    def __init__(self, batch_size: int = DEFAULT_BATCH_SIZE):
        self.batch_size = batch_size

    @property
    @abstractmethod
    def model_name(self) -> str:
        """사람이 읽을 수 있는 모델 식별자 (레포 ID 등). 모든 모델의 공통 속성."""

    # ---- 리소스 해제 -------------------------------------------------------
    def unload(self) -> None:
        """모델을 메모리/GPU에서 내린다.

        파이썬은 GPU 메모리를 자동으로 반납하지 않으므로, 서버 graceful
        shutdown 등에서 명시적으로 호출해 VRAM을 비운다. 여러 번 호출해도 안전.

        실제 모델 객체를 놓는 일은 _release_model() 훅에 위임한다 — 부모가
        자식의 속성 이름을 알지 않도록.
        """
        self._release_model()
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass  # torch 없는 백엔드(예: 해시/외부 API)면 할 일 없음
        logger.info("모델 unload: %s", self.model_name)

    def _release_model(self) -> None:
        """구현체가 들고 있는 모델 객체 참조를 놓는다(여러 번 호출해도 안전).

        기본 구현은 관례적인 self._model 을 비운다. 다른 이름/여러 개를 쓰는
        구현체는 이 훅만 오버라이드하면 되고 unload() 는 그대로 재사용된다.
        """
        if getattr(self, "_model", None) is not None:
            self._model = None


class DenseCapable(ABC):
    """dense 벡터를 뽑을 수 있는 능력.

    정규화, 접두사 처리, 빈 입력 방어를 여기서 공통 완성한다.
    구체 모델은 _encode_raw() 와 dimension 만 채우면 된다.

    배치 분할은 하지 않는다 — 텍스트 리스트를 통째로 _encode_raw() 에 넘기고
    분할은 백엔드 라이브러리에 맡긴다. sentence-transformers/FlagEmbedding 은
    입력을 길이순으로 정렬해 비슷한 길이끼리 묶어 패딩 낭비를 줄이는데, 미리
    잘라서 넘기면 그 정렬이 청크 내부로 제한돼 처리량이 떨어진다. 한 번에
    넘겨도 결과는 텐서 하나이므로 GPU->CPU 동기화 횟수는 어차피 1회다.
    (배치 크기는 구현체가 self.batch_size 를 백엔드에 전달해 반영한다.)
    """

    #: 검색용 인코딩 시 텍스트 앞에 붙는 접두사 (e5 계열 등). 필요한 모델만 오버라이드.
    query_prefix: str = ""
    passage_prefix: str = ""
    #: L2 정규화 여부. 구체 모델의 __init__ 에서 인스턴스 값으로 덮어쓸 수 있다.
    normalize: bool = True
    #: 백엔드에 전달할 배치 크기. BaseEmbeddedModel 과 같은 기본값을 갖고 있어
    #: 이 믹스인만 단독 상속해도 동작한다.
    batch_size: int = DEFAULT_BATCH_SIZE

    @property
    @abstractmethod
    def dimension(self) -> int:
        """dense 벡터 차원 수. 벡터 DB 컬렉션 생성 시 필요하다."""

    @abstractmethod
    def _encode_raw(self, texts: List[str], batch_size: int) -> Any:
        """서브클래스가 구현: 텍스트 전체 -> 모델 고유의 원본 출력.

        배치 분할은 백엔드 라이브러리에 맡기고, batch_size 를 그대로 전달한다.
        GPU 텐서를 반환해도 됨 — 여기서 CPU로 내리지 말 것(변환은 encode() 담당).
        """
        raise NotImplementedError

    def encode(self, texts: List[str], batch_size: Optional[int] = None) -> np.ndarray:
        """텍스트를 dense 벡터로 인코딩한다.

        Args:
            texts: 인코딩할 문장 리스트. 빈 리스트면 (0, dimension) 빈 배열 반환.
            batch_size: 이번 호출에만 적용할 배치 크기. None이면 인스턴스 기본값.

        Returns:
            (len(texts), dimension) float32 배열. normalize=True면 각 행의 L2 노름이 1.
        """
        if not texts:
            # 빈 입력: 백엔드에 빈 배치를 넘기지 않고 빈 배열을 반환한다.
            return np.empty((0, self.dimension), dtype=np.float32)
        raw = self._encode_raw(texts, batch_size or self.batch_size)
        vecs = self._to_numpy(raw).astype(np.float32)
        if self.normalize:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vecs = vecs / norms
        return vecs

    def encode_queries(self, texts: List[str], **kwargs) -> np.ndarray:
        """검색 쿼리 인코딩. 모델별 쿼리 접두사를 붙인 뒤 encode().

        Args:
            texts: 검색 쿼리 리스트.
            **kwargs: encode()로 그대로 전달(batch_size 등).
        """
        return self.encode([self.query_prefix + t for t in texts], **kwargs)

    def encode_documents(self, texts: List[str], **kwargs) -> np.ndarray:
        """색인 대상 문서 인코딩. 모델별 문서 접두사를 붙인 뒤 encode().

        Args:
            texts: 색인할 문서 리스트.
            **kwargs: encode()로 그대로 전달(batch_size 등).
        """
        return self.encode([self.passage_prefix + t for t in texts], **kwargs)

    @staticmethod
    def _to_numpy(x) -> np.ndarray:
        """백엔드 출력을 numpy 로 변환한다. torch.Tensor면 이때 CPU로 내린다."""
        if isinstance(x, np.ndarray):
            return x
        if hasattr(x, "detach"):              # torch.Tensor (torch import 없이 판별)
            return x.detach().cpu().numpy()   # GPU->CPU 동기화는 여기 한 번만
        return np.array(x)


class SparseCapable(ABC):
    """sparse(lexical) 임베딩을 뽑을 수 있는 능력.

    BGE-M3, SPLADE 등만 가진다. dense 없이 sparse만 하는 모델도 표현 가능.
    호출부에서는 isinstance(model, SparseCapable)로 지원 여부를 분기한다.

    DenseCapable 과 대칭 구조: 빈 입력 방어 등 공통 처리는 encode_sparse()가
    맡고, 구현체는 _encode_sparse_raw()만 채운다. 배치 분할을 백엔드에
    맡기는 것도 동일하다.
    """

    #: 백엔드에 전달할 배치 크기. DenseCapable 과 같은 기본값을 갖고 있어
    #: 이 믹스인만 단독 상속해도 동작한다.
    batch_size: int = DEFAULT_BATCH_SIZE

    def encode_sparse(
        self, texts: List[str], batch_size: Optional[int] = None
    ) -> List[Dict[int, float]]:
        """텍스트 리스트 -> 문장마다 {token_id: weight} 딕셔너리.

        Args:
            texts: 인코딩할 문장 리스트. 빈 리스트면 빈 리스트 반환.
            batch_size: 이번 호출에만 적용할 배치 크기. None이면 인스턴스 기본값.

        Returns:
            문장별 {token_id: weight} 딕셔너리 리스트. 등장한 토큰만 담기며
            (희소 표현), 키는 int·값은 float으로 통일된다.
        """
        if not texts:
            return []
        return self._encode_sparse_raw(texts, batch_size or self.batch_size)

    @abstractmethod
    def _encode_sparse_raw(
        self, texts: List[str], batch_size: int
    ) -> List[Dict[int, float]]:
        """서브클래스가 구현: 비어있지 않은 텍스트 리스트 -> {token_id: weight} 리스트.
        배치 분할은 백엔드 라이브러리에 맡기고 batch_size 를 그대로 전달한다."""
        raise NotImplementedError
