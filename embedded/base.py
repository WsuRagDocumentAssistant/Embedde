import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

#: 배치 크기 기본값 (BaseEmbeddedModel 이 소유하는 공통 인자).
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

    공통 인자(model_name, batch_size)는 이 클래스가 소유한다. 능력 믹스인은
    self.batch_size 를 읽지만 스스로 정의하지 않으므로, 반드시 이 클래스와
    함께 조립해야 한다(믹스인 단독 상속은 지원 대상이 아니다).
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
        """모델을 메모리/GPU에서 내린다. 여러 번 호출해도 안전하다.

        **같은 프로세스 안에서 모델을 내리고 다른 모델을 올릴 때** 필요하다.
        torch 는 캐싱 할당자를 써서 참조를 놓아도 VRAM 을 자기 캐시로 쥐고 있어,
        empty_cache() 를 명시적으로 불러야 반납된다. 이걸 하지 않으면 순차로
        여러 모델을 올릴 때 VRAM 이 누적된다.

        프로세스 종료 시에는 호출할 필요가 없다 — CUDA/OS 가 알아서 회수한다.
        모델을 하나(또는 여럿) 올려두고 프로세스 수명 내내 쓰는 구조라면
        이 메서드를 쓸 일이 없다.

        구현체가 관례적으로 self._model 에 모델을 들고 있다고 가정한다.
        다른 이름이나 여러 개를 쓰는 구현체는 이 메서드를 오버라이드한다.
        """
        if getattr(self, "_model", None) is not None:
            self._model = None
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass  # torch 없는 백엔드(예: 해시/외부 API)면 할 일 없음
        logger.info("모델 unload: %s", self.model_name)


class DenseCapable(ABC):
    """dense 벡터를 뽑을 수 있는 능력.

    정규화, 접두사 처리, 빈 입력 방어를 여기서 공통 완성한다.
    구체 모델은 _encode_raw() 와 dimension 만 채우면 된다.

    배치 분할은 하지 않는다 — 텍스트 리스트를 통째로 _encode_raw() 에 넘기고
    분할은 백엔드 라이브러리에 맡긴다. sentence-transformers/FlagEmbedding 은
    입력을 길이순으로 정렬해 비슷한 길이끼리 묶어 패딩 낭비를 줄이는데, 미리
    잘라서 넘기면 그 정렬이 청크 내부로 제한돼 처리량이 떨어진다. 한 번에
    넘겨도 결과는 텐서 하나이므로 GPU->CPU 동기화 횟수는 어차피 1회다.

    Note:
        BaseEmbeddedModel 과 함께 상속해야 한다 — encode() 가 그쪽이 소유한
        self.batch_size 를 읽는다.
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

    Note:
        BaseEmbeddedModel 과 함께 상속해야 한다 — encode_sparse() 가 그쪽이
        소유한 self.batch_size 를 읽는다.
    """

    @property
    @abstractmethod
    def sparse_dimension(self) -> int:
        """sparse 벡터의 전체 차원 수(= 토크나이저 vocab 크기).

        encode_sparse() 는 등장한 토큰만 담은 희소 표현을 돌려주므로, 이를
        벡터 DB에 저장하려면 "전체 차원이 몇인지"가 따로 필요하다.
        (예: pgvector sparsevec 은 '{id:weight,...}/차원수' 형식을 받는다)

        값을 얻는 경로는 백엔드마다 다르다(tokenizer, config, 혹은 상수).
        그 차이를 구현체가 흡수해 이 프로퍼티 하나로 노출한다 — 호출부가
        백엔드 내부 구조를 알지 않아도 되도록.
        """

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
