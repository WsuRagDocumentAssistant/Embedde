import logging
import time
from typing import Dict, List, Optional

from ..base import BaseEmbeddedModel, DEFAULT_BATCH_SIZE, DenseCapable, SparseCapable
from ..hf_utils import ensure_model_dir

logger = logging.getLogger(__name__)


class BGEM3Model(BaseEmbeddedModel, DenseCapable, SparseCapable):
    """BGE-M3: dense + sparse(lexical) 동시 지원. FlagEmbedding 필요.

    이미 받아둔 로컬 폴더만 로드한다 — 다운로드는 하지 않는다. 경로가 없거나
    불완전하면 FileNotFoundError 로 즉시 실패한다(FlagEmbedding 이 경로를 Hub
    레포 ID 로 오인해 조용히 받아오는 것을 막기 위함).
    개발 PC 에서 가중치를 받으려면 embedded.hf_utils.download_model() 을 쓴다.

        model = BGEM3Model("/srv/models/bge-m3", device="cuda")

    Args:
        model_path: 이미 받아둔 모델 폴더 경로. (위치 인자로 줄 수 있다)
        device: 연산 장치("cuda", "cuda:1", "cpu"). None이면 라이브러리가
            자동 선택(GPU 있으면 GPU).
        use_fp16: 16비트로 로드. GPU에서 메모리 절반 + 속도 향상.
            CPU에서는 이점이 없으므로 False 권장.
        normalize: dense 벡터 L2 정규화. True면 내적이 곧 코사인 유사도.
            (FlagEmbedding 자체 정규화는 끄고 base 에서 일괄 처리하므로
            False 를 주면 실제로 정규화되지 않은 벡터가 나온다.)
        batch_size: 백엔드에 전달할 배치 크기. GPU 메모리 여유에 맞춰 조절.
        passage_max_length: 문서 토큰 상한. 짧은 문서만 다루면 줄여서 VRAM 절약.
        query_max_length: 쿼리 토큰 상한.

    Note:
        model_name 속성은 로드한 폴더 경로를 돌려준다.

    Raises:
        FileNotFoundError: 경로가 없거나 모델 폴더로서 불완전할 때.
    """

    def __init__(
        self,
        model_path: str,
        *,
        device: Optional[str] = None,
        use_fp16: bool = True,
        normalize: bool = True,
        batch_size: int = DEFAULT_BATCH_SIZE,
        passage_max_length: int = 512,
        query_max_length: int = 512,
    ):
        super().__init__(batch_size=batch_size)
        from FlagEmbedding import BGEM3FlagModel  # 지연 import

        path = ensure_model_dir(model_path)
        self._name = path
        self.normalize = normalize            # DenseCapable 클래스 기본값을 인스턴스 값으로 덮어씀
        logger.info("모델 로딩 시작: %s", path)
        start = time.time()
        self._model = BGEM3FlagModel(
            path,
            use_fp16=use_fp16,
            # FlagEmbedding 1.3+ 는 인자명이 devices(복수)다. device= 로 주면
            # **kwargs 로 흘러 조용히 무시되므로(GPU 지정이 안 먹음) 주의.
            devices=device,
            # 정규화는 base(DenseCapable)가 담당하므로 여기서는 끈다.
            # 켜두면 normalize=False 를 줘도 정규화된 벡터가 나온다.
            normalize_embeddings=False,
            passage_max_length=passage_max_length,
            query_max_length=query_max_length,
        )
        logger.info("모델 로딩 완료: %s (%.1fs)", path, time.time() - start)

    def _require_model(self) -> None:
        if self._model is None:
            raise RuntimeError(
                f"모델이 이미 unload 되었습니다: {self._name!r}. "
                "다시 사용하려면 인스턴스를 새로 생성하세요."
            )

    @property
    def model_name(self) -> str:
        return self._name

    @property
    def device(self) -> str:
        """가중치가 실제로 올라간 연산 장치("cpu", "cuda:0" 등).

        생성 시 넘긴 device 인자를 그대로 돌려주는 게 아니라 실측한다 — 이
        구분이 중요한 이유가 두 가지 있었다: ① FlagEmbedding 1.3+ 는 device=
        인자명이 devices(복수)로 바뀌어, device= 로 주면 조용히 무시된 채
        CPU로 폴백해도 겉으로는 에러가 없었다. ② FlagEmbedding 은 .to(device)
        를 __init__ 이 아니라 encode() 호출 시점에 수행하므로, encode 를
        부르기 전에 확인하면 아직 이동 전이라 항상 cpu로 보인다.
        """
        self._require_model()
        return str(next(self._model.model.parameters()).device)

    @property
    def dimension(self) -> int:
        self._require_model()
        try:
            return self._model.model.config.hidden_size
        except AttributeError:
            # FlagEmbedding 내부 구조가 바뀐 경우의 폴백 (bge-m3 dense 기본 차원).
            # 조용히 넘기면 잘못된 차원으로 벡터 DB 컬렉션을 만들 수 있어 경고를 남긴다.
            logger.warning(
                "FlagEmbedding 내부 구조에서 hidden_size 를 읽지 못해 기본값 1024 를 "
                "사용합니다. 라이브러리 버전 변경으로 실제 차원과 다를 수 있습니다: %s",
                self._name,
            )
            return 1024

    @property
    def sparse_dimension(self) -> int:
        self._require_model()
        # bge-m3 의 sparse 는 vocab 전체를 차원으로 갖는 희소 표현이다.
        return self._model.tokenizer.vocab_size

    def _encode_raw(self, texts: List[str], batch_size: int):
        # 배치 분할은 FlagEmbedding 에 맡긴다(길이순 정렬로 패딩 낭비를 줄임).
        out = self._model.encode(
            texts,
            batch_size=batch_size,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return out["dense_vecs"]  # numpy (FlagEmbedding이 내부에서 CPU로 반환)

    def _encode_sparse_raw(
        self, texts: List[str], batch_size: int
    ) -> List[Dict[int, float]]:
        # 빈 입력 방어는 SparseCapable.encode_sparse()가 이미 처리 (여기는 항상 non-empty)
        out = self._model.encode(
            texts,
            batch_size=batch_size,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        # lexical_weights: 문장마다 {token_id(str): weight} — 키를 int로 통일
        return [
            {int(tok): float(w) for tok, w in weights.items()}
            for weights in out["lexical_weights"]
        ]
