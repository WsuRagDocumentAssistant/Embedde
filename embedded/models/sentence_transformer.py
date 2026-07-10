from typing import List, Optional

from ..base import BaseEmbeddedModel
from ..hf_utils import resolve_model_path
from ..registry import register, register_preset


@register("sentence-transformer")
class SentenceTransformerModel(BaseEmbeddedModel):
    """sentence-transformers로 로드 가능한 모든 HF 모델을 커버하는 범용 백엔드.

    e5, bge, ko-sroberta 등 대부분의 dense 임베딩 모델은
    이 클래스 하나 + 아래 프리셋(모델 ID/접두사 설정)으로 처리한다.
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        local_dir: Optional[str] = None,
        query_prefix: str = "",
        passage_prefix: str = "",
        normalize: bool = True,
        batch_size: int = 12,
        **hub_kwargs,
    ):
        super().__init__(normalize=normalize, batch_size=batch_size)
        from sentence_transformers import SentenceTransformer  # 지연 import: 미설치 환경에서도 패키지 로드는 가능해야 함

        self._name = model_name
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        path = resolve_model_path(model_name, local_dir, **hub_kwargs)
        self._model = SentenceTransformer(path, device=device)

    @property
    def model_name(self) -> str:
        return self._name

    @property
    def dimension(self) -> int:
        # sentence-transformers 최신 버전에서 메서드명이 변경됨
        getter = getattr(self._model, "get_embedding_dimension", None)
        if getter is None:
            getter = self._model.get_sentence_embedding_dimension
        return getter()

    def _encode_raw(self, texts: List[str]):
        return self._model.encode(
            texts,
            batch_size=len(texts),        # 배치 분할은 base.encode()가 담당
            convert_to_tensor=True,       # GPU 텐서 그대로 반환 (CPU 변환은 base가 마지막에 한 번)
            normalize_embeddings=False,   # 정규화도 base가 담당
            show_progress_bar=False,
        )


# ---- 모델별 프리셋: 같은 백엔드, 설정만 다름 ------------------------------

register_preset(
    "multilingual-e5-large", "sentence-transformer",
    model_name="intfloat/multilingual-e5-large",
    query_prefix="query: ",
    passage_prefix="passage: ",
)

register_preset(
    "multilingual-e5-base", "sentence-transformer",
    model_name="intfloat/multilingual-e5-base",
    query_prefix="query: ",
    passage_prefix="passage: ",
)

register_preset(
    "ko-sroberta", "sentence-transformer",
    model_name="jhgan/ko-sroberta-multitask",
)

register_preset(
    "minilm-l6", "sentence-transformer",
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)
