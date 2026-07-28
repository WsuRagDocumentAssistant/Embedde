from typing import Dict, List, Optional

from ..base import BaseEmbeddedModel, DenseCapable, SparseCapable
from ..hf_utils import resolve_model_path
from ..registry import register


@register("bge-m3")
class BGEM3Model(BaseEmbeddedModel, DenseCapable, SparseCapable):
    """BGE-M3: dense + sparse(lexical) 동시 지원. FlagEmbedding 필요."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: Optional[str] = None,
        local_dir: Optional[str] = None,
        use_fp16: bool = True,
        normalize: bool = True,
        batch_size: int = 12,
        **hub_kwargs,
    ):
        super().__init__(batch_size=batch_size)
        from FlagEmbedding import BGEM3FlagModel  # 지연 import

        self._name = model_name
        self.normalize = normalize            # DenseCapable 클래스 기본값을 인스턴스 값으로 덮어씀
        path = resolve_model_path(model_name, local_dir, **hub_kwargs)
        self._model = BGEM3FlagModel(path, use_fp16=use_fp16, device=device)

    @property
    def model_name(self) -> str:
        return self._name

    @property
    def dimension(self) -> int:
        try:
            return self._model.model.config.hidden_size
        except AttributeError:
            return 1024  # bge-m3 dense 기본 차원

    def _encode_raw(self, texts: List[str]):
        out = self._model.encode(
            texts,
            batch_size=len(texts),  # 배치 분할은 base.encode()가 담당
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return out["dense_vecs"]  # numpy (FlagEmbedding이 내부에서 CPU로 반환)

    def _encode_sparse_raw(self, texts: List[str]) -> List[Dict[int, float]]:
        # 빈 입력 방어는 SparseCapable.encode_sparse()가 이미 처리 (여기는 항상 non-empty)
        out = self._model.encode(
            texts,
            batch_size=self.batch_size,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        # lexical_weights: 문장마다 {token_id(str): weight} — 키를 int로 통일
        return [
            {int(tok): float(w) for tok, w in weights.items()}
            for weights in out["lexical_weights"]
        ]
