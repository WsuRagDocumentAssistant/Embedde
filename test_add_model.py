"""확장성 검증 데모.

embedded/ 패키지 코드를 한 줄도 수정하지 않고,
  방법 1) 기존 백엔드에 프리셋 추가        -> register_preset() 한 줄
  방법 2) 완전히 새로운 백엔드 클래스 추가  -> _encode_raw()만 구현
두 방식 모두 기존 호출 경로(create_model -> encode_documents/queries)에
그대로 올라타는 것을 확인한다.
"""
import numpy as np

from embedded import (
    BaseEmbeddedModel,
    available_models,
    create_model,
    register,
    register_preset,
)

# ---- 방법 1: 프리셋 추가 (한 줄) -----------------------------------------
register_preset(
    "minilm-l12", "sentence-transformer",
    model_name="sentence-transformers/all-MiniLM-L12-v2",
)


# ---- 방법 2: 새 백엔드 클래스 추가 ----------------------------------------
# 데모용 해시 기반 임베더. 실제로는 OpenAI API, ONNX 런타임 등
# 어떤 백엔드든 _encode_raw()만 맞추면 동일하게 동작한다.
@register("hash-demo", dim=64)
class HashEmbedder(BaseEmbeddedModel):

    def __init__(self, dim: int = 64, normalize: bool = True, batch_size: int = 12):
        super().__init__(normalize=normalize, batch_size=batch_size)
        self._dim = dim

    @property
    def model_name(self) -> str:
        return f"hash-demo-{self._dim}"

    @property
    def dimension(self) -> int:
        return self._dim

    def _encode_raw(self, texts):
        import hashlib

        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in text.split():
                h = int(hashlib.md5(token.encode()).hexdigest(), 16)
                out[i, h % self._dim] += 1.0
        return out


# ---- 기존 호출 경로에 그대로 태워서 검증 -----------------------------------
def main() -> None:
    print("등록된 모델:", available_models())

    docs = [
        "임베딩 모델을 교체 가능하고 확장성 있게 설계한다.",
        "오늘 점심은 김치찌개를 먹었다.",
    ]
    query = ["임베딩 아키텍처 설계 방법"]

    for name in ["minilm-l12", "hash-demo"]:
        model = create_model(name)
        doc_vecs = model.encode_documents(docs)
        query_vec = model.encode_queries(query)
        sims = (doc_vecs @ query_vec.T)[:, 0]
        norms = np.linalg.norm(doc_vecs, axis=1)
        print(
            f"\n[{name}] dim={model.dimension}, shape={doc_vecs.shape}, "
            f"norm={norms.round(4)}, sims={sims.round(4)}"
        )
        assert doc_vecs.shape == (len(docs), model.dimension)
        assert np.allclose(norms, 1.0)

    print("\n검증 통과: 패키지 수정 없이 두 방식 모두 동일 호출 경로로 동작")


if __name__ == "__main__":
    main()
