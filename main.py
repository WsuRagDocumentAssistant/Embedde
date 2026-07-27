"""임베딩 모델 교체 스모크 테스트.

레지스트리에 등록된 이름만 바꿔서 같은 코드 경로로 여러 모델을 돌려본다.
"""
import numpy as np

from embedded import SparseCapable, available_models, create_model


def run_model(name: str, **kwargs) -> None:
    print(f"\n=== {name} ===")
    model = create_model(name, **kwargs)
    print(f"model_name: {model.model_name}, dimension: {model.dimension}")

    docs = [
        "임베딩 모델을 교체 가능하고 확장성 있게 설계한다.",
        "오늘 점심은 김치찌개를 먹었다.",
    ]
    doc_vecs = model.encode_documents(docs)
    query_vec = model.encode_queries(["임베딩 아키텍처 설계 방법"])

    sims = doc_vecs @ query_vec.T  # normalize=True이므로 내적 = 코사인 유사도
    print(f"doc_vecs: {doc_vecs.shape}, norm: {np.linalg.norm(doc_vecs, axis=1)}")
    for doc, sim in zip(docs, sims[:, 0]):
        print(f"  sim={sim:.4f}  {doc}")

    if isinstance(model, SparseCapable):
        sparse = model.encode_sparse(docs[:1])
        print(f"sparse 지원: 첫 문장 토큰 {len(sparse[0])}개")


def main() -> None:
    print("등록된 모델:", available_models())

    # 교체 = 이름 문자열 변경이 전부
    # run_model("minilm-l6")
    # run_model("ko-sroberta")
    # run_model("multilingual-e5-large")
    run_model("bge-m3")  # FlagEmbedding 설치 필요, dense + sparse


if __name__ == "__main__":
    main()
