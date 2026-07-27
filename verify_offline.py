"""오프라인 로딩 검증.

네트워크를 차단한 채(HF_HUB_OFFLINE) 로컬 폴더로만 bge-m3를 로드하고
dense + sparse 인코딩이 둘 다 나오는지 확인한다. 서버 반입 전 리허설.
"""
import os

# import 전에 오프라인 모드 강제 — 서버 환경을 흉내
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np

from embedded import SparseCapable, create_model

LOCAL_PATH = "models/bge-m3"


def main() -> None:
    print(f"오프라인 모드로 로드: {LOCAL_PATH}")
    model = create_model("bge-m3", model_name=LOCAL_PATH)
    print(f"model_name={model.model_name}, dimension={model.dimension}")
    assert isinstance(model, SparseCapable)

    docs = ["임베딩 모델을 오프라인 서버에 배포한다.", "오늘 점심은 김치찌개."]
    query = ["오프라인 배포 방법"]

    dvecs = model.encode_documents(docs)
    qvec = model.encode_queries(query)
    sims = (dvecs @ qvec.T)[:, 0]
    print(f"\n[dense] shape={dvecs.shape}, norm={np.linalg.norm(dvecs, axis=1).round(3)}")
    for d, s in zip(docs, sims):
        print(f"  sim={s:.4f}  {d}")

    sparse = model.encode_sparse(docs)
    print(f"\n[sparse] 문장 수={len(sparse)}, 첫 문장 토큰 수={len(sparse[0])}")
    top = sorted(sparse[0].items(), key=lambda kv: kv[1], reverse=True)[:5]
    print(f"  상위 토큰(id:weight): {[(t, round(w, 3)) for t, w in top]}")

    print("\n검증 통과: 오프라인 로컬 폴더로 dense + sparse 모두 동작")


if __name__ == "__main__":
    main()
