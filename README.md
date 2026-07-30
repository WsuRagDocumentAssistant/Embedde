# EmbeddedModel

교체 가능하고 확장성 있는 임베딩 모델 추상화 패키지.

임베딩 모델(dense/sparse)을 **하나의 공통 인터페이스**로 감싸서, 호출부 코드를
바꾸지 않고 모델을 교체하거나 새 모델을 추가할 수 있게 한다. 외부 통신이 차단된
오프라인 서버 배포를 염두에 두고 설계했다.

## 설계 개요

능력(capability) 기반 조립 구조. 구체 모델은 자기가 가진 능력만 골라 다중 상속한다.

```
BaseEmbeddedModel   공통 뿌리 — model_name, batch_size, unload()
DenseCapable        dense 능력 — encode() / encode_queries() / encode_documents()
                              (구현체는 _encode_raw() + dimension 만 채움)
SparseCapable       sparse 능력 — encode_sparse()
                              (구현체는 _encode_sparse_raw() 만 채움)

조립 예:
  BGEM3Model(BaseEmbeddedModel, DenseCapable, SparseCapable)   # 둘 다
  SentenceTransformerModel(BaseEmbeddedModel, DenseCapable)    # dense만
  (SpladeModel(BaseEmbeddedModel, SparseCapable))              # sparse만
```

`DenseCapable`/`SparseCapable` 둘 다 템플릿 메서드 패턴이다 — 배치 분할, GPU→CPU
변환(마지막에 한 번), 정규화, 접두사 처리, 빈 입력 방어 같은 **공통 로직은
능력 클래스가 완성**해두고, 모델마다 다른 부분만 구현체가 채운다.

## 디렉터리 구조

```
embedded/
  __init__.py        공개 API 재수출 (BGEM3Model, SentenceTransformerModel 등)
  base.py            BaseEmbeddedModel / DenseCapable / SparseCapable
  hf_utils.py        HF Hub 경로 해석 (resolve_model_path)
  models/
    __init__.py      구현체 재수출
    sentence_transformer.py   범용 dense 백엔드 (sentence-transformers)
    bge_m3.py                 dense + sparse 백엔드 (FlagEmbedding)
main.py              스모크 테스트
verify_offline.py    오프라인 로딩 검증
requirements.txt     의존 라이브러리 (버전 고정)
```

> 이름 기반 팩토리(레지스트리)는 두지 않는다. 모델 교체가 항상 "서버 내리고
> 코드/설정 수정 후 재배포"로 이루어지고, 모델마다 필요한 생성자 인자가 원래
> 다르므로(예: `use_fp16` vs `query_prefix`) 문자열 이름으로 감싸도 사용자가
> 알아야 할 정보가 줄지 않는다. 대신 클래스를 직접 생성해 IDE 힌트를 그대로
> 받는다. 실제 확장성은 `DenseCapable`/`SparseCapable` 능력 조합에서 나온다.

## 설치

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows
pip install -r requirements.txt
```

`sentence-transformers`(dense), `FlagEmbedding`(bge-m3 sparse)이 핵심 의존성이며
torch·transformers·numpy를 함께 끌어온다.

## 사용법

### 기본 제공 백엔드

클래스를 직접 생성한다. 어떤 모델을 쓸지는 `model_name`으로 지정한다.

- `BGEM3Model` — dense + sparse (FlagEmbedding), 기본값 `BAAI/bge-m3`
- `SentenceTransformerModel` — 범용 dense 백엔드 (e5, ko-sroberta, MiniLM 등 모든 sentence-transformers 모델)

```python
from embedded import BGEM3Model, SparseCapable

model = BGEM3Model(model_name="BAAI/bge-m3")   # 또는 로컬 경로

doc_vecs = model.encode_documents(["문서1", "문서2"])   # (2, 1024) 정규화된 dense
query_vec = model.encode_queries(["검색어"])            # (1, 1024)

if isinstance(model, SparseCapable):
    sparse = model.encode_sparse(["문서1"])   # [{token_id: weight}, ...]
```

```python
from embedded import SentenceTransformerModel

model = SentenceTransformerModel(
    model_name="intfloat/multilingual-e5-large",
    query_prefix="query: ", passage_prefix="passage: ",   # 모델별로 다름, 아래 참고
)
```

> ⚠️ **접두사 주의**: 일부 모델은 query/passage 접두사가 있어야 검색 성능이 나온다.
> 빠뜨려도 에러 없이 조용히 품질만 떨어지므로 생성 시 함께 지정할 것.
> - `intfloat/multilingual-e5-*` → `query_prefix="query: "`, `passage_prefix="passage: "`
> - `jhgan/ko-sroberta-multitask`, `all-MiniLM-*` → 접두사 불필요

### 새 백엔드 클래스 추가

`sentence-transformers`/`FlagEmbedding`으로 안 되는 백엔드(예: 외부 API, ONNX)는
능력을 상속하고 빈칸만 채우면 된다. 등록 절차는 없다 — 정의하고 바로 생성해서 쓴다.

```python
from embedded import BaseEmbeddedModel, DenseCapable

class MyModel(BaseEmbeddedModel, DenseCapable):
    @property
    def model_name(self): return "my-backend"
    @property
    def dimension(self): return 768
    def _encode_raw(self, texts):
        ...   # 배치 하나 -> 벡터. 배치 분할/정규화/CPU변환은 base가 담당

model = MyModel()
```

### 리소스 해제

파이썬은 GPU 메모리를 자동 반납하지 않으므로 명시적으로 내린다.
해제 시점은 사용자가 의도적으로 정한다 — 서버는 보통 모델을 프로세스 내내
상주시키고 종료 훅에서 한 번 호출한다.

```python
model = BGEM3Model(model_name="...")
try:
    vecs = model.encode_documents(docs)
finally:
    model.unload()      # VRAM 반납 (여러 번 호출해도 안전)
```

컨텍스트 매니저(`with`)는 일부러 지원하지 않는다. 모델은 프로세스 수명 동안
상주하는 자원이므로, 블록을 벗어날 때 자동으로 내려가는 동작이 실제 사용
패턴과 맞지 않고 의도치 않은 조기 해제를 유발할 수 있다.

## 오프라인 서버 배포

외부 통신이 차단된 서버는 아래 흐름으로 배포한다.

```
[인터넷 O 개발 PC]        [USB/내부망]        [오프라인 서버]
가중치 폴더 다운로드   →   폴더째 복사    →   로컬 경로로 로딩
```

**1) 개발 PC에서 가중치 받기**

```python
from embedded.hf_utils import resolve_model_path
resolve_model_path("BAAI/bge-m3", local_dir="models/bge-m3")
```

`models/bge-m3/` 폴더(가중치·토크나이저·설정 일체)를 통째로 서버에 옮긴다.
`config.json`, `tokenizer.json`, `model.safetensors`(또는 `pytorch_model.bin`),
bge-m3면 `sparse_linear.pt`까지 빠짐없이 복사해야 한다.

**2) 서버에서 로컬 경로로 로딩**

```python
from embedded import BGEM3Model
model = BGEM3Model(model_name="/srv/models/bge-m3")
```

레포 ID가 아니라 로컬 폴더 경로를 주면 네트워크를 타지 않는다.
실수 방지를 위해 서버 환경변수에 오프라인 모드를 박아두는 것을 권장한다.

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

**3) 검증**

`verify_offline.py`가 네트워크를 차단한 채 로컬 폴더로 dense + sparse가
나오는지 확인한다.

```bash
python verify_offline.py
```

> **라이브러리 반입**: 오프라인 서버는 `pip install`도 안 되므로, 서버 OS/파이썬
> 버전에 맞는 wheel을 함께 반입하거나(예: `pip download -r requirements.txt`),
> 서버에 라이브러리 설치 경로가 따로 마련돼 있어야 한다.
