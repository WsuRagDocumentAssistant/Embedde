# EmbeddedModel

교체 가능하고 확장성 있는 임베딩 모델 추상화 패키지.

임베딩 모델(dense/sparse)을 **하나의 공통 인터페이스**로 감싸서, 호출부 코드를
바꾸지 않고 모델을 교체하거나 새 모델을 추가할 수 있게 한다. 외부 통신이 차단된
오프라인 서버 배포를 염두에 두고 설계했다.

## 설계 개요

능력(capability) 기반 조립 구조. 구체 모델은 자기가 가진 능력만 골라 다중 상속한다.

```
BaseEmbeddedModel   공통 뿌리 — model_name, batch_size, unload()/_release_model()
DenseCapable        dense 능력 — encode() / encode_queries() / encode_documents()
                              (구현체는 _encode_raw() + dimension 만 채움)
SparseCapable       sparse 능력 — encode_sparse()
                              (구현체는 _encode_sparse_raw() 만 채움)

조립 예:
  BGEM3Model(BaseEmbeddedModel, DenseCapable, SparseCapable)   # 둘 다
  SentenceTransformerModel(BaseEmbeddedModel, DenseCapable)    # dense만
  (SpladeModel(BaseEmbeddedModel, SparseCapable))              # sparse만
```

`DenseCapable`/`SparseCapable` 둘 다 템플릿 메서드 패턴이다 — GPU→CPU 변환,
정규화, 접두사 처리, 빈 입력 방어 같은 **공통 로직은 능력 클래스가 완성**해두고,
모델마다 다른 부분만 구현체가 채운다.

**배치 분할은 하지 않는다.** 텍스트를 통째로 백엔드에 넘기고 `batch_size`만
전달한다. sentence-transformers와 FlagEmbedding은 입력을 길이순으로 정렬해
패딩 낭비를 줄이는데, 미리 잘라서 넘기면 그 최적화가 청크 내부로 제한되어
처리량이 떨어진다. 한 번에 넘겨도 결과는 텐서 하나이므로 GPU→CPU 동기화
횟수는 어차피 1회다.

## 디렉터리 구조

```
embedded/
  __init__.py        공개 API 재수출 (BGEM3Model, SentenceTransformerModel 등)
  base.py            BaseEmbeddedModel / DenseCapable / SparseCapable
  hf_utils.py        폴더 검증(ensure_model_dir) / Hub 다운로드(download_model)
  models/
    __init__.py      구현체 재수출
    sentence_transformer.py   범용 dense 백엔드 (sentence-transformers)
    bge_m3.py                 dense + sparse 백엔드 (FlagEmbedding)
requirements.txt     의존 라이브러리 (버전 고정)
pyproject.toml       패키지 정의
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

## 테스트

더미 백엔드로 능력 믹스인의 계약을 검증한다 — GPU도 모델 다운로드도 필요 없다.

```bash
pip install pytest
python -m pytest tests/ -q
```

## 사용법

### 기본 제공 백엔드

클래스를 직접 생성한다. 어떤 모델을 쓸지는 `model_name`으로 지정한다.

- `BGEM3Model` — dense + sparse (FlagEmbedding)
- `SentenceTransformerModel` — 범용 dense 백엔드 (e5, ko-sroberta, MiniLM 등 모든 sentence-transformers 모델)

> **모델 클래스는 다운로드하지 않는다.** `model_path`에는 이미 받아둔 로컬 폴더
> 경로를 넘긴다. 경로가 없거나 불완전하면 Hub를 조회하지 않고
> `FileNotFoundError`로 즉시 실패한다 — 오프라인 배포에서 의도치 않은 네트워크
> 접근을 막기 위한 것이다. 가중치를 새로 받는 것은 `download_model()`의 몫이며
> 개발 PC에서만 쓴다([오프라인 서버 배포](#오프라인-서버-배포) 참고).
>
> `model_path` 외의 인자는 모두 키워드 전용이다(`*`). 12개 남짓한 설정을
> 순서로 넘기다 값이 뒤바뀌는 사고를 막는다.

```python
from embedded import BGEM3Model, SparseCapable

model = BGEM3Model("models/bge-m3")   # 이미 받아둔 폴더

doc_vecs = model.encode_documents(["문서1", "문서2"])   # (2, 1024) 정규화된 dense
query_vec = model.encode_queries(["검색어"])            # (1, 1024)

if isinstance(model, SparseCapable):
    sparse = model.encode_sparse(["문서1"])   # [{token_id: weight}, ...]
```

```python
from embedded import SentenceTransformerModel

model = SentenceTransformerModel(
    "models/e5-large",                                    # 이미 받아둔 폴더
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
    def _encode_raw(self, texts, batch_size):
        ...   # 텍스트 전체 -> 벡터. 정규화/CPU변환/접두사는 base가 담당
              # 배치 분할은 백엔드 라이브러리에 맡긴다(batch_size 를 그대로 전달)

model = MyModel()
```

`unload()`가 정리할 대상이 `self._model`이 아니라면 `_release_model()` 훅만
오버라이드하면 된다.

### 리소스 해제

파이썬은 GPU 메모리를 자동 반납하지 않으므로 명시적으로 내린다.
해제 시점은 사용자가 의도적으로 정한다 — 서버는 보통 모델을 프로세스 내내
상주시키고 종료 훅에서 한 번 호출한다.

```python
model = BGEM3Model("models/bge-m3")
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

`download_model()`이 Hub에서 가중치를 받아 지정한 폴더에 놓는다. 받을 폴더는
필수 인자다 — 가중치가 전역 HF 캐시에만 남으면 반입할 폴더를 만들 수 없기
때문이다. (가중치가 캐시에 중복 저장되지는 않는다 — 커밋 해시를 적은 1KB
포인터만 남는다.)

```python
from embedded.hf_utils import download_model

download_model("BAAI/bge-m3", "models/bge-m3")
```

`models/bge-m3/` 폴더(가중치·토크나이저·설정 일체)를 통째로 서버에 옮긴다.
`config.json`, `tokenizer.json`, `model.safetensors`(또는 `pytorch_model.bin`),
bge-m3면 `sparse_linear.pt`까지 빠짐없이 복사해야 한다.

**2) 서버에서 로컬 경로로 로딩**

```python
from embedded import BGEM3Model
model = BGEM3Model("/srv/models/bge-m3")   # 모델 클래스는 다운로드하지 않는다
```

로컬 폴더 경로를 주면 네트워크를 타지 않는다. 경로가 잘못되면 Hub로
폴백하지 않고 `FileNotFoundError`가 발생하므로 배포 실수를 즉시 알 수 있다.
실수 방지를 위해 서버 환경변수에 오프라인 모드를 박아두는 것을 권장한다.

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

**3) 검증**

서버에 반입한 뒤, 네트워크를 차단한 상태로 로컬 폴더에서 dense + sparse가
모두 나오는지 확인한다.

```python
import os
os.environ["HF_HUB_OFFLINE"] = "1"        # import 전에 설정
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from embedded import BGEM3Model

model = BGEM3Model("/srv/models/bge-m3")
print(model.dimension)                                  # 1024
print(model.encode_documents(["테스트"]).shape)          # (1, 1024)
print(len(model.encode_sparse(["테스트"])[0]))           # sparse 토큰 수 > 0
model.unload()
```

> **라이브러리 반입**: 오프라인 서버는 `pip install`도 안 되므로, 서버 OS/파이썬
> 버전에 맞는 wheel을 함께 반입하거나(예: `pip download -r requirements.txt`),
> 서버에 라이브러리 설치 경로가 따로 마련돼 있어야 한다.
