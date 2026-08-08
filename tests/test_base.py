"""능력 믹스인의 계약 검증 — 실제 모델/GPU 없이 더미 백엔드로 확인한다.

여기서 검증하는 것이 이 패키지의 핵심 가치다: 구현체는 _encode_raw() 만
채우면 정규화·접두사·빈 입력 방어·배치 위임을 공통으로 얻는다.
"""
import numpy as np
import pytest

from embedded import BaseEmbeddedModel, DenseCapable, SparseCapable
from embedded.base import DEFAULT_BATCH_SIZE


class DummyDense(BaseEmbeddedModel, DenseCapable):
    """호출 인자를 기록하는 더미 dense 백엔드."""

    DIM = 4

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model = object()          # unload 대상
        self.calls: list[tuple[list[str], int]] = []

    @property
    def model_name(self) -> str:
        return "dummy-dense"

    @property
    def dimension(self) -> int:
        return self.DIM

    def _encode_raw(self, texts, batch_size):
        self.calls.append((list(texts), batch_size))
        # 문장 길이에 비례한 값 -> 정규화 전/후를 구분할 수 있게
        return np.array([[float(len(t))] * self.DIM for t in texts], dtype=np.float32)


class DummySparse(BaseEmbeddedModel, SparseCapable):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls: list[tuple[list[str], int]] = []

    @property
    def model_name(self) -> str:
        return "dummy-sparse"

    def _encode_sparse_raw(self, texts, batch_size):
        self.calls.append((list(texts), batch_size))
        return [{1: 1.0, 2: 0.5} for _ in texts]


# ---- dense: 정규화 ---------------------------------------------------------

def test_normalize_true_gives_unit_norm():
    m = DummyDense()
    vecs = m.encode(["a", "bbb"])
    assert vecs.shape == (2, DummyDense.DIM)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0)


def test_normalize_false_keeps_magnitude():
    m = DummyDense()
    m.normalize = False
    vecs = m.encode(["a", "bbb"])
    # 원본은 길이(1, 3)를 4번 반복 -> norm 은 1이 아니어야 한다
    assert not np.allclose(np.linalg.norm(vecs, axis=1), 1.0)


def test_zero_vector_does_not_produce_nan():
    """빈 문자열 -> 전부 0 벡터. 0으로 나누지 않고 그대로 통과해야 한다."""
    m = DummyDense()
    vecs = m.encode([""])
    assert not np.isnan(vecs).any()


def test_dtype_is_float32():
    assert DummyDense().encode(["a"]).dtype == np.float32


# ---- dense: 빈 입력 --------------------------------------------------------

def test_empty_input_returns_empty_array_without_calling_backend():
    m = DummyDense()
    vecs = m.encode([])
    assert vecs.shape == (0, DummyDense.DIM)
    assert vecs.dtype == np.float32
    assert m.calls == []          # 백엔드를 아예 호출하지 않아야 한다


@pytest.mark.parametrize("method", ["encode_queries", "encode_documents"])
def test_empty_input_via_prefix_methods(method):
    m = DummyDense()
    assert getattr(m, method)([]).shape == (0, DummyDense.DIM)


# ---- dense: 접두사 ---------------------------------------------------------

def test_prefixes_are_applied():
    m = DummyDense()
    m.query_prefix = "query: "
    m.passage_prefix = "passage: "
    m.encode_queries(["q"])
    m.encode_documents(["d"])
    assert m.calls[0][0] == ["query: q"]
    assert m.calls[1][0] == ["passage: d"]


def test_no_prefix_by_default():
    m = DummyDense()
    m.encode_queries(["q"])
    assert m.calls[0][0] == ["q"]


# ---- 배치: 분할하지 않고 통째로 위임 ---------------------------------------

def test_texts_passed_whole_with_batch_size():
    """base 가 나누지 않고 전체를 넘기며 batch_size 를 전달해야 한다."""
    m = DummyDense(batch_size=2)
    texts = [f"t{i}" for i in range(5)]
    m.encode(texts)
    assert len(m.calls) == 1              # 한 번만 호출 (분할 없음)
    assert m.calls[0] == (texts, 2)       # 전체 텍스트 + batch_size 전달


def test_per_call_batch_size_overrides_instance_default():
    m = DummyDense(batch_size=2)
    m.encode(["a"], batch_size=7)
    assert m.calls[0][1] == 7


def test_default_batch_size():
    assert DummyDense().batch_size == DEFAULT_BATCH_SIZE


# ---- 믹스인 조립 계약 ------------------------------------------------------
# 공통 인자(batch_size)는 BaseEmbeddedModel 이 소유한다. 능력 믹스인은 그것을
# 읽기만 하므로 반드시 Base 와 함께 조립해야 한다.

def test_batch_size_is_owned_by_base_not_mixins():
    assert BaseEmbeddedModel.batch_size == DEFAULT_BATCH_SIZE
    assert "batch_size" not in vars(DenseCapable)
    assert "batch_size" not in vars(SparseCapable)


def test_base_constructor_sets_batch_size():
    assert DummyDense(batch_size=5).batch_size == 5


# ---- sparse ----------------------------------------------------------------

def test_sparse_empty_input_skips_backend():
    m = DummySparse()
    assert m.encode_sparse([]) == []
    assert m.calls == []


def test_sparse_passes_texts_whole_with_batch_size():
    m = DummySparse(batch_size=3)
    texts = ["a", "b", "c", "d"]
    m.encode_sparse(texts)
    assert m.calls == [(texts, 3)]


def test_sparse_per_call_batch_size():
    m = DummySparse(batch_size=3)
    m.encode_sparse(["a"], batch_size=9)
    assert m.calls[0][1] == 9


# ---- 능력 분기 (isinstance) ------------------------------------------------

def test_capability_dispatch():
    dense, sparse = DummyDense(), DummySparse()
    assert isinstance(dense, DenseCapable) and not isinstance(dense, SparseCapable)
    assert isinstance(sparse, SparseCapable) and not isinstance(sparse, DenseCapable)


def test_sparse_only_model_has_no_dense_api():
    """dense 를 구현하지 않은 모델에 encode() 가 생기지 않아야 한다."""
    assert not hasattr(DummySparse(), "encode")


def test_base_is_abstract():
    with pytest.raises(TypeError):
        BaseEmbeddedModel()


def test_dense_capable_requires_encode_raw():
    class Incomplete(BaseEmbeddedModel, DenseCapable):
        @property
        def model_name(self):
            return "x"

        @property
        def dimension(self):
            return 1
        # _encode_raw 미구현

    with pytest.raises(TypeError):
        Incomplete()


# ---- unload ----------------------------------------------------------------

def test_unload_releases_model_and_is_idempotent():
    m = DummyDense()
    m.unload()
    m.unload()                    # 두 번 호출해도 예외 없음
    assert m._model is None


def test_unload_can_be_overridden_for_other_attribute_names():
    """_model 이 아닌 이름을 쓰는 구현체는 unload() 를 오버라이드한다."""

    class CustomRelease(BaseEmbeddedModel, DenseCapable):
        def __init__(self):
            super().__init__()
            self._engine = object()

        @property
        def model_name(self):
            return "custom"

        @property
        def dimension(self):
            return 1

        def _encode_raw(self, texts, batch_size):
            return np.ones((len(texts), 1), dtype=np.float32)

        def unload(self):
            self._engine = None
            super().unload()

    m = CustomRelease()
    m.unload()
    assert m._engine is None


def test_context_manager_not_supported():
    """unload 는 의도적 호출만 허용한다 — with 문은 지원하지 않는다."""
    with pytest.raises(TypeError):
        with DummyDense():
            pass
