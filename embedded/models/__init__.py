# 모델 클래스를 여기서 재수출한다 — 사용자는 필요한 클래스를 직접 생성한다.
# (레지스트리/이름 기반 팩토리는 두지 않는다: 모델 교체가 항상 코드 수정 +
#  재배포로 이루어지므로, 문자열 간접층보다 클래스 직접 생성이 더 단순하고
#  IDE 힌트도 그대로 받는다. 확장성은 BaseEmbeddedModel/DenseCapable/
#  SparseCapable 조합만으로 이미 충분하다.)
from .bge_m3 import BGEM3Model
from .sentence_transformer import SentenceTransformerModel

__all__ = ["BGEM3Model", "SentenceTransformerModel"]
