# import 자체가 레지스트리 등록을 트리거한다.
# 무거운 의존성(sentence_transformers, FlagEmbedding)은 각 클래스의
# __init__ 안에서 지연 import하므로, 미설치 환경에서도 여기는 안전하다.
from . import bge_m3  # noqa: F401
from . import sentence_transformer  # noqa: F401
