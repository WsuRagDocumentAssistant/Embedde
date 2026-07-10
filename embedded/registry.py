"""모델 이름 문자열 -> 인스턴스 팩토리.

호출부는 create_model("이름") 한 줄만 쓰고, 모델 교체는 설정 문자열 변경으로 끝낸다.

등록 방법 두 가지:
  1. @register("이름", **기본인자)   — 새 백엔드 클래스를 만들 때
  2. register_preset("이름", "기존등록명", **오버라이드)
     — 같은 백엔드에 설정(모델 ID, 접두사 등)만 다른 변형을 추가할 때
"""
from typing import TYPE_CHECKING, Dict, List, Tuple

if TYPE_CHECKING:
    from .base import BaseEmbeddedModel

_REGISTRY: Dict[str, Tuple[type, dict]] = {}


def register(name: str, **defaults):
    """클래스 데코레이터. defaults는 create_model 시 기본 생성자 인자가 된다."""

    def deco(cls):
        _REGISTRY[name] = (cls, defaults)
        return cls

    return deco


def register_preset(name: str, base: str, **overrides) -> None:
    """이미 등록된 백엔드에 설정만 바꾼 별칭을 추가한다."""
    if base not in _REGISTRY:
        raise KeyError(f"기반 모델이 등록되어 있지 않음: {base}")
    cls, defaults = _REGISTRY[base]
    _REGISTRY[name] = (cls, {**defaults, **overrides})


def create_model(name: str, **kwargs) -> "BaseEmbeddedModel":
    """등록된 이름으로 모델 인스턴스를 생성한다. kwargs가 프리셋 기본값을 덮어쓴다."""
    if name not in _REGISTRY:
        raise KeyError(
            f"등록되지 않은 모델: {name!r} (사용 가능: {available_models()})"
        )
    cls, defaults = _REGISTRY[name]
    return cls(**{**defaults, **kwargs})


def available_models() -> List[str]:
    return sorted(_REGISTRY)
