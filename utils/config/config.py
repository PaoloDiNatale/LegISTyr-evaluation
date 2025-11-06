from contextvars import ContextVar

_LANG = ContextVar("LANG", default="de")

def set_lang(value: str) -> None:
    if value not in ("de", "it"):
        raise ValueError("Unsupported language. Choose 'de' or 'it'.")
    _LANG.set(value)

def get_lang() -> str:
    return _LANG.get()
