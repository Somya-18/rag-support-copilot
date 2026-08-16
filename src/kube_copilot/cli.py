def main() -> None:
    from . import models as _models  # noqa: F401 — registers tables to Base.metadata
    from .db import ensure_schema
    ensure_schema()
    print("Schema initialized.")


if __name__ == "__main__":
    main()
