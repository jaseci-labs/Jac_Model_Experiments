def is_lib(fn: str, osname: str) -> bool:
    if osname == "Darwin":
        return fn.startswith("libraylib") and fn.endswith(".dylib")
    return fn.startswith("libraylib.so")
