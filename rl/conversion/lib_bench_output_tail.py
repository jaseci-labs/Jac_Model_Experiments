def output_tail(out: str) -> str:
    return "\n".join(out.strip().splitlines()[-8:])
