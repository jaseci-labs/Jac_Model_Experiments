import json

def _parse_bench_json(text: str):
    marker = "BENCH_JSON "
    for line in text.splitlines():
        s = line.strip()
        if s.startswith(marker):
            try:
                return json.loads(s[len(marker):])
            except Exception as e:
                return None
    return None
