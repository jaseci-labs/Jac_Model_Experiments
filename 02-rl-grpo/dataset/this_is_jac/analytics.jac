"""This is Jac - the analytics microservice (the provider).

A standalone server module. The guestbook server `sv import`s this; the first
time it calls `analyze(...)` the runtime spawns THIS module as its own
`jac start` process and routes the call to it. `pid` is this process's own
os.getpid() - proof it is not the caller's process."""

import os;
import time;
import from typing { Generator }

obj AnalyticsReport {
    has pid: int,
        score: float,
        service: str = "analytics";
}

def:pub analyze(signatures: int) -> AnalyticsReport {
    # runs in analytics' OWN process - pid proves it
    return AnalyticsReport(pid=os.getpid(), score=float(signatures) * 1.5);
}

"""The streamed commentary - the demo here is the TRANSPORT. Each yielded
phrase leaves THIS process as one server-sent event, crosses the service
boundary into the gateway, and reaches the browser still streaming. No
buffering anywhere on the path."""
def:pub narrate(signatures: int) -> Generator {
    pid = os.getpid();
    plural = "s" if signatures != 1 else "";
    text = (
        f"Analytics speaking - process {pid}, a separate HTTP process, live over SSE. "
        f"The gateway just handed me {signatures} signature{plural} from a walk "
        "of the shared guestbook graph. Every word you are reading left my "
        "process as its own server-sent event, was re-streamed by the guestbook "
        "server, and rendered the moment it reached your browser. Two "
        "processes, one import, zero glue - and not a single line of HTTP "
        "client code anywhere."
    );
    def stream -> Generator[str, None, None] {
        for word in text.split(" ") {
            time.sleep(0.05);
            yield word + " ";
        }
    }
    report stream();
}
