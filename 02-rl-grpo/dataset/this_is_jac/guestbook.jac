"""This is Jac - the live object-spatial backend.

Two demos run against a real persistent graph:

  - **Guestbook** - every signature is a `Visitor` node grouped under a `Day`
    node for the date it was signed: `root.shared -> Day -> Visitor`. ALL
    signatures live on `root.shared` (the public graph anonymous requests run
    on), so any user - logged in or not - signs and reads the SAME guestbook
    with one walk: no `allroots()` fan-out. Data lives in the graph; it
    survives restarts.
  - **Graph traversal** - `explore_graph` walks the REAL shared graph,
    crossing each `Day` and the `Visitor`s under it, recording the order it
    crossed them, so the client draws the actual live topology (no demo data).

Walkers are the mobile agents the client `root spawn`s. Each declares where it
starts (`with Root entry`) and how it reacts at each node type (`with Visitor
entry`). Traversal is always `visit`, never a hand-rolled loop.
"""

import datetime;
import os;
import from typing { Generator }

# Microservice boundary: one line. This module already serves its walkers to the
# browser (cl->sv); this relative `sv import` ALSO makes it a server->server
# consumer. The runtime spawns `analytics` as its own process the first time
# `analyze(...)` is called and routes to it.
sv import from .analytics { analyze, AnalyticsReport, narrate }

def _now -> str {
    return datetime.datetime.now(datetime.UTC).replace(tzinfo=None).isoformat() + "Z";
}

def _today -> str {
    return datetime.datetime.now(datetime.UTC).date().isoformat();
}

# ─── View models: the typed contract handed back to the client ───────────────
obj GuestEntry {
    has id: str,
        name: str,
        message: str,
        created_at: str;
}

obj GraphNodeView {
    has id: str,
        label: str,
        kind: str;
}

obj GraphSnapshot {
    has nodes: list[GraphNodeView],
        edges: list[list[str]],
        path: list[str],
        log: list[str];
}

# ─── Edges: the relationships in the graph ───────────────────────────────────
edge OnDay {}  # root -> Day



edge Signed {}  # Day -> Visitor



# ─── Nodes: data plus the method that presents it as a view ──────────────────
# Signatures are grouped under the day they were signed.
node Day {
    has date: str = "";
}

node Visitor {
    has name: str = "",
        message: str = "",
        created_at: str = "";

    def to_view -> GuestEntry {
        return GuestEntry(
            id=jid(self),
            name=self.name,
            message=self.message,
            created_at=self.created_at
        );
    }
}

# ─── Guestbook walkers ───────────────────────────────────────────────────────
walker:pub sign_guestbook {
    has name: str = "",
        message: str = "",
        reports: list[GuestEntry] = [];

    can run with Root entry {
        if not self.message.strip() {
            return;
        }
        # Every signature hangs off the SHARED root - the commons every
        # user can read and attach to by default, logged in or not.
        # Walk into today's Day node, creating it on the day's first sign. The
        # `else` fires when the visit enqueues nothing (no Day for today yet).
        today = _today();
        visit [root.shared-->[?:Day, date == today]] else {
            fresh_day = root.shared +>: OnDay() :+> Day(date=today);
            # Day is a container: others attach their signatures under it.
            grant(fresh_day[0], level=ConnectPerm);
            visit fresh_day;
        }
    }

    can sign with Day entry {
        # Now standing on today's Day node - hang the signature off `here`.
        fresh = here +>: Signed() :+> Visitor(
            name=self.name.strip()[:40] or "anonymous",
            message=self.message.strip()[:200],
            created_at=_now()
        );
        # A signature is a leaf: everyone may read it, only its author
        # (and the walkers the author runs) may change it.
        grant(fresh[0], level=ReadPerm);
        report fresh[0].to_view();
    }
}

walker:pub get_guestbook {
    has results: list[GuestEntry] = [],
        reports: list[list[GuestEntry]] = [];

    can run with Root entry {
        # One graph, one walk: every signature lives under `root.shared`,
        # the commons that is readable and attachable by default.
        visit [root.shared-->[?:Day]];
    }

    can descend with Day entry {
        visit [here-->[?:Visitor]];
    }

    can gather with Visitor entry {
        self.results.append(here.to_view());
    }

    can deliver with Root exit {
        self.results.sort(key=lambda e: GuestEntry : e.created_at, reverse=True);
        report self.results[:50];
    }
}

# ─── Microservice call: hand the live signature count to the analytics service ─
# This function runs in the guestbook (gateway) process; `analyze(...)` runs in a
# SEPARATE process the runtime spawned from the `sv import` above. The two pids
# in the result prove they are not the same process.
def:pub guestbook_stats() -> dict {
    signatures = 0;
    for day in [root.shared-->[?:Day]] {
        signatures += len([day-->[?:Visitor]]);
    }
    # cross-process call -> separate process. An sv->sv import lowers to a
    # synchronous stub (the runtime routes to analytics' process), so the call
    # resolves directly to the typed AnalyticsReport -- no `await` needed.
    rep: AnalyticsReport = analyze(signatures);
    return {
        "signatures": signatures,
        "gateway_pid": os.getpid(),
        "analytics_pid": rep.pid,
        "score": rep.score
    };
}

# ─── Streaming microservice call: re-stream analytics' commentary ────────────
# `narrate(...)` runs in the analytics process and yields one phrase at a time;
# the sv->sv stub hands back a LIVE generator - one SSE frame per yield.
# Re-yielding forwards each frame to the browser the moment it arrives: the
# gateway sits in the middle of the stream without buffering it.
def:pub guestbook_story() -> Generator {
    signatures = 0;
    for day in [root.shared-->[?:Day]] {
        signatures += len([day-->[?:Visitor]]);
    }
    def stream -> Generator[str, None, None] {
        for chunk in narrate(signatures) {
            yield str(chunk);
        }
    }
    report stream();
}

# ─── Graph traversal (real data) ─────────────────────────────────────────────
# Walk the real shared graph: visit every Visitor the guestbook created,
# recording the visit order + the root->visitor edges, so the OSP section draws
# the actual live topology. No synthetic/demo nodes.
walker:pub explore_graph {
    has rid: str = "",
        nodes: list[GraphNodeView] = [],
        edges: list[list[str]] = [],
        path: list[str] = [],
        log: list[str] = [],
        reports: list[GraphSnapshot] = [];

    can run with Root entry {
        self.rid = jid(root.shared);
        self.nodes.append(GraphNodeView(id=self.rid, label="root", kind="root"));
        self.path.append(self.rid);
        self.log.append("walker spawned at the shared root");
        visit [root.shared-->[?:Day]];
    }

    can walk_day with Day entry {
        did = jid(here);
        self.nodes.append(GraphNodeView(id=did, label=here.date, kind="day"));
        self.edges.append([self.rid, did]);
        self.path.append(did);
        self.log.append("entered " + here.date);
        visit [-->[?:Visitor]];
    }

    can walk with Visitor entry {
        vid = jid(here);
        parents = [here<--[?:Day]];
        did = jid(parents[0]) if parents else self.rid;
        self.nodes.append(GraphNodeView(id=vid, label=here.name, kind="visitor"));
        self.edges.append([did, vid]);
        self.path.append(vid);
        self.log.append("visited " + here.name);
    }

    can deliver with Root exit {
        self.log.append("traversal complete: " + str(len(self.nodes)) + " nodes");
        report GraphSnapshot(
            nodes=self.nodes, edges=self.edges, path=self.path, log=self.log
        );
    }
}
