"""This is Jac - frontend handler implementations.

Each handler spawns a `guestbook.jac` walker on the graph (`root spawn ...`) and
folds its `reports` back into client state. Identical call shape to the server
- no fetch, no client SDK, no serializers."""

impl Showcase.loadInitial -> None {
    gb = root spawn get_guestbook();
    guestbook = gb.reports[0] if gb.reports else [];
}

impl Showcase.signGuestbook -> None {
    if not gbMessage.strip() {
        return;
    }
    gbSigning = True;
    root spawn sign_guestbook(name=gbName, message=gbMessage);
    gbMessage = "";
    gb = root spawn get_guestbook();
    guestbook = gb.reports[0] if gb.reports else [];
    gbSigning = False;
    # bump the nonce so GraphView reloads its /graph iframe with the new node
    graphNonce += 1;
}

impl Showcase.runStats -> None {
    microRunning = True;
    story = "";
    microStats = await guestbook_stats();
    microRunning = False;
    # Second call to the SAME service - this one STREAMS. `guestbook_story`
    # re-yields analytics' SSE frames; read them straight off the response
    # body and grow the paragraph one frame at a time. (No `\n` literals in
    # cl code - jac2js mangles them - hence chr(10).)
    storyOn = True;
    nl = chr(10);
    sep = nl + nl;
    # Accumulate in a local: `story` is reactive state, and reading it back
    # inside this async loop sees the render-time snapshot, not the latest
    # value - append to `acc` and assign the whole paragraph each frame.
    acc = "";
    try {
        resp = await fetch("/function/guestbook_story", {
            "method": "POST",
            "headers": {"Content-Type": "application/json"},
            "body": "{}"
        });
        reader = resp.body.getReader();
        decoder = new(TextDecoder);
        buf = "";
        feeding = True;
        while feeding {
            part = await reader.read();
            if part.done {
                feeding = False;
            } else {
                buf = buf + decoder.decode(part.value, {"stream": True});
                frames = buf.split(sep);
                buf = frames.pop();
                for frame in frames {
                    for line in frame.split(nl) {
                        if line.startswith("data: ") and not frame.startswith("event:") {
                            acc = acc + JSON.parse(line[6:]);
                            story = acc;
                        }
                    }
                }
            }
        }
    } except Exception as e {
        story = acc + " [stream interrupted]";
    }
    storyOn = False;
}

impl Showcase.loadSource -> None {
    srcFiles = await list_source_files();
    srcLoading = True;
    srcFile = await get_source_file("frontend.cl.jac");
    srcLoading = False;
}

impl Showcase.openSource(path: str) -> None {
    srcLoading = True;
    srcFile = await get_source_file(path);
    srcLoading = False;
}
