"""'A whole app, one component' - the entire littleX social app embedded as a
single `<LittleX/>`. This is the real littleX from `littlex/` (auth, feed,
follows, channels, profiles); its `sv import` walkers register in this very
server, so it reads and writes the SAME persistent graph as the rest of the
showcase. littleX scales to its container (its `min-h-screen` was changed to
`h-full`), so it fills this responsive 'app window' frame; the full standalone
app is one click away at `/littlex`."""

import from "lucide-react" { AppWindow, ArrowUpRight }
import from "@jac/runtime" { Link }
import from .SectionShell { SectionShell }
import from .CodeBlock { CodeBlock }
import from .DocsView { DocsView }
import from ..littlex.frontend { app as LittleX }

def:pub LittleXSection -> JsxElement {
    app_code = (
        """# app.jac - the data, the endpoint, and the UI in one file.

# this is a class
node Tweet {
    has content: str,
        author: str,
        likes: int = 0;
}

# Back end: compiles to a public endpoint - this IS the API.
def:pub create_tweet(content: str) -> Tweet {
    # this might look strange, but it abstracts away auth and db
    root ++> (tweet := Tweet(content=content, author="ada"));
    return tweet;
}

# Front end: a cl def calls the endpoint like a function.
cl def Composer -> JsxElement {
    has text: str = "";

    async def post -> None {
        new_tweet = await create_tweet(text);  # typed end to end
        print(new_tweet.author, new_tweet.likes);
    }

    return <button onClick={post}>Tweet</button>;
}"""
    );

    return
        <SectionShell
            id="littlex"
            eyebrow="Real apps"
            title="Below is Real Twitter-like App. Use It."
            subtitle="In a single component. All frontend/backend interfacing, data handling, and db organization is abstracted away (done for you and optimized by the compiler)."
        >
            <div className="mx-auto max-w-5xl">
                <div
                    className="overflow-hidden rounded-2xl border border-border bg-card shadow-2xl shadow-black/40"
                >
                    <div
                        className="flex items-center gap-2 border-b border-border/70 bg-white/[0.02] px-4 py-2.5"
                    >
                        <span className="size-3 rounded-full bg-[#ff5f57]"/>
                        <span className="size-3 rounded-full bg-[#febc2e]"/>
                        <span className="size-3 rounded-full bg-[#28c840]"/>
                        <span
                            className="ml-2 flex items-center gap-1.5 font-mono text-xs text-muted-foreground"
                        >
                            <AppWindow size={13}/>
                            littlex - embedded live app, same graph
                        </span>
                        <Link
                            to="/littlex"
                            className="ml-auto flex items-center gap-1 rounded-md px-2 py-1 text-xs text-muted-foreground transition hover:bg-white/5 hover:text-foreground"
                        >
                            open full app
                            <ArrowUpRight size={13}/>
                        </Link>
                    </div>
                    <div
                        className="h-[78vh] min-h-[600px] max-h-[860px] overflow-hidden bg-background"
                    >
                        <LittleX/>
                    </div>
                </div>
                <p className="mt-4 text-center text-sm text-muted-foreground">
                    Sign up inside the frame to try it - your account and posts persist in the shared graph. Or{" "}
                    <Link
                        to="/littlex"
                        className="font-medium text-primary hover:underline"
                    >
                        open the full app at /littlex
                    </Link>
                    .
                </p>
                <div className="mb-10">
                    <p className="mb-4 text-center text-sm text-muted-foreground">
                        The data, the endpoint, and the UI in one file: the front end calls the back end like a function and gets a typed{" "}
                        <span className="font-mono text-foreground">Tweet</span>{" "}
                        back. No fetch, no REST route, no JSON.
                    </p>
                    <div className="grid items-stretch gap-4 lg:grid-cols-2">
                        <CodeBlock filename="app.jac" code={app_code} dense={True}/>
                        <DocsView/>
                    </div>
                    <p className="mt-3 text-center text-xs text-muted-foreground">
                        That single file is already a live API: every endpoint shows up in the auto-generated Swagger UI on the right.
                    </p>
                </div>
            </div>
        </SectionShell>;
}
