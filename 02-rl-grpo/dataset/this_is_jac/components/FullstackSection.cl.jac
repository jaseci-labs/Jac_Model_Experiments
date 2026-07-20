"""Fullstack section: one language across the wire. The left panel is the real
server + client code; the right panel is a LIVE guestbook backed by the
`sign_guestbook` / `get_guestbook` walkers in `guestbook.jac`. What you type is a
node written into a persistent graph and read back by a walker."""

import from "lucide-react" { Send, MessageSquare, Database }
import from .ui.card { Card, CardContent }
import from .ui.input { Input }
import from .ui.textarea { Textarea }
import from .ui.button { Button }
import from .ui.avatar { Avatar, AvatarFallback }
import from .ui.badge { Badge }
import from .CodeBlock { CodeBlock }
import from .GraphView { GraphView }
import from .SectionShell { SectionShell }
sv import from ..guestbook { GuestEntry }

glob _CODE: str = """# Signatures are grouped by date: root.shared -> Day -> Visitor,
# all real graph data, not a fixture. `root.shared` is the public
# graph every anonymous request runs on - one walk reads it all.
node Day { has date: str; }
node Visitor { has name: str, message: str; }
edge OnDay {}  edge Signed {}

# The walker starts at the shared root and reacts at each node
# type it lands on. Traversal is `visit`, never a hand-rolled loop.
walker:pub explore_graph {
    has path: list[str] = [];

    can run with Root entry {
        visit [root.shared-->[?:Day]]; # step onto the day nodes
    }
    can walk_day with Day entry {
        visit [-->[?:Visitor]];        # then each day's signatures
    }
    can walk with Visitor entry {
        self.path.append(here.name);   # record the visit
    }
}""";

def:pub FullstackSection(
    entries: list[GuestEntry],
    name: str,
    message: str,
    onNameChange: Callable[[str], None],
    onMessageChange: Callable[[str], None],
    onSign: Callable[[], None],
    signing: bool,
    graphRefresh: int = 0,
) -> JsxElement {
    return
        <SectionShell
            id="fullstack"
            eyebrow="Fullstack"
            title="Introducing a New Paradigm, for a New Way of Problem Solving."
            subtitle="Use graph centric PL semantics to solve problems in an intuitive and elegant way. Sign below and it's persisted to a real graph. (and yes, this is a secret weapon for agentic ai harness building, but thats for another demo ;-))"
        >
            <div>
                <div className="grid items-start gap-8 lg:grid-cols-2">
                    <Card className="border-border bg-card/60">
                        <CardContent className="flex flex-col gap-4 p-6">
                            <div className="flex items-center justify-between">
                                <span
                                    className="flex items-center gap-2 text-sm font-medium text-foreground"
                                >
                                    <MessageSquare size={15} className="text-primary"/>
                                    Live guestbook
                                </span>
                                <Badge
                                    variant="secondary"
                                    className="gap-1 font-normal"
                                >
                                    <Database size={11}/>
                                    {str(len(entries))}{" "}
                                    in graph
                                </Badge>
                            </div>
                            <div className="flex flex-col gap-2">
                                <Input
                                    placeholder="Your name"
                                    value={name}
                                    maxLength={40}
                                    onChange={lambda e: ChangeEvent { onNameChange(
                                        e.target.value
                                    ); }}
                                />
                                <Textarea
                                    placeholder="Leave a message for the graph..."
                                    value={message}
                                    rows={2}
                                    maxLength={200}
                                    className="resize-none"
                                    onChange={lambda e: ChangeEvent { onMessageChange(
                                        e.target.value
                                    ); }}
                                />
                                <Button
                                    className="self-end rounded-full px-5 font-semibold"
                                    disabled={signing or not message.strip()}
                                    onClick={lambda { onSign(); }}
                                >
                                    <Send size={15}/>
                                    {"Signing..." if signing else "Sign the graph"}
                                </Button>
                            </div>
                            <div
                                className="flex max-h-72 flex-col gap-3 overflow-y-auto pr-1"
                            >
                                {if len(entries) == 0 {
                                    <div
                                        className="py-8 text-center text-sm text-muted-foreground"
                                    >
                                        No signatures yet - be the first node.
                                    </div>
                                }}
                                {[
                                    <div className="tij-fade-up flex gap-3" key={e.id}>
                                        <Avatar className="size-9 shrink-0">
                                            <AvatarFallback
                                                className="bg-primary/15 text-sm text-primary"
                                            >
                                                {(e.name[0].upper() if e.name else "?")}
                                            </AvatarFallback>
                                        </Avatar>
                                        <div
                                            className="min-w-0 flex-1 rounded-lg border border-border bg-background/50 px-3 py-2"
                                        >
                                            <div
                                                className="text-sm font-semibold text-foreground"
                                            >
                                                {e.name}
                                            </div>
                                            <div
                                                className="break-words text-sm text-muted-foreground"
                                            >
                                                {e.message}
                                            </div>
                                        </div>
                                    </div> for e in entries
                                ]}
                            </div>
                        </CardContent>
                    </Card>
                    <CodeBlock code={_CODE} filename="guestbook.jac"/>
                </div>
                <div className="mt-8">
                    <GraphView refreshKey={graphRefresh}/>
                </div>
            </div>
        </SectionShell>;
}
