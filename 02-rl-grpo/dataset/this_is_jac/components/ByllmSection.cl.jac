"""AI-native section: a function body delegated to an LLM with `by llm()`. The
types ARE the prompt - no string wrangling. The right-hand card illustrates a
sample call and the structured value it returns."""

import from "lucide-react" { Sparkles, ArrowRight }
import from .ui.card { Card, CardContent }
import from .ui.badge { Badge }
import from .CodeBlock { CodeBlock }
import from .SectionShell { SectionShell }

glob _CODE: str = """import from byllm { Model }

glob llm = Model(model_name="gpt-4o");

enum Priority { LOW, MEDIUM, HIGH, URGENT }

obj Ticket {
    has summary: str,
        priority: Priority,
        tags: list[str];
}

# No prompt string. The signature is the contract; byLLM
# fills the body and returns a typed, validated Ticket.
def triage(message: str) -> Ticket by llm();""";

def:pub ByllmSection -> JsxElement {
    return
        <SectionShell
            id="ai"
            eyebrow="AI-native"
            title="Functions powered by an LLM"
            subtitle="Write the signature, write `by llm()`, and the model becomes the implementation - returning typed, validated objects, not loose text. Nothing to run here because tokens cost $$$ (for now ;-) :-) :-))."
        >
            <div className="grid items-center gap-8 lg:grid-cols-2">
                <CodeBlock code={_CODE} filename="triage.jac"/>
                <Card className="border-border bg-card/60">
                    <CardContent className="flex flex-col gap-4 p-6">
                        <div
                            className="flex items-center gap-2 text-sm text-muted-foreground"
                        >
                            <Sparkles size={15} className="text-primary"/>
                            triage(message) call
                        </div>
                        <div
                            className="rounded-lg border border-border bg-background/60 p-4 font-mono text-sm text-foreground"
                        >
                            "the checkout page 500s every time I hit pay, losing sales right now!!"
                        </div>
                        <div className="flex justify-center text-muted-foreground">
                            <ArrowRight size={18} className="rotate-90"/>
                        </div>
                        <div
                            className="flex flex-col gap-3 rounded-lg border border-primary/30 bg-primary/5 p-4"
                        >
                            <div className="flex items-center justify-between">
                                <span
                                    className="font-mono text-xs text-muted-foreground"
                                >
                                    Ticket
                                </span>
                                <Badge className="bg-destructive/15 text-destructive">
                                    URGENT
                                </Badge>
                            </div>
                            <div className="font-mono text-sm">
                                <span className="text-muted-foreground">summary: </span>
                                <span className="text-foreground">
                                    "Checkout returns 500 on payment"
                                </span>
                            </div>
                            <div
                                className="flex flex-wrap items-center gap-2 font-mono text-sm"
                            >
                                <span className="text-muted-foreground">tags:</span>
                                {[
                                    <Badge
                                        variant="secondary"
                                        key={t}
                                        className="font-normal"
                                    >
                                        {t}
                                    </Badge>
                                    for t in ["checkout", "payments", "regression"]
                                ]}
                            </div>
                        </div>
                    </CardContent>
                </Card>
            </div>
        </SectionShell>;
}
