"""A horizontal labelled progress bar used to compare two measurements
(e.g. Jac vs Zig average FPS). The fill width is `value / max_value`."""

def:pub StatBar(
    label: str,
    value: float,
    max_value: float,
    display: str,
    barClass: str = "bg-primary",
    sub: str = "",
) -> JsxElement {
    pct = 0.0;
    if max_value > 0.0 {
        pct = min(100.0, value / max_value * 100.0);
    }
    return
        <div className="flex flex-col gap-1.5">
            <div className="flex items-baseline justify-between">
                <span className="text-sm font-medium text-foreground">{label}</span>
                <span className="font-mono text-sm tabular-nums text-muted-foreground">
                    {display}
                </span>
            </div>
            <div className="h-2.5 w-full overflow-hidden rounded-full bg-muted">
                <div
                    className={"h-full rounded-full transition-all duration-700 ease-out " + barClass}
                    style={{"width": str(pct) + "%"}}
                />
            </div>
            {if sub {
                <span className="text-xs text-muted-foreground">{sub}</span>
            }}
        </div>;
}
