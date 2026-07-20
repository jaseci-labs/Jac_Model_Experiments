"""Consistent shell for every showcase section: a scroll-anchor `id`, generous
vertical rhythm, an eyebrow label, a large display title, an optional subtitle,
and the section body as children."""

def:pub SectionShell(
    id: str,
    eyebrow: str,
    title: str,
    subtitle: str = "",
    children: any = None,
    className: str = "",
) -> JsxElement {
    return
        <section
            id={id}
            className={"relative flex min-h-screen w-full scroll-mt-20 flex-col justify-center px-6 py-24 " + className}
        >
            <div className="mx-auto w-full max-w-6xl">
                <div className="mx-auto mb-12 max-w-2xl text-center">
                    <div
                        className="mb-3 inline-flex items-center gap-2 rounded-full border border-primary/30 bg-primary/10 px-3 py-1 font-mono text-xs font-medium uppercase tracking-widest text-primary"
                    >
                        <span className="size-1.5 rounded-full bg-primary tij-pulse"/>
                        {eyebrow}
                    </div>
                    <h2
                        className="font-display text-3xl font-bold tracking-tight text-foreground md:text-4xl"
                    >
                        {title}
                    </h2>
                    {if subtitle {
                        <p className="mt-4 text-balance text-lg text-muted-foreground">
                            {subtitle}
                        </p>
                    }}
                </div>
                {children}
            </div>
        </section>;
}
