"""This is Jac - frontend shell.

Owns the shared client state and the server-walker handlers, then composes the
scroll-driven showcase from the section components in `components/`. The
guestbook section is wired to live `guestbook.jac` walkers;
the rest pair real Jac source with captured or in-browser demos. Handler bodies
live in `frontend.impl.jac`.
"""

sv import from .guestbook { sign_guestbook, get_guestbook, guestbook_stats }
sv import from .guestbook { GuestEntry }
sv import from .source_viewer { list_source_files, get_source_file }

import from .components.Nav { Nav }
import from .components.HeroSection { HeroSection }
import from .components.InteropSection { InteropSection }
import from .components.FullstackSection { FullstackSection }
import from .components.MicroservicesSection { MicroservicesSection }
import from .components.ByllmSection { ByllmSection }
import from .components.NativeSection { NativeSection }
import from .components.WasmShooterSection { WasmShooterSection }
import from .components.LittleXSection { LittleXSection }
import from .components.SourceViewer { SourceViewer }
import from .components.SeamsSection { SeamsSection }
import from .components.OutroSection { OutroSection }
import from "@jac/runtime" { Router, Routes, Route }
import from .littlex.frontend { app as LittleX }
import "./global.css";

# Top-level router: the scroll showcase at `/`, and the full littleX app
# (the same `<LittleX/>` the section embeds) standalone at `/littlex`.
def:pub app -> JsxElement {
    return
        <Router>
            <Routes>
                <Route
                    path="/littlex"
                    element={<div
                        className="h-screen overflow-hidden bg-background text-foreground"
                    >
                        <LittleX/>
                    </div>}
                />
                <Route path="/*" element={<Showcase/>}/>
            </Routes>
        </Router>;
}

def:pub Showcase -> JsxElement {
    has darkMode: bool = True,
        guestbook: list[GuestEntry] = [],
        gbName: str = "",
        gbMessage: str = "",
        gbSigning: bool = False,
        graphNonce: int = 0,
        microStats: dict | None = None,
        microRunning: bool = False,
        story: str = "",
        storyOn: bool = False,
        srcFiles: list[dict] = [],
        srcFile: dict | None = None,
        srcLoading: bool = False;

    can with entry {
        loadInitial();
    }

    can with entry {
        loadSource();
    }

    can with [darkMode] entry {
        if darkMode {
            document.documentElement.classList.add("dark");
        } else {
            document.documentElement.classList.remove("dark");
        }
    }

    async def loadInitial -> None;
    async def signGuestbook -> None;
    async def runStats -> None;
    async def loadSource -> None;
    async def openSource(path: str) -> None;

    return
        <div className="min-h-screen bg-background text-foreground">
            <Nav {darkMode} onToggleTheme={lambda { darkMode = not darkMode; }}/>
            <main>
                <HeroSection/>
                <InteropSection/>
                <LittleXSection/>
                <FullstackSection
                    entries={guestbook}
                    name={gbName}
                    message={gbMessage}
                    onNameChange={lambda v: str { gbName = v; }}
                    onMessageChange={lambda v: str { gbMessage = v; }}
                    onSign={signGuestbook}
                    signing={gbSigning}
                    graphRefresh={graphNonce}
                />
                <MicroservicesSection
                    stats={microStats}
                    running={microRunning}
                    {story}
                    {storyOn}
                    onRun={runStats}
                />
                <NativeSection/>
                <WasmShooterSection/>
                <ByllmSection/>
                <SourceViewer
                    files={srcFiles}
                    current={srcFile}
                    loading={srcLoading}
                    onOpen={openSource}
                />
                <SeamsSection/>
                <OutroSection/>
            </main>
        </div>;
}
