"""RL driver (pure-lib): source_lexer.analyse_text (heuristic source-type sniff).

Origin: this_is_jac/source_lexer.jac (the JacLexer.analyse_text logic, isolated from the
Pygments RegexLexer class machinery). Deterministic: the body is a pure boolean heuristic over
its string argument (a fixed-pattern shebang match plus a literal substring scan of the first
1000 chars), so for fixed inputs the result never changes. The marked body is the unit the
policy must regenerate; the with entry block exercises it on fixed strings and prints a
deterministic boolean projection.
"""

import from pygments.util { shebang_matches }

def analyse_text(text: str) -> bool {
    __HOLE__
}

with entry {
    print(analyse_text("import os;"));
    print(analyse_text("#!/usr/bin/env python3\nx = 1;"));
    print(analyse_text("walker foo {}"));
}
