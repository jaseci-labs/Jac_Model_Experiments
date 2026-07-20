"""
Smoke test for driving Claude Opus + Fable through byllm's `by llm()` mechanism.

Reads model_name pins from models.json (unverified aliases pending a real
ANTHROPIC_API_KEY + live run -- see models.json note), loads .env if present
(no python-dotenv dependency), then tries one ping call per model.
"""

import from byllm.lib { Model }
import from common { load_env, load_model_name }

glob llm_opus: Model = Model(model_name=load_model_name("opus"));
glob llm_fable: Model = Model(model_name=load_model_name("fable"));

def ping_opus(word: str) -> str by llm_opus();
sem ping_opus = "Reply briefly, using the given word somewhere in your reply.";
sem ping_opus.word = "The word that must appear in the reply.";

def ping_fable(word: str) -> str by llm_fable();
sem ping_fable = "Reply briefly, using the given word somewhere in your reply.";
sem ping_fable.word = "The word that must appear in the reply.";

with entry {
    load_env();

    print("testing opus...");
    r1 = ping_opus("say pong");
    assert "pong" in r1.lower();
    print("opus ok: " + r1);

    print("testing fable...");
    r2 = ping_fable("say pong");
    assert "pong" in r2.lower();
    print("fable ok: " + r2);
}
