"""Checkpoint 1 semantic MCQ bank: 20 questions on Jac/OSP concepts, isolated
from syntax (no compiler involved, tests conceptual understanding via
multiple-choice recognition). Per design.md's eval section.

Hand-authored (not LLM+MCP-generated, per this attempt's scope decision) and
grounded against this project's own verified facts about Jac (context.md,
the jac-node-edge-patterns/jac-walker-patterns skill descriptions, and this
session's direct reading of the docs corpus during the CPT dataset build).
The answer key IS the human-verification step -- substitutes for the formal
LLM-grades-vs-human-sample trust check from the original deck's Plan 1.
"""

QUESTIONS = [
    {
        "id": "walker_def",
        "q": "What is a `walker` in Jac?",
        "options": {
            "A": "A traversal agent that moves through the graph, carrying abilities that fire at nodes/edges",
            "B": "A static configuration file loaded at startup",
            "C": "A type of database index",
            "D": "A compiler optimization pass",
        }, "correct": "A",
    },
    {
        "id": "obj_vs_class",
        "q": "Which keyword does Jac prefer over `class` for defining archetypes?",
        "options": {"A": "type", "B": "obj", "C": "struct", "D": "record"}, "correct": "B",
    },
    {
        "id": "can_with_entry",
        "q": "What does `can foo with entry` define in Jac?",
        "options": {
            "A": "An event-triggered ability, distinct from a regular `def` method",
            "B": "A private variable declaration",
            "C": "A loop construct",
            "D": "An import statement",
        }, "correct": "A",
    },
    {
        "id": "spawn_operator",
        "q": "Which operator spawns a new node connected by an edge from the current node (e.g. `root ++> MyNode()`)?",
        "options": {"A": "->", "B": "++>", "C": "==", "D": "::"}, "correct": "B",
    },
    {
        "id": "visit_stmt",
        "q": "What does the `visit` statement do inside a walker?",
        "options": {
            "A": "Deletes a node",
            "B": "Moves the walker to specified nodes/edges to continue traversal",
            "C": "Declares a new variable",
            "D": "Imports a module",
        }, "correct": "B",
    },
    {
        "id": "disengage",
        "q": "What does `disengage` do inside a walker?",
        "options": {
            "A": "Stops the walker's traversal immediately",
            "B": "Creates a new edge",
            "C": "Restarts the walker from root",
            "D": "Prints debug output",
        }, "correct": "A",
    },
    {
        "id": "has_fields",
        "q": "How are typed fields declared on a node/obj archetype in Jac?",
        "options": {"A": "Using the `has` keyword with type annotations", "B": "Using `let`", "C": "Using `field`", "D": "Using `prop`"}, "correct": "A",
    },
    {
        "id": "root_node",
        "q": "What is `root` in Jac's Object-Spatial Programming model?",
        "options": {
            "A": "A reserved anchor node — the current user's top-level entry point into their graph",
            "B": "A Python builtin function",
            "C": "The name of the compiler binary",
            "D": "A synonym for `main`",
        }, "correct": "A",
    },
    {
        "id": "edge_filter",
        "q": "What does a filter like `[-->](?:Type)` do when traversing edges?",
        "options": {
            "A": "Filters outgoing edges/nodes to only those of archetype `Type`",
            "B": "Deletes all edges of type Type",
            "C": "Sorts nodes alphabetically",
            "D": "Converts nodes to JSON",
        }, "correct": "A",
    },
    {
        "id": "osp_philosophy",
        "q": "What is the core philosophical shift Object-Spatial Programming makes relative to traditional OOP?",
        "options": {
            "A": "Computation moves to the data (walkers traverse a graph) instead of data moving to computation",
            "B": "Classes are simply renamed to objects",
            "C": "All variables become global",
            "D": "Functions can no longer take arguments",
        }, "correct": "A",
    },
    {
        "id": "by_llm",
        "q": "What does the `by llm()` construct attach to in Jac?",
        "options": {
            "A": "A function/ability whose implementation is delegated to an LLM call instead of hand-written code",
            "B": "A node's physical GPU location",
            "C": "A network socket",
            "D": "A database connection string",
        }, "correct": "A",
    },
    {
        "id": "sem_annotation",
        "q": "What does a `sem` annotation (e.g. `sem MyFunc.param = \"...\"`) typically do in Jac?",
        "options": {
            "A": "Attaches a semantic/natural-language description used to guide LLM-backed abilities",
            "B": "Declares a semaphore for threading",
            "C": "Marks a function as deprecated",
            "D": "Sets a memory alignment",
        }, "correct": "A",
    },
    {
        "id": "node_edge_relation",
        "q": "What best describes the relationship between `node` and `edge` archetypes in Jac's graph model?",
        "options": {
            "A": "Nodes are entities/vertices in the graph; edges are the connections between them",
            "B": "Nodes and edges are interchangeable synonyms",
            "C": "Edges contain nodes as children in a strict tree",
            "D": "A node can only ever have one edge",
        }, "correct": "A",
    },
    {
        "id": "ability_trigger",
        "q": "A walker's ability defined as `can foo with SomeNode entry` fires:",
        "options": {
            "A": "Only once when the walker is first created, regardless of location",
            "B": "When the walker enters (visits) a node of type SomeNode",
            "C": "When the program exits",
            "D": "Never automatically — it must be called manually",
        }, "correct": "B",
    },
    {
        "id": "jac_python_relation",
        "q": "What is the relationship between Jac and Python?",
        "options": {
            "A": "Jac compiles to Python bytecode and interops with the Python ecosystem",
            "B": "Jac is a database query language unrelated to Python",
            "C": "Jac only runs inside a browser",
            "D": "Jac replaces Python's package manager",
        }, "correct": "A",
    },
    {
        "id": "not_real_construct",
        "q": "Which of the following is NOT a real Jac construct?",
        "options": {"A": "walker", "B": "spawn", "C": "rule (as in `rule traverse { }`)", "D": "has"}, "correct": "C",
    },
    {
        "id": "spawn_semantics",
        "q": "What does `root ++> MyNode()` do?",
        "options": {
            "A": "Creates a new MyNode and connects it to root via an edge",
            "B": "Increments root's value by one",
            "C": "Compares root to MyNode for equality",
            "D": "Casts root to type MyNode",
        }, "correct": "A",
    },
    {
        "id": "walker_movement",
        "q": "Operationally, what does it mean when \"the walker moves to a node\" in OSP?",
        "options": {
            "A": "The walker's abilities are invoked in the context of that node — execution relocates to where the data lives",
            "B": "The node's memory address is copied into the walker",
            "C": "The walker deletes the node it was previously on",
            "D": "Nothing — walkers cannot move, only nodes move",
        }, "correct": "A",
    },
    {
        "id": "byllm_syntax",
        "q": "Which syntax lets a Jac function's entire logic be handled by an LLM rather than explicit code (byLLM feature)?",
        "options": {
            "A": "`def foo(...) -> T by llm();`",
            "B": "`def foo(...) -> T { auto_ai(); }`",
            "C": "`#pragma llm foo`",
            "D": "`import llm; foo.enable()`",
        }, "correct": "A",
    },
    {
        "id": "def_vs_can",
        "q": "What distinguishes a `def` method from a `can` ability on a Jac archetype?",
        "options": {
            "A": "`can` abilities are event-triggered (fire on entry/exit to nodes during traversal); `def` methods are called explicitly like normal functions",
            "B": "They are exactly the same, just different keywords for style preference",
            "C": "`def` is for walkers only, `can` is for nodes only",
            "D": "`can` methods cannot take parameters",
        }, "correct": "A",
    },
]
