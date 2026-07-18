# CPT-v1 Head-to-Head: Full Prompts + Verbatim Outputs

*Raw data behind `analysis.md`'s head-to-head section. Every prompt and every model response below is copied word-for-word from `03-new/results/cpt-v1/headtohead.json` — nothing trimmed, nothing paraphrased. Same sampler both models (temp 0.2, top_p 0.9), single sample each, no cherry-picking.*

## syntax_fn

**Prompt:**
`````
Write a Jac function `add(a: int, b: int) -> int` that returns their sum.
`````

**FRESH (base `qwen-q4`) output:**
````
Here's a simple Jac function that adds two integers:

```jac
func add(a: int, b: int) -> int:
    return a + b
```

This function:
- Takes two integer parameters `a` and `b`
- Returns their sum as an integer
- Uses the `func` keyword to define the function
- Specifies the parameter types with `: int` and return type with `-> int`

You can use it like:
```jac
result = add(5, 3)  # Returns 8
```
````

`jac check` result: **FAIL**


**CPT-v1 (`qwen-cpt-v1-fused-q4`) output:**
````
Here's a simple Jac function that adds two integers:

```jac
func add(a: int, b: int) -> int:
    return a + b
```

This function:
- Takes two integer parameters `a` and `b`
- Returns their sum as an integer
- Uses the standard addition operator `+`

Example usage:
```jac
# add(3, 5) returns 8
# add(-2, 7) returns 5
# add(0, 0) returns 0
```
````

`jac check` result: **FAIL**


---

## osp_walker

**Prompt:**
`````
Write a Jac walker named `CountNodes` that starts at root, visits every node connected via outgoing edges, and prints the total count of nodes visited.
`````

**FRESH (base `qwen-q4`) output:**
````
Here's a Jac walker named `CountNodes` that traverses the graph and counts all visited nodes:

```python
from jaclang import jac

# Define the CountNodes walker
walker CountNodes {
    # Start at the root node
    entry = root
    
    # Initialize counter
    count = 0
    
    # Main traversal logic
    rule traverse {
        # Increment counter for each node visited
        count += 1
        
        # Print current count
        print(f"Visited node {count}")
        
        # Continue traversal to all outgoing edges
        foreach edge in self.outgoing_edges {
            # Move to the connected node
            next_node = edge.target
            # Continue traversal from this node
            continue next_node
        }
    }
    
    # Final result
    rule finalize {
        print(f"Total nodes visited: {count}")
    }
    
    # Execute the walker
    rule execute {
        # Start traversal from root
        self.traverse(root)
        self.finalize()
    }
}
```

However, since the exact syntax might vary depending on the specific Jac implementation, here's a more standard approach:

```python
walker CountNodes {
    # Initialize counter
    count = 0
    
    # Main traversal rule
    rule on_entry {
        # Increment counter for current node
        self.count += 1
        
        # Print current count
        print(f"Visiting node {self.count}")
        
        # Continue to all connected nodes
        foreach node in self.outgoing_nodes {
            # Recursively visit connected nodes
            node.visit()
        }
    }
    
    # Final output
    rule on_exit {
        print(f"Total nodes visited: {self.count}")
    }
}
```

Or a simpler version that directly counts:

```python
walker CountNodes {
    count = 0
    
    rule start {
        # Start from root and count all reachable nodes
        self.count_nodes(root)

````

`jac check` result: **FAIL**


**CPT-v1 (`qwen-cpt-v1-fused-q4`) output:**
````
Here's a Jac walker named `CountNodes` that traverses the graph and counts all visited nodes:

```python
walker CountNodes {
    has root;

    rule start {
        # Start at the root node
        spawn here -> root;
    }

    rule traverse {
        # Visit all outgoing edges from current node
        for edge in here.out_edges {
            spawn here -> edge.target;
        }
        
        # Print the total count when we're done
        if not here.out_edges:
            # This is a leaf node, print the count
            # In a real implementation, you'd need to track the global count
            # This is a simplified version showing the concept
            print(f"Total nodes visited: {count_nodes(here)}");
    }

    # Helper function to count nodes (would need to be implemented in actual Jac)
    # This represents the logic for counting nodes in the traversal
    def count_nodes(node):
        count = 1  # Count current node
        for edge in node.out_edges:
            count += count_nodes(edge.target)  # Recursively count children
        return count
}
```

However, since Jac's walker syntax is more limited, here's a more practical implementation:

```python
walker CountNodes {
    has root;
    has count = 0;

    rule start {
        # Initialize counter and start traversal
        self.count = 0;
        spawn here -> root;
    }

    rule visit_node {
        # Increment counter for each node visited
        self.count += 1;
        
        # Visit all outgoing edges
        for edge in here.out_edges {
            spawn here -> edge.target;
        }
        
        # When we've processed all outgoing edges, print result
        if len(here.out_edges) == 0 or self.count > 1000:  # Prevent infinite loop
            print(f"Total nodes visited: {self.count}");
    }
}

````

`jac check` result: **FAIL**


---

## py2jac

**Prompt:**
`````
Convert this Python function to idiomatic Jac:
```python
def is_even(n):
    return n % 2 == 0
```
`````

**FRESH (base `qwen-q4`) output:**
````
Here's the idiomatic Jac conversion:

```jac
fun is_even(n):
    return n % 2 == 0
```

The main differences are:
- Use `fun` instead of `def` for function definitions in Jac
- No colon (`:`) needed after the function signature
- The function body syntax remains the same

Alternatively, you could write it more concisely as a one-liner:

```jac
fun is_even(n): return n % 2 == 0
```

Or using Jac's expression syntax:

```jac
fun is_even(n):
    return n % 2 == 0
```

All three forms are valid, but the first is the most readable and idiomatic.
````

`jac check` result: **FAIL**


**CPT-v1 (`qwen-cpt-v1-fused-q4`) output:**
````
Here's the idiomatic Jac conversion:

```jac
def is_even(n):
    return n % 2 == 0
```

The conversion is actually identical! Jac's syntax for functions and basic operations is very similar to Python, so this function translates directly. In Jac, you can also write it more concisely as:

```jac
def is_even(n):
    return n % 2 == 0
```

Or using the implicit return syntax (if supported by your Jac version):

```jac
def is_even(n) -> bool:
    return n % 2 == 0
```

The function works the same way in Jac - it takes a number `n` and returns `true` if it's even, `false` otherwise.
````

`jac check` result: **FAIL**


---

## concept_node_edge

**Prompt:**
`````
In Jac's Object-Spatial Programming model, what is the difference between a `node` and an `edge`? Answer in 2-3 sentences.
`````

**FRESH (base `qwen-q4`) output:**
````
In Jac's Object-Spatial Programming model, a `node` represents a discrete object or entity that exists at a specific spatial location, while an `edge` represents the relationships or connections between nodes in the spatial structure. Nodes are the fundamental building blocks that contain data and properties, whereas edges define how these objects relate to each other in space. The key distinction is that nodes are spatially located entities, while edges are the spatial relationships that connect and organize these entities within the computational framework.
````

**CPT-v1 (`qwen-cpt-v1-fused-q4`) output:**
````
In Jac's Object-Spatial Programming model, a `node` represents a discrete object or entity that exists at specific spatial coordinates, while an `edge` represents a connection or relationship between nodes. Nodes are the fundamental units that store data and can move through space, whereas edges define the topological relationships and communication pathways between these spatially-distributed objects. The key distinction is that nodes are the "things" in the system, while edges are the "connections" that define how these things relate to each other in both spatial and conceptual terms.
````

---

## concept_by_llm

**Prompt:**
`````
What does the `by llm()` construct do in Jac? Answer in 2-3 sentences.
`````

**FRESH (base `qwen-q4`) output:**
````
The `by llm()` construct in Jac is used to specify that a behavior or action should be executed by a Language Model (LLM) rather than being handled by the Jac runtime or other built-in mechanisms. It allows developers to delegate specific tasks, such as natural language processing, text generation, or complex reasoning, to an LLM for execution. This construct is particularly useful for incorporating AI-powered capabilities into Jac programs, enabling them to leverage LLMs for tasks like generating responses, analyzing text, or performing intelligent decision-making.
````

**CPT-v1 (`qwen-cpt-v1-fused-q4`) output:**
````
The `by llm()` construct in Jac is used to integrate and interact with large language models directly within Jac code, allowing developers to leverage LLM capabilities for tasks like text generation, natural language processing, and AI-powered functionality. It provides a seamless way to call LLM APIs and incorporate their responses into Jac programs, making it easier to build AI-enhanced applications. This construct simplifies the process of incorporating machine learning models into Jac applications without requiring extensive API handling code.
````

---
