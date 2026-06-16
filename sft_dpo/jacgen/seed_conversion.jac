"""Seed batch of Python -> idiomatic Jac conversion examples.

Every Jac body here was generated agentically and validated live against the
Jac runtime via jac-mcp (compiler exit 0 + stdout == expected). Running this
module writes the SFT/DPO JSONL files and then re-validates every stored line
through `jac run` so the persisted dataset is provably correct.

Run from the repo root:
    jac run srccurrent/jacgen/seed_conversion.jac
"""

import from writer { make_sft_example, append_jsonl, revalidate_example }
import from dedup { dedup_examples }
import from pathlib { Path }

glob SFT_PATH: str = "dataset/conversion/sft.jsonl";
glob DPO_PATH: str = "dataset/conversion/dpo.jsonl";

glob SEEDS: list[dict] = [
    # ---------------- atomic: direct function mapping ----------------
    {
        "difficulty": "atomic", "expected": "120",
        "py": '''
def factorial(n: int) -> int:
    """Return n!."""
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result
''',
        "jac": '''
def factorial(n: int) -> int {
    result = 1;
    for i in range(2, n + 1) {
        result *= i;
    }
    return result;
}

with entry {
    print(factorial(5));
}
'''
    },
    {
        "difficulty": "atomic", "expected": "55",
        "py": '''
def fib(n: int) -> int:
    """Return the nth Fibonacci number."""
    if n < 2:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
''',
        "jac": '''
def fib(n: int) -> int {
    if n < 2 { return n; }
    a = 0; b = 1;
    for _ in range(2, n + 1) { (a, b) = (b, a + b); }
    return b;
}

with entry {
    print(fib(10));
}
'''
    },
    {
        "difficulty": "atomic", "expected": "True\nFalse",
        "py": '''
def is_palindrome(s: str) -> bool:
    """Return True if s reads the same forwards and backwards."""
    return s == s[::-1]
''',
        "jac": '''
def is_palindrome(s: str) -> bool {
    return s == s[::-1];
}

with entry {
    print(is_palindrome("racecar"));
    print(is_palindrome("hello"));
}
'''
    },
    {
        "difficulty": "atomic", "expected": "6",
        "py": '''
def gcd(a: int, b: int) -> int:
    """Return the greatest common divisor of a and b."""
    while b != 0:
        a, b = b, a % b
    return a
''',
        "jac": '''
def gcd(a: int, b: int) -> int {
    while b != 0 { (a, b) = (b, a % b); }
    return a;
}

with entry {
    print(gcd(48, 18));
}
'''
    },
    {
        "difficulty": "atomic", "expected": "15",
        "py": '''
def sum_list(nums: list[int]) -> int:
    """Return the sum of a list of integers."""
    total = 0
    for n in nums:
        total += n
    return total
''',
        "jac": '''
def sum_list(nums: list[int]) -> int {
    total = 0;
    for n in nums { total += n; }
    return total;
}

with entry {
    print(sum_list([1, 2, 3, 4, 5]));
}
'''
    },
    # ---------------- idiomatic: Python class -> Jac obj ----------------
    {
        "difficulty": "idiomatic", "expected": "120.0",
        "py": '''
class BankAccount:
    """A simple bank account."""
    def __init__(self, balance: float = 0.0):
        self.balance = balance

    def deposit(self, amount: float) -> float:
        self.balance += amount
        return self.balance

    def withdraw(self, amount: float) -> float:
        if amount > self.balance:
            raise ValueError("Insufficient funds")
        self.balance -= amount
        return self.balance
''',
        "jac": '''
obj BankAccount {
    has balance: float = 0.0;

    def deposit(amount: float) -> float {
        self.balance += amount;
        return self.balance;
    }

    def withdraw(amount: float) -> float {
        if amount > self.balance {
            raise ValueError("Insufficient funds");
        }
        self.balance -= amount;
        return self.balance;
    }
}

with entry {
    acct = BankAccount(balance=100.0);
    acct.deposit(50.0);
    print(acct.withdraw(30.0));
}
'''
    },
    {
        "difficulty": "idiomatic", "expected": "3\n2",
        "py": '''
class Stack:
    """A LIFO stack of integers."""
    def __init__(self):
        self.items = []

    def push(self, x: int) -> None:
        self.items.append(x)

    def pop(self) -> int:
        return self.items.pop()

    def size(self) -> int:
        return len(self.items)
''',
        "jac": '''
obj Stack {
    has items: list[int] = [];

    def push(x: int) -> None { self.items.append(x); }

    def pop() -> int { return self.items.pop(); }

    def size() -> int { return len(self.items); }
}

with entry {
    s = Stack();
    s.push(1); s.push(2); s.push(3);
    print(s.pop());
    print(s.size());
}
'''
    },
    {
        "difficulty": "idiomatic", "expected": "3\n1",
        "py": '''
class WordCounter:
    """Counts occurrences of words."""
    def __init__(self):
        self.counts = {}

    def add(self, word: str) -> None:
        self.counts[word] = self.counts.get(word, 0) + 1

    def get_count(self, word: str) -> int:
        return self.counts.get(word, 0)
''',
        "jac": '''
obj WordCounter {
    has counts: dict[str, int] = {};

    def add(word: str) -> None {
        self.counts[word] = self.counts.get(word, 0) + 1;
    }

    def get_count(word: str) -> int {
        return self.counts.get(word, 0);
    }
}

with entry {
    c = WordCounter();
    for w in ["a", "b", "a", "a"] { c.add(w); }
    print(c.get_count("a"));
    print(c.get_count("b"));
}
'''
    },
    {
        "difficulty": "idiomatic", "expected": "12.0\n14.0",
        "py": '''
class Rectangle:
    """An axis-aligned rectangle."""
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height

    def area(self) -> float:
        return self.width * self.height

    def perimeter(self) -> float:
        return 2 * (self.width + self.height)
''',
        "jac": '''
obj Rectangle {
    has width: float;
    has height: float;

    def area() -> float { return self.width * self.height; }

    def perimeter() -> float { return 2 * (self.width + self.height); }
}

with entry {
    r = Rectangle(width=3.0, height=4.0);
    print(r.area());
    print(r.perimeter());
}
'''
    },
    # ---------------- composed: paradigm shift to graph/walker ----------------
    {
        "difficulty": "composed", "expected": "600",
        "py": '''
def total_population(graph: dict, start: str, pops: dict) -> int:
    """Sum populations of all cities reachable from start (DFS)."""
    seen = set()
    stack = [start]
    total = 0
    while stack:
        city = stack.pop()
        if city in seen:
            continue
        seen.add(city)
        total += pops[city]
        stack.extend(graph.get(city, []))
    return total
''',
        "jac": '''
node City {
    has name: str;
    has population: int = 0;
}

walker SumPop {
    has total: int = 0;

    can tally with City entry {
        self.total += here.population;
        visit [-->];
    }
}

with entry {
    a = City(name="A", population=100);
    b = City(name="B", population=200);
    c = City(name="C", population=300);
    root ++> a;
    a ++> b;
    b ++> c;
    w = a spawn SumPop();
    print(w.total);
}
'''
    },
    {
        "difficulty": "composed", "expected": "['dave']",
        "py": '''
def friends_of_friends(graph: dict, person: str) -> list:
    """Return sorted friends-of-friends, excluding self and direct friends."""
    direct = set(graph.get(person, []))
    result = set()
    for friend in direct:
        for f in graph.get(friend, []):
            if f != person and f not in direct:
                result.add(f)
    return sorted(result)
''',
        "jac": '''
node Person {
    has name: str;
}

edge Friend {}

walker FoF {
    has origin: str = "";
    has direct: list[str] = [];
    has result: list[str] = [];

    can collect with Person entry {
        if self.origin == "" {
            self.origin = here.name;
            self.direct = [p.name for p in [-->]];
            visit [-->];
        } else {
            for p in [-->] {
                if p.name != self.origin
                   and p.name not in self.direct
                   and p.name not in self.result {
                    self.result.append(p.name);
                }
            }
        }
    }
}

with entry {
    alice = Person(name="alice");
    bob = Person(name="bob");
    carol = Person(name="carol");
    dave = Person(name="dave");
    root ++> alice;
    alice +>:Friend:+> bob;
    alice +>:Friend:+> carol;
    bob +>:Friend:+> dave;
    carol +>:Friend:+> dave;
    w = alice spawn FoF();
    print(sorted(w.result));
}
'''
    },
    {
        "difficulty": "composed", "expected": "4",
        "py": '''
def count_nodes(tree: dict, start: str) -> int:
    """Count all nodes reachable from start in a tree."""
    count = 0
    stack = [start]
    while stack:
        node = stack.pop()
        count += 1
        stack.extend(tree.get(node, []))
    return count
''',
        "jac": '''
node TreeNode {
    has val: int;
}

walker NodeCounter {
    has count: int = 0;

    can tally with TreeNode entry {
        self.count += 1;
        visit [-->];
    }
}

with entry {
    r1 = TreeNode(val=1);
    r2 = TreeNode(val=2);
    r3 = TreeNode(val=3);
    r4 = TreeNode(val=4);
    root ++> r1;
    r1 ++> r2;
    r1 ++> r3;
    r2 ++> r4;
    w = r1 spawn NodeCounter();
    print(w.count);
}
'''
    },
    {
        "difficulty": "composed", "expected": "60",
        "py": '''
class ListNode:
    def __init__(self, val: int, next=None):
        self.val = val
        self.next = next


def sum_linked_list(head: "ListNode") -> int:
    """Sum the values of a singly linked list."""
    total = 0
    node = head
    while node is not None:
        total += node.val
        node = node.next
    return total
''',
        "jac": '''
node LLNode {
    has val: int;
}

edge Next {}

walker SumList {
    has total: int = 0;

    can add with LLNode entry {
        self.total += here.val;
        visit [-->];
    }
}

with entry {
    n1 = LLNode(val=10);
    n2 = LLNode(val=20);
    n3 = LLNode(val=30);
    root ++> n1;
    n1 +>:Next:+> n2;
    n2 +>:Next:+> n3;
    w = n1 spawn SumList();
    print(w.total);
}
'''
    },
    # ---------------- batch 2: atomic ----------------
    {
        "difficulty": "atomic", "expected": "olleh",
        "py": '''
def reverse_string(s: str) -> str:
    """Return s reversed."""
    return s[::-1]
''',
        "jac": '''
def reverse_string(s: str) -> str {
    return s[::-1];
}

with entry {
    print(reverse_string("hello"));
}
'''
    },
    {
        "difficulty": "atomic", "expected": "5",
        "py": '''
def count_vowels(s: str) -> int:
    """Count vowels in s (case-insensitive)."""
    count = 0
    for ch in s.lower():
        if ch in "aeiou":
            count += 1
    return count
''',
        "jac": '''
def count_vowels(s: str) -> int {
    count = 0;
    for ch in s.lower() {
        if ch in "aeiou" { count += 1; }
    }
    return count;
}

with entry {
    print(count_vowels("Education"));
}
'''
    },
    {
        "difficulty": "atomic", "expected": "9",
        "py": '''
def max_list(nums: list[int]) -> int:
    """Return the largest value in nums."""
    best = nums[0]
    for n in nums:
        if n > best:
            best = n
    return best
''',
        "jac": '''
def max_list(nums: list[int]) -> int {
    best = nums[0];
    for n in nums {
        if n > best { best = n; }
    }
    return best;
}

with entry {
    print(max_list([3, 7, 2, 9, 4]));
}
'''
    },
    {
        "difficulty": "atomic", "expected": "1024",
        "py": '''
def power(base: int, exp: int) -> int:
    """Return base raised to exp."""
    result = 1
    for _ in range(exp):
        result *= base
    return result
''',
        "jac": '''
def power(base: int, exp: int) -> int {
    result = 1;
    for _ in range(exp) { result *= base; }
    return result;
}

with entry {
    print(power(2, 10));
}
'''
    },
    # ---------------- batch 2: idiomatic (Python class -> obj) ----------------
    {
        "difficulty": "idiomatic", "expected": "1\n2",
        "py": '''
class Queue:
    """A FIFO queue of integers."""
    def __init__(self):
        self.items = []

    def enqueue(self, x: int) -> None:
        self.items.append(x)

    def dequeue(self) -> int:
        return self.items.pop(0)

    def size(self) -> int:
        return len(self.items)
''',
        "jac": '''
obj Queue {
    has items: list[int] = [];

    def enqueue(x: int) -> None { self.items.append(x); }

    def dequeue() -> int { return self.items.pop(0); }

    def size() -> int { return len(self.items); }
}

with entry {
    q = Queue();
    q.enqueue(1); q.enqueue(2); q.enqueue(3);
    print(q.dequeue());
    print(q.size());
}
'''
    },
    {
        "difficulty": "idiomatic", "expected": "12.57",
        "py": '''
class Circle:
    """A circle."""
    def __init__(self, radius: float):
        self.radius = radius

    def area(self) -> float:
        return 3.14159 * self.radius * self.radius
''',
        "jac": '''
obj Circle {
    has radius: float;

    def area() -> float { return 3.14159 * self.radius * self.radius; }
}

with entry {
    c = Circle(radius=2.0);
    print(round(c.area(), 2));
}
'''
    },
    {
        "difficulty": "idiomatic", "expected": "5.0",
        "py": '''
import math


class Point:
    """A 2D point."""
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def distance_to(self, other: "Point") -> float:
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)
''',
        "jac": '''
import math;

obj Point {
    has x: float;
    has y: float;

    def distance_to(other: Point) -> float {
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2);
    }
}

with entry {
    a = Point(x=0.0, y=0.0);
    b = Point(x=3.0, y=4.0);
    print(a.distance_to(b));
}
'''
    },
    # ---------------- batch 2: composed (graph/walker) ----------------
    {
        "difficulty": "composed", "expected": "['a', 'b', 'c']",
        "py": '''
def collect_names(graph: dict, start: str) -> list:
    """Collect all node names reachable from start, sorted."""
    seen = []
    stack = [start]
    while stack:
        node = stack.pop()
        seen.append(node)
        stack.extend(graph.get(node, []))
    return sorted(seen)
''',
        "jac": '''
node GNode {
    has name: str;
}

walker CollectNames {
    has names: list[str] = [];

    can grab with GNode entry {
        self.names.append(here.name);
        visit [-->];
    }
}

with entry {
    a = GNode(name="a");
    b = GNode(name="b");
    c = GNode(name="c");
    root ++> a;
    a ++> b;
    a ++> c;
    w = a spawn CollectNames();
    print(sorted(w.names));
}
'''
    },
    {
        "difficulty": "composed", "expected": "True",
        "py": '''
def can_reach(graph: dict, start: str, target: str) -> bool:
    """Return True if target is reachable from start."""
    stack = [start]
    while stack:
        node = stack.pop()
        if node == target:
            return True
        stack.extend(graph.get(node, []))
    return False
''',
        "jac": '''
node RNode {
    has name: str;
}

walker Reach {
    has target: str;
    has found: bool = False;

    can look with RNode entry {
        if here.name == self.target {
            self.found = True;
            disengage;
        }
        visit [-->];
    }
}

with entry {
    a = RNode(name="a");
    b = RNode(name="b");
    c = RNode(name="c");
    root ++> a;
    a ++> b;
    b ++> c;
    w = a spawn Reach(target="c");
    print(w.found);
}
'''
    },
    {
        "difficulty": "composed", "expected": "6",
        "py": '''
def sum_even(graph: dict, start: str, vals: dict) -> int:
    """Sum even node values reachable from start."""
    total = 0
    stack = [start]
    while stack:
        node = stack.pop()
        if vals[node] % 2 == 0:
            total += vals[node]
        stack.extend(graph.get(node, []))
    return total
''',
        "jac": '''
node VNode {
    has val: int;
}

walker SumEven {
    has total: int = 0;

    can add with VNode entry {
        if here.val % 2 == 0 { self.total += here.val; }
        visit [-->];
    }
}

with entry {
    a = VNode(val=1);
    b = VNode(val=2);
    c = VNode(val=3);
    d = VNode(val=4);
    root ++> a;
    a ++> b;
    b ++> c;
    c ++> d;
    w = a spawn SumEven();
    print(w.total);
}
'''
    },
    {
        "difficulty": "composed", "expected": "3",
        "py": '''
def count_reports(org: dict, manager: str) -> int:
    """Count all employees under manager (transitively)."""
    count = 0
    stack = list(org.get(manager, []))
    while stack:
        emp = stack.pop()
        count += 1
        stack.extend(org.get(emp, []))
    return count
''',
        "jac": '''
node Employee {
    has name: str;
}

walker CountReports {
    has count: int = 0;
    has started: bool = False;

    can tally with Employee entry {
        if self.started { self.count += 1; } else { self.started = True; }
        visit [-->];
    }
}

with entry {
    ceo = Employee(name="ceo");
    a = Employee(name="a");
    b = Employee(name="b");
    c = Employee(name="c");
    root ++> ceo;
    ceo ++> a;
    ceo ++> b;
    a ++> c;
    w = ceo spawn CountReports();
    print(w.count);
}
'''
    },
    # ---------------- batch 3: MINED (real corpus: Vezora/Tested-22k-Python-Alpaca) ----------------
    {
        "difficulty": "atomic", "expected": "True\nFalse", "source": "Vezora/Tested-22k-Python-Alpaca",
        "py": '''
def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True
''',
        "jac": '''
def is_prime(n: int) -> bool {
    if n <= 1 { return False; }
    for i in range(2, int(n ** 0.5) + 1) {
        if n % i == 0 { return False; }
    }
    return True;
}

with entry {
    print(is_prime(7));
    print(is_prime(8));
}
'''
    },
    {
        "difficulty": "atomic", "expected": "2", "source": "Vezora/Tested-22k-Python-Alpaca",
        "py": '''
def count_unique_consonants(string):
    consonants = set()
    lowercase_string = string.lower()
    for char in lowercase_string:
        if char.isalpha() and char not in 'aeiou':
            consonants.add(char)
    return len(consonants)
''',
        "jac": '''
def count_unique_consonants(s: str) -> int {
    consonants: set = set();
    for char in s.lower() {
        if char.isalpha() and char not in "aeiou" { consonants.add(char); }
    }
    return len(consonants);
}

with entry {
    print(count_unique_consonants("hello"));
}
'''
    },
    {
        "difficulty": "idiomatic", "expected": "helo", "source": "Vezora/Tested-22k-Python-Alpaca",
        "py": '''
def removeDuplicates(s):
    result = ''
    encountered = 0
    for char in s:
        ascii_value = ord(char)
        if not (encountered & (1 << ascii_value)):
            result += char
            encountered |= 1 << ascii_value
    return result
''',
        "jac": '''
def remove_duplicates(s: str) -> str {
    result = "";
    seen: set = set();
    for char in s {
        if char not in seen {
            result += char;
            seen.add(char);
        }
    }
    return result;
}

with entry {
    print(remove_duplicates("hello"));
}
'''
    },
    {
        "difficulty": "idiomatic", "expected": "10", "source": "Vezora/Tested-22k-Python-Alpaca",
        "py": '''
def sum_prime_numbers(arr):
    def is_prime(n):
        if n <= 1:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True

    prime_sum = 0
    for num in arr:
        if is_prime(num):
            prime_sum += num
    return prime_sum
''',
        "jac": '''
def is_prime(n: int) -> bool {
    if n <= 1 { return False; }
    for i in range(2, int(n ** 0.5) + 1) {
        if n % i == 0 { return False; }
    }
    return True;
}

def sum_prime_numbers(nums: list[int]) -> int {
    total = 0;
    for num in nums {
        if is_prime(num) { total += num; }
    }
    return total;
}

with entry {
    print(sum_prime_numbers([2, 3, 4, 5, 6]));
}
'''
    },
    {
        "difficulty": "idiomatic", "expected": "[5, 4, 3, 2, 1]", "source": "Vezora/Tested-22k-Python-Alpaca",
        "py": '''
def bubble_sort_descending(arr):
    n = len(arr)
    for i in range(n - 1):
        for j in range(n - i - 1):
            if arr[j] < arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
''',
        "jac": '''
def bubble_sort_descending(arr: list[int]) -> list[int] {
    n = len(arr);
    for i in range(n - 1) {
        for j in range(n - i - 1) {
            if arr[j] < arr[j + 1] { (arr[j], arr[j + 1]) = (arr[j + 1], arr[j]); }
        }
    }
    return arr;
}

with entry {
    print(bubble_sort_descending([3, 1, 2, 5, 4]));
}
'''
    },
    {
        "difficulty": "atomic", "expected": "20", "source": "Vezora/Tested-22k-Python-Alpaca",
        "py": '''
def average_temperature(temperatures):
    if len(temperatures) == 0:
        raise ValueError("No temperatures provided.")
    total = sum(temperatures)
    average = total / len(temperatures)
    return round(average)
''',
        "jac": '''
def average_temperature(temps: list[int]) -> int {
    if len(temps) == 0 { raise ValueError("No temperatures provided."); }
    total = sum(temps);
    return round(total / len(temps));
}

with entry {
    print(average_temperature([10, 20, 30]));
}
'''
    },
    {
        "difficulty": "atomic", "expected": "HelloWorld", "source": "Vezora/Tested-22k-Python-Alpaca",
        "py": '''
def remove_spaces_and_punctuation(s):
    result = ""
    for char in s:
        if char.isalnum():
            result += char
    return result
''',
        "jac": '''
def remove_spaces_and_punctuation(s: str) -> str {
    result = "";
    for char in s {
        if char.isalnum() { result += char; }
    }
    return result;
}

with entry {
    print(remove_spaces_and_punctuation("Hello, World!"));
}
'''
    },
    {
        "difficulty": "atomic", "expected": "720", "source": "Vezora/Tested-22k-Python-Alpaca",
        "py": '''
def calculate_factorial(n):
    if n == 0:
        return 1
    else:
        return n * calculate_factorial(n - 1)
''',
        "jac": '''
def calculate_factorial(n: int) -> int {
    if n == 0 { return 1; }
    return n * calculate_factorial(n - 1);
}

with entry {
    print(calculate_factorial(6));
}
'''
    }
];

# DPO: chosen = idiomatic graph-spatial Jac; rejected = Python-shaped Jac
# (compiles, but ignores walkers/nodes -- the de-Python-ification negative).
glob DPO_PAIRS: list[dict] = [
    {
        "prompt": "Convert this Python to idiomatic Jac:\n\n```python\ndef total_population(graph: dict, start: str, pops: dict) -> int:\n    seen = set()\n    stack = [start]\n    total = 0\n    while stack:\n        city = stack.pop()\n        if city in seen:\n            continue\n        seen.add(city)\n        total += pops[city]\n        stack.extend(graph.get(city, []))\n    return total\n```",
        "chosen": "```jac\nnode City {\n    has name: str;\n    has population: int = 0;\n}\n\nwalker SumPop {\n    has total: int = 0;\n\n    can tally with City entry {\n        self.total += here.population;\n        visit [-->];\n    }\n}\n```",
        "rejected": "```jac\ndef total_population(graph: dict, start: str, pops: dict) -> int {\n    seen = set();\n    stack = [start];\n    total = 0;\n    while stack {\n        city = stack.pop();\n        if city in seen { continue; }\n        seen.add(city);\n        total += pops[city];\n        stack.extend(graph.get(city, []));\n    }\n    return total;\n}\n```"
    },
    {
        "prompt": "Convert this Python to idiomatic Jac:\n\n```python\ndef count_nodes(tree: dict, start: str) -> int:\n    count = 0\n    stack = [start]\n    while stack:\n        node = stack.pop()\n        count += 1\n        stack.extend(tree.get(node, []))\n    return count\n```",
        "chosen": "```jac\nnode TreeNode {\n    has val: int;\n}\n\nwalker NodeCounter {\n    has count: int = 0;\n\n    can tally with TreeNode entry {\n        self.count += 1;\n        visit [-->];\n    }\n}\n```",
        "rejected": "```jac\ndef count_nodes(tree: dict, start: str) -> int {\n    count = 0;\n    stack = [start];\n    while stack {\n        node = stack.pop();\n        count += 1;\n        stack.extend(tree.get(node, []));\n    }\n    return count;\n}\n```"
    }
];


def build() {
    Path(SFT_PATH).parent.mkdir(parents=True, exist_ok=True);
    open(SFT_PATH, "w").close();
    open(DPO_PATH, "w").close();

    # build all candidate SFT examples; keep difficulty labels in parallel
    candidates: list[dict] = [];
    labels: list[str] = [];
    for s in SEEDS {
        py: str = str(s["py"]);
        jaccode: str = str(s["jac"]);
        difficulty: str = str(s["difficulty"]);
        expected: str = str(s["expected"]);
        src: str = str(s.get("source", "generated_seed"));
        candidates.append(make_sft_example(py, jaccode, difficulty, expected, src));
        labels.append(difficulty);
    }

    counts: dict[str, int] = {};
    for lab in labels { counts[lab] = counts.get(lab, 0) + 1; }

    # cross-example near-duplicate guard (ROUGE-L >= 0.9 on the Jac body).
    # NOTE: a low threshold collapses distinct walker tasks that share node/
    # walker skeleton, so this stays high; it only catches near-identical dups.
    (examples, removed) = dedup_examples(candidates, 0.9);
    if removed > 0 { print("dedup: dropped " + str(removed) + " near-duplicate(s)"); }

    for ex in examples { append_jsonl(SFT_PATH, ex); }

    for d in DPO_PAIRS {
        prompt: str = str(d["prompt"]);
        chosen: str = str(d["chosen"]);
        rejected: str = str(d["rejected"]);
        append_jsonl(DPO_PATH, {
            "prompt": prompt, "chosen": chosen, "rejected": rejected,
            "meta": {
                "category": "conversion", "recipe": "R3",
                "generator": "claude-code-mcp", "rejected_style": "python_shaped_jac"
            }
        });
    }

    print("by difficulty (pre-dedup): " + str(counts));
    print("wrote " + str(len(examples)) + " SFT -> " + SFT_PATH);
    print("wrote " + str(len(DPO_PAIRS)) + " DPO -> " + DPO_PATH);
    print("re-validating SFT examples through `jac run`...");
    failures: int = 0;
    for (i, ex) in enumerate(examples) {
        (ok, detail) = revalidate_example(ex);
        tag: str = "PASS";
        if not ok { tag = "FAIL"; failures += 1; }
        print("  [" + tag + "] #" + str(i) + "  " + detail);
    }
    passed: int = len(examples) - failures;
    print(str(passed) + "/" + str(len(examples)) + " re-validated, " + str(failures) + " failed");
}


with entry {
    build();
}
