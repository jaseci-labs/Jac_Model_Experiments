"""CF (catastrophic-forgetting) regression task bank.

Deliberately NOT Jac -- this checks whether CPT hurt general Python coding
ability, per design.md's CF guard ("any regression is a stop signal").
16 classic small problems, each with a fixed entry_point and 2-4 exact-match
test cases. Grading (in run_cf_check.py) extracts the model's python code,
execs it in a subprocess with a timeout, calls entry_point per test case,
compares to the expected value exactly.
"""

TASKS = [
    {
        "id": "is_prime", "entry_point": "is_prime",
        "prompt": "Write a Python function `is_prime(n)` that returns True if n is a prime number, False otherwise.",
        "tests": [((2,), True), ((1,), False), ((17,), True), ((18,), False)],
    },
    {
        "id": "reverse_string", "entry_point": "reverse_string",
        "prompt": "Write a Python function `reverse_string(s)` that returns the string s reversed.",
        "tests": [(("hello",), "olleh"), (("",), ""), (("a",), "a")],
    },
    {
        "id": "fibonacci", "entry_point": "fibonacci",
        "prompt": "Write a Python function `fibonacci(n)` that returns the nth Fibonacci number (0-indexed: fibonacci(0)=0, fibonacci(1)=1).",
        "tests": [((0,), 0), ((1,), 1), ((10,), 55)],
    },
    {
        "id": "factorial", "entry_point": "factorial",
        "prompt": "Write a Python function `factorial(n)` that returns n factorial.",
        "tests": [((0,), 1), ((5,), 120), ((1,), 1)],
    },
    {
        "id": "is_palindrome", "entry_point": "is_palindrome",
        "prompt": "Write a Python function `is_palindrome(s)` that returns True if s reads the same forwards and backwards, False otherwise.",
        "tests": [(("racecar",), True), (("hello",), False), (("",), True)],
    },
    {
        "id": "bubble_sort", "entry_point": "bubble_sort",
        "prompt": "Write a Python function `bubble_sort(lst)` that returns a new list with the elements of lst sorted in ascending order.",
        "tests": [(([3, 1, 2],), [1, 2, 3]), (([],), []), (([5, 5, 1],), [1, 5, 5])],
    },
    {
        "id": "count_vowels", "entry_point": "count_vowels",
        "prompt": "Write a Python function `count_vowels(s)` that returns the count of vowels (a, e, i, o, u, case-insensitive) in s.",
        "tests": [(("hello",), 2), (("xyz",), 0), (("AEIOU",), 5)],
    },
    {
        "id": "max_of_list", "entry_point": "max_of_list",
        "prompt": "Write a Python function `max_of_list(lst)` that returns the maximum value in lst.",
        "tests": [(([1, 5, 3],), 5), (([-1, -5, -3],), -1), (([7],), 7)],
    },
    {
        "id": "flatten", "entry_point": "flatten",
        "prompt": "Write a Python function `flatten(nested)` that flattens a list of lists (one level deep) into a single list, preserving order.",
        "tests": [(([[1, 2], [3], [4, 5]],), [1, 2, 3, 4, 5]), (([[], [1]],), [1])],
    },
    {
        "id": "remove_duplicates", "entry_point": "remove_duplicates",
        "prompt": "Write a Python function `remove_duplicates(lst)` that returns a new list with duplicate values removed, preserving the first occurrence's order.",
        "tests": [(([1, 2, 2, 3, 1],), [1, 2, 3]), (([],), [])],
    },
    {
        "id": "gcd", "entry_point": "gcd",
        "prompt": "Write a Python function `gcd(a, b)` that returns the greatest common divisor of a and b.",
        "tests": [((12, 18), 6), ((7, 13), 1), ((0, 5), 5)],
    },
    {
        "id": "binary_search", "entry_point": "binary_search",
        "prompt": "Write a Python function `binary_search(sorted_list, target)` that returns the index of target in sorted_list (ascending order), or -1 if not found.",
        "tests": [(([1, 3, 5, 7, 9], 5), 2), (([1, 3, 5], 4), -1), (([], 1), -1)],
    },
    {
        "id": "is_anagram", "entry_point": "is_anagram",
        "prompt": "Write a Python function `is_anagram(s1, s2)` that returns True if s1 and s2 are anagrams of each other, False otherwise.",
        "tests": [(("listen", "silent"), True), (("hello", "world"), False)],
    },
    {
        "id": "sum_digits", "entry_point": "sum_digits",
        "prompt": "Write a Python function `sum_digits(n)` that returns the sum of the digits of non-negative integer n.",
        "tests": [((123,), 6), ((0,), 0), ((9999,), 36)],
    },
    {
        "id": "merge_dicts", "entry_point": "merge_dicts",
        "prompt": "Write a Python function `merge_dicts(d1, d2)` that returns a new dict merging d1 and d2, where d2's values override d1's on key conflicts.",
        "tests": [(({"a": 1, "b": 2}, {"b": 3, "c": 4}), {"a": 1, "b": 3, "c": 4})],
    },
    {
        "id": "caesar_cipher", "entry_point": "caesar_cipher",
        "prompt": "Write a Python function `caesar_cipher(s, shift)` that shifts each lowercase letter in s forward by shift positions in the alphabet (wrapping z->a), leaving non-lowercase-letter characters unchanged.",
        "tests": [(("abc", 1), "bcd"), (("xyz", 3), "abc"), (("hi there", 0), "hi there")],
    },
]
