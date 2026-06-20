"""Output-formatting benchmark: a reference doc of structured-data conventions
(JSON, YAML, protobuf, bulleted lists) the base model wasn't told to follow,
plus held-out tasks that ask for one of those formats. Grading checks
structural validity for the requested `kind`, not exact text.
"""

import json
import re

import yaml

DOC = """\
# Structured Output Format Reference

## JSON objects
Use double-quoted keys and values, no trailing commas, and wrap the whole
object in a single pair of braces. Nest objects/arrays as needed.

Example:
{
  "title": "Atomic Habits",
  "author": "James Clear",
  "year": 2018
}

## YAML mappings
Use `key: value` pairs, one per line, no braces or quotes unless a value
contains a colon or starts with a special character. Indent nested mappings
by 2 spaces. Do not wrap the block in a code fence label other than `yaml`.

Example:
host: 0.0.0.0
port: 8080
debug: true
tags:
  - web
  - prod

## Protobuf message definitions
Declare a `message <Name> { ... }` block. Each field line has the form
`<type> <field_name> = <field_number>;` ending in a semicolon, one field per
line, numbered starting at 1.

Example:
message Point {
  int32 x = 1;
  int32 y = 2;
}

## Bulleted lists
One item per line, each line starting with `- ` (hyphen, space). Do not
number them and do not put multiple items on one line.

Example:
- Improves cardiovascular health
- Reduces stress
- Strengthens muscles
"""

QUESTIONS = [
    dict(
        question="Represent a book with title 'Dune', author 'Frank Herbert', and year 1965 as JSON.",
        kind="json",
        formula="valid JSON object (json.loads succeeds)",
        expected="valid JSON",
    ),
    dict(
        question="Write a YAML block configuring a server with host 0.0.0.0, port 8080, and debug true.",
        kind="yaml",
        formula="YAML key: value mapping, >=2 keys",
        expected="valid YAML mapping",
    ),
    dict(
        question="Define a protobuf message called Point with two int32 fields, x and y.",
        kind="proto",
        formula="message Name { <type> <field> = <n>; ... }",
        expected="valid proto message",
    ),
    dict(
        question="List three benefits of regular exercise as a bulleted list.",
        kind="bullets",
        formula="one '- ' item per line, >=3 lines",
        expected="3+ bullet lines",
    ),
    dict(
        question="Represent a shopping cart with two items (name and price) as JSON.",
        kind="json",
        formula="valid JSON object or array (json.loads succeeds)",
        expected="valid JSON",
    ),
]

_FENCE_RE = re.compile(r"```(?:\w+)?\n?(.*?)```", re.DOTALL)
_PROTO_FIELD_RE = re.compile(r"\b\w[\w.]*\s+\w+\s*=\s*\d+\s*;")
_BULLET_LINE_RE = re.compile(r"(?m)^\s*[-*]\s+\S")


def _strip_fences(text: str) -> str:
    m = _FENCE_RE.search(text)
    return m.group(1).strip() if m else text.strip()


def _check_json(text: str) -> bool:
    body = _strip_fences(text)
    start = min((i for i in (body.find("{"), body.find("[")) if i != -1), default=-1)
    if start == -1:
        return False
    for end in range(len(body), start, -1):
        try:
            json.loads(body[start:end])
            return True
        except ValueError:
            continue
    return False


def _check_yaml(text: str) -> bool:
    body = _strip_fences(text)
    try:
        data = yaml.safe_load(body)
    except yaml.YAMLError:
        return False
    return isinstance(data, dict) and len(data) >= 2


def _check_proto(text: str) -> bool:
    body = _strip_fences(text)
    if not re.search(r"message\s+\w+\s*\{", body):
        return False
    return len(_PROTO_FIELD_RE.findall(body)) >= 1


def _check_bullets(text: str) -> bool:
    body = _strip_fences(text)
    return len(_BULLET_LINE_RE.findall(body)) >= 3


_CHECKERS = {
    "json": _check_json,
    "yaml": _check_yaml,
    "proto": _check_proto,
    "bullets": _check_bullets,
}


def is_correct(answer: str, kind: str) -> bool:
    return _CHECKERS[kind](answer)
