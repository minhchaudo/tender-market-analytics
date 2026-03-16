from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, List, Union

from lark import Lark, Token, Transformer
import sqlite3
import pandas as pd
import unicodedata
import ast

GRAMMAR = r"""
?start: expr

?expr: or_expr

?or_expr: and_expr
        | or_expr OR and_expr                  -> or_op

?and_expr: not_expr
         | and_expr AND not_expr               -> and_op

?not_expr: NOT not_expr                        -> not_op
         | atom

?atom: comparison
     | "(" expr ")"

comparison: FIELD OP cmp_expr                  -> comparison

?cmp_expr: cmp_or

?cmp_or: cmp_and
       | cmp_or OR cmp_and                     -> or_op

?cmp_and: cmp_not
        | cmp_and AND cmp_not                  -> and_op

?cmp_not: NOT cmp_not                          -> not_op
        | cmp_term

?cmp_term: cmp_atom
         | "(" cmp_expr ")"

?cmp_atom: SIGNED_NUMBER                       -> number
         | DATE                                -> date_value
         | ESCAPED_STRING                      -> quoted_value
         | phrase

phrase: WORD+                                  -> phrase

FIELD: /[a-zA-Z_][a-zA-Z0-9_.-]*/
OP: ":" | "=" | ">=" | "<=" | ">" | "<"

DATE.20: /\d{2}-\d{2}-\d{2}/

AND.10: "AND"
OR.10: "OR"
NOT.10: "NOT"

WORD: /[^()\s:><="]+/

%import common.SIGNED_NUMBER
%import common.ESCAPED_STRING
%import common.WS
%ignore WS
"""


TABLE_NAME = "products"
FTS_TABLE_NAME = "products_fts"
PRIMARY_KEY = "id"

TEXT_FIELDS = {
    "bid_name",
    "investor",
    "location",
    "winner",
    "item_name",
    "category",
    "manufacturer",
    "origin",
    "unit",
}

NUMERIC_FIELDS = {
    "quantity",
    "unit_price",
    "total_price",
}

DATE_FIELDS = {
    "posting_time",
    "closing_time",
}

ALL_QUERY_FIELDS = TEXT_FIELDS | NUMERIC_FIELDS | DATE_FIELDS


@dataclass
class Phrase:
    value: str


@dataclass
class DateValue:
    raw: str


@dataclass
class Comparison:
    field: str
    op: str
    value: Any


@dataclass
class And:
    args: List[Any]


@dataclass
class Or:
    args: List[Any]


@dataclass
class Not:
    arg: Any


class QueryTransformer(Transformer):
    def FIELD(self, t: Token) -> str:
        return str(t)

    def OP(self, t: Token) -> str:
        return str(t)

    def WORD(self, t: Token) -> str:
        return str(t)

    def DATE(self, t: Token) -> str:
        return str(t)

    def number(self, items: list[Any]) -> Union[int, float]:
        s = str(items[0])
        if "." in s or "e" in s.lower():
            return float(s)
        return int(s)

    def phrase(self, items: list[str]) -> Phrase:
        return Phrase(" ".join(items))

    def quoted_value(self, items: list[Any]) -> Phrase:
        raw = str(items[0])
        return Phrase(ast.literal_eval(raw))    
        
    def date_value(self, items: list[Any]) -> DateValue:
        return DateValue(str(items[0]))

    def comparison(self, items: list[Any]) -> Comparison:
        field, op, value = items
        return Comparison(field=field, op=op, value=value)

    def and_op(self, items: list[Any]) -> And:
        args: list[Any] = []
        for item in items:
            if isinstance(item, Token):
                continue
            if isinstance(item, And):
                args.extend(item.args)
            else:
                args.append(item)
        return And(args)

    def or_op(self, items: list[Any]) -> Or:
        args: list[Any] = []
        for item in items:
            if isinstance(item, Token):
                continue
            if isinstance(item, Or):
                args.extend(item.args)
            else:
                args.append(item)
        return Or(args)

    def not_op(self, items: list[Any]) -> Not:
        non_tokens = [item for item in items if not isinstance(item, Token)]
        return Not(non_tokens[0])


_parser = Lark(GRAMMAR, parser="lalr", maybe_placeholders=False)
_transformer = QueryTransformer()


def parse_query(query: str) -> Any:
    return _transformer.transform(_parser.parse(query))


class SQLCompileError(ValueError):
    pass


def _looks_like_ddmmyy(value: str) -> bool:
    try:
        datetime.strptime(value, "%d-%m-%y")
        return True
    except ValueError:
        return False
    
def _unicode_casefold(s: Any) -> Any:
    if s is None:
        return None
    return unicodedata.normalize("NFC", str(s)).casefold()


def _normalize_date_ddmmyy_to_iso(value: str) -> str:
    return datetime.strptime(value, "%d-%m-%y").strftime("%Y-%m-%d %H:%M:%S")


def _quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _quote_fts_phrase(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _is_text_field(field: str) -> bool:
    return field in TEXT_FIELDS


def _is_numeric_field(field: str) -> bool:
    return field in NUMERIC_FIELDS


def _is_date_field(field: str) -> bool:
    return field in DATE_FIELDS


def _column_for_field(field: str) -> str:
    if field not in ALL_QUERY_FIELDS:
        raise SQLCompileError(f"Unknown field: {field!r}")
    return field


def _coerce_numeric(value: Any, field: str) -> Union[int, float]:
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, DateValue):
        raise SQLCompileError(f"Date literal {value.raw!r} is invalid for numeric field {field!r}")

    text = value.value if isinstance(value, Phrase) else str(value)
    try:
        return float(text) if any(ch in text for ch in ".eE") else int(text)
    except ValueError as exc:
        raise SQLCompileError(f"Invalid numeric value {value!r} for field {field!r}") from exc


def _coerce_date(value: Any, field: str) -> str:
    if isinstance(value, DateValue):
        return _normalize_date_ddmmyy_to_iso(value.raw)

    text = value.value if isinstance(value, Phrase) else str(value)
    if _looks_like_ddmmyy(text):
        return _normalize_date_ddmmyy_to_iso(text)

    raise SQLCompileError(f"Invalid date value {value!r} for field {field!r}; expected DD-MM-YY")


def _coerce_text_scalar(value: Any, field: str) -> str:
    if isinstance(value, (And, Or, Not)):
        raise SQLCompileError(
            f"Operator '=' on text field {field!r} requires text terms"
        )
    if isinstance(value, Phrase):
        return value.value
    if isinstance(value, DateValue):
        return value.raw
    return str(value)

def _compile_text_match_expr(node: Any, field: str) -> str:
    if isinstance(node, str):
        return f'{field}:{_quote_fts_phrase(node)}'

    if isinstance(node, Phrase):
        return f'{field}:{_quote_fts_phrase(node.value)}'

    if isinstance(node, And):
        return "(" + " AND ".join(_compile_text_match_expr(child, field) for child in node.args) + ")"

    if isinstance(node, Or):
        return "(" + " OR ".join(_compile_text_match_expr(child, field) for child in node.args) + ")"

    if isinstance(node, Not):
        return f"(NOT {_compile_text_match_expr(node.arg, field)})"

    raise SQLCompileError(f"Unsupported text MATCH expression: {node!r}")


def _compile_text_exact_expr(node: Any, column_sql: str, field: str) -> tuple[str, list[Any]]:
    if isinstance(node, And):
        parts, params = [], []
        for child in node.args:
            child_sql, child_params = _compile_text_exact_expr(child, column_sql, field)
            parts.append(f"({child_sql})")
            params.extend(child_params)
        return " AND ".join(parts), params

    if isinstance(node, Or):
        parts, params = [], []
        for child in node.args:
            child_sql, child_params = _compile_text_exact_expr(child, column_sql, field)
            parts.append(f"({child_sql})")
            params.extend(child_params)
        return " OR ".join(parts), params

    if isinstance(node, Not):
        child_sql, child_params = _compile_text_exact_expr(node.arg, column_sql, field)
        return f"NOT ({child_sql})", child_params

    return f"CASEFOLD({column_sql}) = CASEFOLD(?)", [_coerce_text_scalar(node, field)]

def _ensure_scalar_rhs(value: Any, field: str, op: str) -> Any:
    if isinstance(value, (And, Or, Not)):
        raise SQLCompileError(
            f"Operator {op!r} on field {field!r} requires a scalar value, not a boolean expression"
        )
    return value


def _compile_node(node: Any) -> tuple[str, list[Any]]:
    if isinstance(node, Comparison):
        field = node.field
        op = node.op
        value = node.value

        if field not in ALL_QUERY_FIELDS:
            raise SQLCompileError(f"Unknown field: {field!r}")

        if _is_text_field(field):
            column_sql = f'{_quote_identifier(TABLE_NAME)}.{_quote_identifier(field)}'

            if op == ":":
                match_expr = _compile_text_match_expr(value, field)
                sql = (
                    f"EXISTS ("
                    f"SELECT 1 FROM {_quote_identifier(FTS_TABLE_NAME)} "
                    f"WHERE {_quote_identifier(FTS_TABLE_NAME)}.rowid = "
                    f"{_quote_identifier(TABLE_NAME)}.{_quote_identifier(PRIMARY_KEY)} "
                    f"AND {_quote_identifier(FTS_TABLE_NAME)} MATCH ?"
                    f")"
                )
                return sql, [match_expr]

            if op == "=":
                return _compile_text_exact_expr(value, column_sql, field)

            raise SQLCompileError(f"Operator {op!r} is invalid for text field {field!r}. Use ':' or '='.")

        column = _column_for_field(field)

        if _is_numeric_field(field):
            if op not in ("=", ">", ">=", "<", "<="):
                raise SQLCompileError(f"Operator {op!r} is invalid for numeric field {field!r}")
            value = _ensure_scalar_rhs(value, field, op)
            return f'{_quote_identifier(TABLE_NAME)}.{_quote_identifier(column)} {op} ?', [_coerce_numeric(value, field)]

        if _is_date_field(field):
            if op not in ("=", ">", ">=", "<", "<="):
                raise SQLCompileError(f"Operator {op!r} is invalid for date field {field!r}")
            value = _ensure_scalar_rhs(value, field, op)
            return f'{_quote_identifier(TABLE_NAME)}.{_quote_identifier(column)} {op} ?', [_coerce_date(value, field)]

        raise SQLCompileError(f"Unknown field: {field!r}")

    if isinstance(node, And):
        parts, params = [], []
        for child in node.args:
            child_sql, child_params = _compile_node(child)
            parts.append(f"({child_sql})")
            params.extend(child_params)
        return " AND ".join(parts), params

    if isinstance(node, Or):
        parts, params = [], []
        for child in node.args:
            child_sql, child_params = _compile_node(child)
            parts.append(f"({child_sql})")
            params.extend(child_params)
        return " OR ".join(parts), params

    if isinstance(node, Not):
        child_sql, child_params = _compile_node(node.arg)
        return f"NOT ({child_sql})", child_params

    raise SQLCompileError(f"Unsupported AST node: {node!r}")


def compile_ast_to_sql(ast: Any) -> tuple[str, list[Any]]:
    where_sql, params = _compile_node(ast)
    sql = f"""
SELECT {_quote_identifier(TABLE_NAME)}.*
FROM {_quote_identifier(TABLE_NAME)}
WHERE {where_sql}
""".strip()
    return sql, params


def compile_query_to_sql(query: str) -> tuple[str, list[Any]]:
    return compile_ast_to_sql(parse_query(query))


if __name__ == "__main__":
    conn = sqlite3.connect("products.db")
    conn.create_function("CASEFOLD", 1, _unicode_casefold)
    cursor = conn.cursor()

    query = '(bid_name: quảng ngãi OR dịch vụ) AND (location: hà nội OR thanh hoá) AND (quantity > 100) AND (unit = "bơm tiêm" OR "lọ")'

    sql, params = compile_query_to_sql(query)

    cursor.execute(sql, params)

    df = pd.read_sql_query(sql, conn, params=params)

    df.to_excel("test.xlsx", index=False)

    conn.close()



