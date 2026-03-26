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

?cmp_atom: ESCAPED_STRING                      -> quoted_value
         | phrase

phrase: WORD+                                  -> phrase

FIELD: /[a-zA-Z_][a-zA-Z0-9_.-]*/
OP: ":" | "=" | ">=" | "<=" | ">" | "<"

AND.10: "AND"
OR.10: "OR"
NOT.10: "NOT"

WORD: /[^()\s:><="]+/

%import common.ESCAPED_STRING
%import common.WS
%ignore WS
"""


TABLE_NAME = "products"
FTS_TABLE_NAME = "products_fts"
PRIMARY_KEY = "id"

TEXT_FIELDS = {
    "contractor_name",
    "product",
    "unit",
    "bid_package_name",
    "investor",
    "manufacturer",
    "country_of_origin",
    "region_of_origin",
    "province"
}

NUMERIC_FIELDS = {
    "quantity",
    "unit_price",
    "total_price",
}

DATE_FIELDS = {
    "posting_date",
    "closing_date",
}

ALL_QUERY_FIELDS = TEXT_FIELDS | NUMERIC_FIELDS | DATE_FIELDS


@dataclass
class Phrase:
    value: str


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

    def phrase(self, items: list[str]) -> Phrase:
        return Phrase(" ".join(items))

    def quoted_value(self, items: list[Any]) -> Phrase:
        raw = str(items[0])
        return Phrase(ast.literal_eval(raw))    
        
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
    

def _unicode_casefold(s: Any) -> Any:
    if s is None:
        return None
    return unicodedata.normalize("NFC", str(s)).casefold()


def _normalize_date_ddmmyy_to_iso(value: str) -> str:
    return datetime.strptime(value, "%d-%m-%Y").strftime("%Y-%m-%d %H:%M:%S")


def _quote(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _coerce_numeric(value: Any, field: str) -> Union[int, float]:
    text = value.value
    try:
        return float(text) if any(ch in text for ch in ".eE") else int(text)
    except ValueError as exc:
        raise SQLCompileError(f"Invalid numeric value {value!r} for field {field!r}") from exc


def _coerce_date(value: Any, field: str) -> str:
    text = value.value
    try:
        return _normalize_date_ddmmyy_to_iso(text)
    except ValueError as exc:
        raise SQLCompileError(f"Invalid date value {value!r} for field {field!r}; expected DD-MM-YYYY") from exc


def _compile_fts_single_field_(node: Any, field: str) -> tuple[str, list[Any]]: 
    if isinstance(node, And):
        parts, params = [], []
        for child in node.args:
            child_sql, child_params = _compile_fts_single_field_(child, field)
            parts.append(f"({child_sql})")
            params.extend(child_params)
        return " AND ".join(parts), params

    if isinstance(node, Or):
        parts, params = [], []
        for child in node.args:
            child_sql, child_params = _compile_fts_single_field_(child, field)
            parts.append(f"({child_sql})")
            params.extend(child_params)
        return " OR ".join(parts), params

    if isinstance(node, Not):
        child_sql, child_params = _compile_fts_single_field_(node.arg, field)
        return f"NOT ({child_sql})", child_params

    if isinstance(node, Phrase):
        sql = (
            f"EXISTS ("
            f"SELECT 1 FROM {_quote(FTS_TABLE_NAME)} "
            f"WHERE {_quote(FTS_TABLE_NAME)}.rowid = "
            f"{_quote(TABLE_NAME)}.{_quote(PRIMARY_KEY)} "
            f"AND {_quote(FTS_TABLE_NAME)} MATCH ?"
            f")"
        )
        return sql, [f'{field}:{_quote(node.value)}']
    
    raise SQLCompileError(f"Unsupported full-text search expression: {node!r}")


def _compile_exact_expr(node: Any, column_sql: str, field: str) -> tuple[str, list[Any]]:
    if isinstance(node, And):
        raise SQLCompileError(f"Operator '=' (exact match) does not support AND expressions. Use ':' (contains) instead.")

    if isinstance(node, Or):
        parts, params = [], []
        for child in node.args:
            child_sql, child_params = _compile_exact_expr(child, column_sql, field)
            parts.append(f"({child_sql})")
            params.extend(child_params)
        return " OR ".join(parts), params

    if isinstance(node, Not):
        child_sql, child_params = _compile_exact_expr(node.arg, column_sql, field)
        return f"NOT ({child_sql})", child_params

    if isinstance(node, Phrase):
        if field in TEXT_FIELDS:
            return f"CASEFOLD({column_sql}) = CASEFOLD(?)", [node.value]
        elif field in NUMERIC_FIELDS:
            return f"{column_sql} = ?", [_coerce_numeric(node, field)]
        elif field in DATE_FIELDS:
            return f"{column_sql} = ?", [_coerce_date(node, field)]
    
    raise SQLCompileError(f"Unsupported exact-match expression: {node!r}")


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
        
        column_sql = f'{_quote(TABLE_NAME)}.{_quote(field)}'

        if field in TEXT_FIELDS:
            if op == ":":
                return _compile_fts_single_field_(value, field)
    
            if op == "=":
                return _compile_exact_expr(value, column_sql, field)

            raise SQLCompileError(f"Operator {op!r} is invalid for text field {field!r}. Use ':' or '='.")

        if field in NUMERIC_FIELDS:
            if op not in ("=", ">", ">=", "<", "<="):
                raise SQLCompileError(f"Operator {op!r} is invalid for numeric field {field!r}. Use '=', '>', '>=', '<', or '<='.")
            
            if op == "=":
                return _compile_exact_expr(value, column_sql, field)
            
            value = _ensure_scalar_rhs(value, field, op)
            return f'{column_sql} {op} ?', [_coerce_numeric(value, field)]

        if field in DATE_FIELDS:
            if op not in ("=", ">", ">=", "<", "<="):
                raise SQLCompileError(f"Operator {op!r} is invalid for date field {field!r}. Use '=', '>', '>=', '<', or '<='.")
            
            if op == "=":
                return _compile_exact_expr(value, column_sql, field) 
            
            value = _ensure_scalar_rhs(value, field, op)
            return f'{column_sql} {op} ?', [_coerce_date(value, field)]

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
    SELECT {_quote(TABLE_NAME)}.*
    FROM {_quote(TABLE_NAME)}
    WHERE {where_sql}
    ORDER BY id
    """.strip()
    return sql, params


def compile_query_to_sql(query: str) -> tuple[str, list[Any]]:
    return compile_ast_to_sql(parse_query(query))


def query(query: str):
    conn = sqlite3.connect("products.db")
    conn.create_function("CASEFOLD", 1, _unicode_casefold)
    cursor = conn.cursor()

    sql, params = compile_query_to_sql(query)

    df = pd.read_sql_query(sql, conn, params=params)

    conn.close()

    return df
