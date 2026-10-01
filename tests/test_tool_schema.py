from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from enum import Enum
from typing import Annotated, Any, Callable, Literal
from uuid import UUID

import pytest
from pydantic import BaseModel, Field
from typing_extensions import NotRequired, TypedDict

from scratch_agents.tools.helpers import function_to_input_schema


class Node(BaseModel):
    value: int
    children: list['Node'] = []


class Options(TypedDict):
    count: int
    label: NotRequired[str]


@dataclass
class Point:
    x: float


class Color(str, Enum):
    RED = 'red'
    BLUE = 'blue'


def test_nested_containers_and_required_nullable():
    def tool(values: list[dict[str, list[float]]], count: int | None, flag: bool = False):
        pass
    schema = function_to_input_schema(tool)
    assert schema['required'] == ['values', 'count']
    assert schema['properties']['values']['items']['additionalProperties']['items']['type'] == 'number'
    assert schema['properties']['count']['anyOf'] == [{'type': 'integer'}, {'type': 'null'}]
    assert schema['properties']['flag']['type'] == 'boolean'


def test_structured_types_and_recursive_references():
    def tool(node: Node, nodes: list[Node], options: Options, point: Point):
        pass
    schema = function_to_input_schema(tool)
    definitions = schema['$defs']
    def check_refs(value):
        if isinstance(value, dict):
            if '$ref' in value:
                assert value['$ref'].removeprefix('#/$defs/') in definitions
            for child in value.values():
                check_refs(child)
        elif isinstance(value, list):
            for child in value:
                check_refs(child)
    check_refs(schema)
    assert definitions['Options']['required'] == ['count']
    assert definitions['Point']['properties']['x']['type'] == 'number'


def test_constraints_and_special_types():
    def tool(size: Annotated[int, Field(gt=0, description='Positive size')],
             mode: Literal['a', 'b'], color: Color, pair: tuple[int, str],
             tags: set[str], day: date, uid: UUID, amount: Decimal, data: bytes):
        pass
    schema = function_to_input_schema(tool)
    props = schema['properties']
    assert props['size']['exclusiveMinimum'] == 0
    assert props['size']['description'] == 'Positive size'
    assert props['mode']['enum'] == ['a', 'b']
    assert schema['$defs']['Color']['enum'] == ['red', 'blue']
    assert props['pair']['prefixItems'] == [{'type': 'integer'}, {'type': 'string'}]
    assert props['tags']['uniqueItems'] is True
    assert props['day']['format'] == 'date'
    assert props['uid']['format'] == 'uuid'
    assert 'anyOf' in props['amount']
    assert props['data']['type'] == 'string'


def test_missing_hints_and_injected_context():
    def tool(context: 'MissingContext', value, anything: Any, /):
        pass
    with pytest.raises(TypeError, match='named tool argument'):
        function_to_input_schema(tool)
    def tool(context: 'MissingContext', value, anything: Any) -> 'MissingReturn':
        pass
    schema = function_to_input_schema(tool)
    assert set(schema['properties']) == {'value', 'anything'}
    assert 'type' not in schema['properties']['value']
    assert 'type' not in schema['properties']['anything']


def test_local_forward_reference_and_unresolved_hint():
    def tool(value: 'LocalType'):
        pass
    with pytest.raises(TypeError, match='resolve input annotations'):
        function_to_input_schema(tool)
    assert function_to_input_schema(tool, localns={'LocalType': int})['properties']['value']['type'] == 'integer'


class UnsupportedType:
    pass


@pytest.mark.parametrize('annotation', [Callable[[int], str], UnsupportedType])
def test_unsupported_annotation_fails(annotation):
    def tool(value):
        pass
    tool.__annotations__ = {'value': annotation}
    with pytest.raises(TypeError):
        function_to_input_schema(tool)


@pytest.mark.parametrize('func', [lambda *args: None, lambda **kwargs: None])
def test_variadic_signature_fails(func):
    with pytest.raises(TypeError, match='named tool argument'):
        function_to_input_schema(func)


def test_no_inputs():
    assert function_to_input_schema(lambda: None) == {'type': 'object', 'properties': {}}
