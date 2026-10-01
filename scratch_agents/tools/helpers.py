import inspect
import json
from types import SimpleNamespace
from typing import Any, get_type_hints

from pydantic import TypeAdapter


def function_to_input_schema(func, *, localns=None) -> dict:
    """Build general JSON Schema for a tool called with func(**arguments).

    Supports Pydantic-compatible annotations, nested containers, unions,
    Literal/Enum, Annotated constraints, dataclasses, TypedDict and models.
    Shared definitions preserve recursive references and model-name collisions.
    Missing annotations mean Any. Defaults determine requiredness but are not
    emitted: Python applies them at call time. Injected self/context are skipped.
    Unsupported signatures/types and unresolved references raise TypeError.
    Pass localns for string references to locally defined types.

    This does not deserialize arguments or enforce provider-specific schema
    restrictions. Only use trusted functions: resolving annotations evaluates
    their forward-reference expressions.
    """
    parameters = {
        name: param for name, param in inspect.signature(func).parameters.items()
        if name not in ("self", "context")
    }
    for name, param in parameters.items():
        if param.kind not in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            raise TypeError(f"Parameter {name!r} cannot be passed as a named tool argument")

    # Resolve only exposed inputs, excluding return and injected context hints.
    annotations = {
        name: param.annotation for name, param in parameters.items()
        if param.annotation is not inspect.Parameter.empty
    }
    target = inspect.unwrap(func)
    try:
        hints = get_type_hints(
            SimpleNamespace(__annotations__=annotations),
            globalns=getattr(target, "__globals__", {}),
            localns=localns, include_extras=True,
        )
    except Exception as exc:
        raise TypeError(f"Cannot resolve input annotations: {exc}") from exc

    adapters = []
    for name in parameters:
        try:
            adapter = TypeAdapter(hints.get(name, Any))
        except Exception as exc:
            raise TypeError(f"Unsupported annotation for parameter {name!r}: {exc}") from exc
        adapters.append((name, "validation", adapter))
    try:
        fields, definitions = TypeAdapter.json_schemas(adapters)
    except Exception as exc:
        raise TypeError(f"Cannot generate input JSON Schema: {exc}") from exc

    properties = {}
    for name in parameters:
        prop = dict(fields[(name, "validation")])
        prop.setdefault("description", f"Parameter: {name}")
        properties[name] = prop
    schema = {"type": "object", "properties": properties, **definitions}
    required = [name for name, param in parameters.items()
                if param.default is inspect.Parameter.empty]
    if required:
        schema["required"] = required
    return schema


def format_tool_definition(name: str, description: str, parameters: dict) -> dict:
    """Format a tool definition in the OpenAI function calling format."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        }
    }


def function_to_tool_definition(func) -> dict:
    """Convert a Python function to an OpenAI-format tool definition.

    Uses the function name, docstring, and type hints.
    """
    name = func.__name__
    description = inspect.getdoc(func) or f"Function: {name}"
    parameters = function_to_input_schema(func)
    return format_tool_definition(name, description, parameters)


def tool_execution(tool_box: dict, tool_call) -> str:
    """Execute a tool call using a tool_box mapping.

    Args:
        tool_box: Dict mapping tool names to callables
        tool_call: Tool call object with function.name and function.arguments
    """
    func_name = tool_call.function.name
    if func_name not in tool_box:
        return f"Error: Unknown tool '{func_name}'"

    try:
        args = json.loads(tool_call.function.arguments)
        result = tool_box[func_name](**args)
        return str(result)
    except Exception as e:
        return f"Error executing {func_name}: {str(e)}"
