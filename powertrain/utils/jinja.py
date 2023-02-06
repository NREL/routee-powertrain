from pathlib import Path
from typing import Any, Dict

from powertrain.utils.minify import minify_source

TEMPLATE_PATH = Path(__file__).parent.parent / "resources" / "templates"
INPUT_PRECISION = 4
OUTPUT_PRECISION = 8
JINJA_DEFAULTS = {
    "f": {
        "enumerate": enumerate,
        "round_inputs": lambda x: round(x, INPUT_PRECISION),
        "round_outputs": lambda x: round(x, OUTPUT_PRECISION),
    },
}


def jinja(template_file: str, data: Dict[str, Any]):
    try:
        from jinja2 import Environment, FileSystemLoader
    except ImportError:
        raise ImportError("package jinja2 is required for this function")

    loader = FileSystemLoader(TEMPLATE_PATH)
    template = Environment(loader=loader).get_template(template_file)
    data = {k: v for k, v in data.items() if v is not None}
    data = {
        **JINJA_DEFAULTS,
        **data,
    }
    code = template.render(data)

    mini_code = minify_source(code)

    return mini_code
