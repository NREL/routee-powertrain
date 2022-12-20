from typing import Any, Dict
from pathlib import Path


from powertrain.utils.minify import minify_source

TEMPLATE_PATH = Path(__file__).parent.parent / "resources" / "templates"
JINJA_DEFAULTS = {
    "f": {
        "enumerate": enumerate,
    },
}


def jinja(template_file: str, data: Dict[str, Any]):
    try:
        from jinja2 import FileSystemLoader, Environment
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
