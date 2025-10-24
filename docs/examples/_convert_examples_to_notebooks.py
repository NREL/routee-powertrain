from pathlib import Path
import nbformat
import argparse
import sys


def script_to_notebook(script_path: Path, notebook_path: Path) -> None:
    # Read the script
    with open(script_path, "r") as script_file:
        lines = script_file.readlines()

    notebook = nbformat.v4.new_notebook()
    current_code_block: list[str] = []
    current_markdown_block: list[str] = []

    def add_code_cell(block: list[str]) -> None:
        if block:
            notebook.cells.append(nbformat.v4.new_code_cell("".join(block)))

    def add_markdown_cell(block: list[str]) -> None:
        if block:
            notebook.cells.append(nbformat.v4.new_markdown_cell("".join(block).strip()))

    # markdown cells will be enclosed in triple quotes
    # the remaining cells will be code cells
    in_markdown = False
    for line in lines:
        # strip any ipython code blocks
        if line.strip().startswith("# %%"):
            continue
        if line.strip().startswith('"""'):
            in_markdown = not in_markdown
            if in_markdown:
                add_code_cell(current_code_block)
                current_code_block = []
            else:
                add_markdown_cell(current_markdown_block)
                current_markdown_block = []
        elif in_markdown:
            current_markdown_block.append(line)
        else:
            current_code_block.append(line)

    add_code_cell(current_code_block)
    add_markdown_cell(current_markdown_block)

    with open(notebook_path, "w") as notebook_file:
        nbformat.write(notebook, notebook_file)


def notebook_to_script(notebook_path: Path, script_path: Path) -> None:
    """Convert a Jupyter notebook back to a Python script with markdown in triple quotes."""
    # Read the notebook
    with open(notebook_path, "r") as notebook_file:
        notebook = nbformat.read(notebook_file, as_version=4)

    script_lines = []

    for cell in notebook.cells:
        if cell.cell_type == "markdown":
            # Add markdown content wrapped in triple quotes
            script_lines.append('"""\n')
            script_lines.append(cell.source)
            if not cell.source.endswith("\n"):
                script_lines.append("\n")
            script_lines.append('"""\n')
        elif cell.cell_type == "code":
            # Add code content directly
            if cell.source.strip():  # Only add non-empty code cells
                script_lines.append(cell.source)
                if not cell.source.endswith("\n"):
                    script_lines.append("\n")

    # Write the script
    with open(script_path, "w") as script_file:
        script_file.writelines(script_lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert between Python scripts and Jupyter notebooks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert all example scripts to notebooks (default behavior)
  python _convert_examples_to_notebooks.py

  # Convert all example notebooks back to scripts  
  python _convert_examples_to_notebooks.py --to-script

  # Convert specific file to notebook
  python _convert_examples_to_notebooks.py --file my_example.py

  # Convert specific notebook to script
  python _convert_examples_to_notebooks.py --file my_example.ipynb --to-script
        """,
    )

    parser.add_argument(
        "--to-script",
        action="store_true",
        help="Convert notebooks to scripts (default: convert scripts to notebooks)",
    )
    parser.add_argument(
        "--file", type=Path, help="Convert a specific file instead of all examples"
    )

    args = parser.parse_args()
    here = Path(__file__).parent

    if args.file:
        # Convert specific file
        input_file = args.file
        if not input_file.is_absolute():
            input_file = here / input_file

        if not input_file.exists():
            print(f"Error: File {input_file} does not exist")
            sys.exit(1)

        if args.to_script:
            if input_file.suffix != ".ipynb":
                print(f"Error: {input_file} is not a notebook file")
                sys.exit(1)
            output_file = input_file.with_suffix(".py")
            notebook_to_script(input_file, output_file)
            print(f"Converted {input_file} to {output_file}")
        else:
            if input_file.suffix != ".py":
                print(f"Error: {input_file} is not a Python file")
                sys.exit(1)
            output_file = input_file.with_suffix(".ipynb")
            script_to_notebook(input_file, output_file)
            print(f"Converted {input_file} to {output_file}")
    else:
        # Convert all example files
        if args.to_script:
            # Convert all notebooks to scripts
            notebooks = here.glob("*example.ipynb")
            converted_count = 0
            for notebook in notebooks:
                script = notebook.with_suffix(".py")
                notebook_to_script(notebook, script)
                print(f"Converted {notebook.name} to {script.name}")
                converted_count += 1

            if converted_count == 0:
                print("No example notebooks found to convert")
        else:
            # Convert all scripts to notebooks (original behavior)
            scripts = here.glob("*example.py")
            converted_count = 0
            for script in scripts:
                notebook = script.with_suffix(".ipynb")
                script_to_notebook(script, notebook)
                print(f"Converted {script.name} to {notebook.name}")
                converted_count += 1

            if converted_count == 0:
                print("No example scripts found to convert")
