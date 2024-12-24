import ast
import argparse
import mindspore.mint

def extract_function_names(file_path):
    """Extract top-level function names from a Python file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        tree = ast.parse(file.read())

    function_names = [
        node.name for node in tree.body if isinstance(node, ast.FunctionDef)
    ]
    return function_names

def update_function_body(lines, function_name):
    """Update the function body to modify use_pyboost() to use_pyboost() and has_{function_name}."""
    updated_lines = []
    for line in lines:
        if "use_pyboost()" in line and 'and' not in line:
            line = line.replace("use_pyboost()", f"use_pyboost() and has_{function_name}")
        updated_lines.append(line)
    return updated_lines

def generate_global_variables(file_path):
    """Generate global variables, insert them before top-level functions, and update function bodies."""
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    tree = ast.parse("".join(lines))
    function_positions = {
        node.name: node.lineno - 1 for node in tree.body if isinstance(node, ast.FunctionDef)
    }

    existing_globals = {
        line.strip(): True for line in lines if line.strip().startswith("has_") and "= hasattr(" in line
    }

    new_lines = lines[:]
    offset = 0
    for func_name, line_no in sorted(function_positions.items(), key=lambda x: x[1]):
        # Test if the function exists in mindspore.mint
        if hasattr(mindspore.mint, func_name):
            # Insert global variable if not already present
            variable_declaration = f"has_{func_name} = hasattr(mindspore.mint, '{func_name}')"
            if variable_declaration not in existing_globals:
                new_lines.insert(line_no + offset, variable_declaration + "\n")
                offset += 1

            # Update function body
            func_start = line_no + offset
            func_end = func_start
            for i in range(func_start, len(new_lines)):
                if new_lines[i].strip() == "":
                    break
                func_end = i

            if variable_declaration not in existing_globals:
                updated_body = update_function_body(new_lines[func_start:func_end + 1], func_name)
                new_lines[func_start:func_end + 1] = updated_body

    with open(file_path, 'w', encoding='utf-8') as file:
        file.writelines(new_lines)

    print(f"Global variables have been inserted and function bodies updated in {file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Insert global variables and update function bodies in a Python file.")
    parser.add_argument("file", help="Path to the Python file to process.")
    args = parser.parse_args()

    generate_global_variables(args.file)
