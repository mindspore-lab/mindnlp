GENERATOR_FUNCTION_COMMENT = """
The function's code is: '{function_code}'.
The function named '{function_name}'. 
This function takes '{args_count}' parameters: {args_list}. 
The function returns a value of type '{return_type}'. 
These sections can be omitted in cases where the function’s name and signature are informative enough that it can be aptly described using a one-line docstring.
The docstring should contain the following sections:
Args: Describe each parameter including its type, purpose, and any restrictions.
Returns: Describe the type and purpose of the return value.
Raises: Document all exceptions that the function may raise.
For consistency, always use triple double quotes around docstrings. 
Please generate a docstring based on the above information and do not include method signatures or any other code.
"""

# 生成类注释
GENERATOR_CLASS_COMMENT = """
The class's code is: '{class_code}'.
Generate a detailed docstring for a Python class named '{class_name}'. 
This class inherits from {base_classes}.
Class docstrings should start with a one-line summary that describes what the class instance represents. 
For consistency, always use triple double quotes around docstrings.
Please generate a docstring based on the above information and do not include  signatures or any other code.
"""

# 生成类中函数的注释
GENERATOR_METHOD_COMMENT = """
The method's code is: '{method_code}'.
Generate a detailed docstring for a method named '{method_name}' in the class named '{class_name}'.
This method takes '{args_count}' parameters: {args_list}.
The method returns a value of type '{return_type}'.
The docstring should contain the following sections:
Args: Describe each parameter including its type, purpose, and any restrictions.
Returns: Describe the type and purpose of the return value.
Raises: Document all exceptions that the function may raise.
For consistency, always use triple double quotes around docstrings.
Please generate a docstring based on the above information and do not include method signatures or any other code.
"""

import os
from tqdm import tqdm
import argparse
import astroid
from mindnlp.transformers import pipeline

ARGS = None

# 建议使用至少强于gpt-3.5的模型
generator = pipeline(model="Qwen/Qwen1.5-1.8B-Chat")

def generate_docstring_batch(inputs):
    outputs = generator(inputs)
    res = []
    for output in outputs:
        res.append(output[0]['generated_text'])
    print(res)


def extract_and_generate_comments(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        node = astroid.parse(file.read())

    functions = []
    classes = []

    # 遍历 AST 节点，收集函数和类的信息
    for child in node.get_children():
        if isinstance(child, astroid.FunctionDef):
            func_info = extract_function_info(child)
            functions.append(func_info)
        elif isinstance(child, astroid.ClassDef):
            class_info = extract_class_info(child)
            classes.append(class_info)

    comments = generate_comments_skip_existing(functions, classes)
    insert_comments(file_path, comments)


def extract_function_info(child):
    """ 提取函数的相关信息。 """
    return {
        'startline': child.lineno,
        'name': child.name,
        'args': [arg.name for arg in child.args.args],
        'docstring': child.doc_node.value if child.doc_node else None,
        'returns': child.returns.as_string() if child.returns else None,
        'code': child.as_string(),

    }


def extract_class_info(child):
    """ 提取类的相关信息。 """
    class_info = {
        'startline': child.lineno,
        'name': child.name,
        'bases': [base.as_string() for base in child.bases],
        'docstring': child.doc_node.value if child.doc_node else None,
        'methods': [],

    }
    for method in child.get_children():
        if isinstance(method, astroid.FunctionDef):
            method_info = extract_function_info(method)
            class_info['methods'].append(method_info)
    return class_info


def generate_comments_skip_existing(functions, classes):
    inputs = []
    to_generate = []
    # 处理独立函数
    for function in functions:
        if not function['docstring']:
            comment_text = GENERATOR_FUNCTION_COMMENT.format(
                function_name=function['name'],
                args_count=len(function['args']),
                args_list=', '.join(function['args']),
                return_type=function['returns'],
                function_code=function['code'],
            )
            inputs.append(comment_text)
            to_generate.append(('function', function))  # 记录需要生成文档字符串的函数

    # 处理类及其方法
    for class_info in classes:
        if not class_info['docstring']:  # 检查类本身是否已有 docstring
            class_comment_text = GENERATOR_CLASS_COMMENT.format(
                class_name=class_info['name'],
                base_classes=', '.join(class_info['bases']),
                class_code='\n'.join(method['code'] for method in class_info['methods']),  # 汇总所有方法的代码为类代码部分
            )
            inputs.append(class_comment_text)
            to_generate.append(('class', class_info))

        # 处理类中的方法
        for method in class_info['methods']:
            if not method['docstring']:
                method_comment_text = GENERATOR_METHOD_COMMENT.format(
                    method_name=method['name'],
                    class_name=class_info['name'],
                    args_count=len(method['args']),
                    args_list=', '.join(method['args']),
                    return_type=method['returns'],
                    method_code=method['code'],
                )
                inputs.append(method_comment_text)
                to_generate.append(('method', method, class_info['name']))

    batch_docstrings = generate_docstring_batch(inputs)
    comments = []

    # 分配批量结果到函数和类方法
    for i, result in enumerate(batch_docstrings):
        entity_type, entity = to_generate[i][0], to_generate[i][1]
        if entity_type == 'function':
            comments.append((entity['startline'], entity['name'], result, 1))
        elif entity_type == 'class':
            comments.append((entity['startline'], entity['name'], result, 1))
        elif entity_type == 'method':
            class_name = to_generate[i][2]
            comments.append((entity['startline'], entity['name'], result, 2))

    comments.sort(key=lambda x: x[0], reverse=True)
    return comments


def find_def_end(lines, start_index):
    open_brackets = 0
    def_code = lines[start_index]
    for index, line in enumerate(lines[start_index:], start=start_index):
        open_brackets += line.count('(') - line.count(')')
        if open_brackets == 0 and ':' in line:
            return index
    return start_index


def insert_comments(file_path, comments):
    suffix = ARGS.suffix
    new_file_path = f"{file_path.rsplit('.', 1)[0]}{suffix}.py"  # 添加后缀并构造新文件名

    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    with open(new_file_path, 'w', encoding='utf-8') as new_file:
        new_file.writelines(lines)

    with open(new_file_path, 'r+', encoding='utf-8') as file:
        lines = file.readlines()

        comments.sort(key=lambda x: x[0], reverse=True)

        for startline, name, docstring, indent_level in comments:

            docstring = docstring.replace("```python", '')
            docstring = docstring.replace("```", '')

            import re
            match = re.search(r'(?:r|u|b|f)*?("""|\'\'\')[\s\S]*?\1', docstring).group()
            if match:
                docstring = match.group()
            else:
                print(f"No valid multiline string literal found in：{name}")
                continue

            insert_point = find_def_end(lines, startline - 1)
            indent = '    ' * indent_level
            indented_docstring = '\n'.join(indent + line for line in docstring.split('\n'))
            comment_str = f"\n{indented_docstring}\n"
            lines.insert(insert_point + 1, comment_str)

        file.seek(0)

        file.writelines(lines)

        file.truncate()

    print(f"Annotated file created: {new_file_path}")

def process_directory(directory_path, exclude=None, ):
    directory_path = os.path.abspath(directory_path)

    if os.path.isfile(directory_path):
        if directory_path.endswith('.py'):
            print(f"Processing file: {directory_path}")
            extract_and_generate_comments(directory_path)
    elif os.path.isdir(directory_path):
        python_files = []
        for root, dirs, files in os.walk(directory_path):
            python_files.extend(os.path.join(root, file) for file in files if file.endswith('.py'))

        for file_path in tqdm(python_files, desc="正在处理的文件"):
            if exclude and any(os.path.abspath(ex_path) == file_path for ex_path in exclude):
                continue
            print(f"Processing file: {file_path}")
            extract_and_generate_comments(file_path)
    else:
        print(f"Invalid path: {directory_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="自动为 Python 文件生成文档字符串。")
    parser.add_argument(
        'directory_path',
        type=str,
        default='./mindnlp',
        nargs='?',
        help='要处理的目录或文件的路径'
    )
    parser.add_argument(
        '--exclude_files',
        nargs='*',
        default=[],  # 默认排除文件或文件夹列表
        help='要排除的文件列表'
    )
    parser.add_argument(
        '--suffix',
        type=str,
        default='_annotated',
        help='输出文件的后缀，如果为空则替换原文件'
    )
    ARGS = parser.parse_args()  # 初始化全局变量

    print(f"目录路径: {ARGS.directory_path}")
    print(f"排除的文件: {ARGS.exclude_files}")
    print(f"文件后缀: {ARGS.suffix}")
    process_directory(ARGS.directory_path, exclude=ARGS.exclude_files)
