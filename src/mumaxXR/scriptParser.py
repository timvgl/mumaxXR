import re

# Global environment to hold assigned variables (e.g. operatorskspace)
global_env = {}

def range_str(a, b):
    # If a+1 equals b, return just a as string; otherwise, return "a-b".
    return str(a) if a + 1 == b else f"{a}-{b}"

# Tokenizer: Recognizes numbers (including scientific notation), identifiers (lowercase only),
# and punctuation (including dot, comma, parentheses, and equal sign).
def tokenize(s):
    tokens = re.findall(r'\d+(?:\.\d+)?(?:e-?\d+)?|[a-z_]\w*|[(),.=]', s)
    return tokens

# The parser converts the input expression into a simple AST.
# It distinguishes between:
#  - Function calls (type 'call', e.g. cropx(...), fft3d(...), fft4d(...))
#  - Member function calls (type 'member_call', e.g. m.comp(1) or fft4d(m,10e-12).saveabs())
#  - Identifiers and numbers.
def parse_expression(s):
    tokens = tokenize(s)
    pos = 0

    # Function to handle postfix expressions (member accesses or calls)
    def parse_postfix(expr):
        nonlocal pos
        while pos < len(tokens) and tokens[pos] == '.':
            pos += 1  # Skip the dot.
            if pos >= len(tokens):
                raise Exception("Expected member name after dot")
            member = tokens[pos]
            pos += 1
            # If a member call is detected (with parentheses).
            if pos < len(tokens) and tokens[pos] == '(':
                pos += 1  # Skip '('.
                args = []
                if tokens[pos] != ')':
                    args.append(parse_expr())
                    while pos < len(tokens) and tokens[pos] == ',':
                        pos += 1  # Skip comma.
                        args.append(parse_expr())
                if pos >= len(tokens) or tokens[pos] != ')':
                    raise Exception("Expected closing parenthesis in member call")
                pos += 1  # Skip ')'.
                expr = {'type': 'member_call', 'object': expr, 'name': member, 'args': args}
            else:
                # Plain member access without arguments.
                expr = {'type': 'member_access', 'object': expr, 'name': member}
        return expr

    # Main parser for an expression.
    def parse_expr():
        nonlocal pos
        token = tokens[pos]
        pos += 1
        # Check if the next token indicates a function call.
        if pos < len(tokens) and tokens[pos] == '(':
            pos += 1  # Skip '('.
            args = []
            args.append(parse_expr())
            while pos < len(tokens) and tokens[pos] == ',':
                pos += 1  # Skip comma.
                # Try to parse numbers (including floats) or sub-expressions.
                if re.fullmatch(r'\d+(?:\.\d+)?(?:e-?\d+)?', tokens[pos]):
                    if '.' in tokens[pos] or 'e' in tokens[pos]:
                        args.append(float(tokens[pos]))
                    else:
                        args.append(int(tokens[pos]))
                    pos += 1
                else:
                    args.append(parse_expr())
            if tokens[pos] != ')':
                raise Exception("Expected closing parenthesis")
            pos += 1  # Skip ')'.
            expr = {'type': 'call', 'name': token, 'args': args}
        else:
            # Otherwise, token is either a number or an identifier.
            try:
                if '.' in token or 'e' in token:
                    expr = float(token)
                else:
                    expr = int(token)
            except ValueError:
                expr = token
        return parse_postfix(expr)

    ast = parse_expr()
    if pos != len(tokens):
        raise Exception("Unexpected tokens remaining: " + " ".join(tokens[pos:]))
    return ast

# Evaluator: Recursively processes the AST to build the final name.
def eval_ast(node):
    # Base case: if the node is a number, return it; if it's a string, check for special cases.
    if isinstance(node, (int, float)):
        return node
    if isinstance(node, str):
        # Special treatment for emptyoperator: always return an empty string.
        if node == "emptyoperator":
            return ""
        return global_env[node] if node in global_env else node

    if node['type'] == 'call':
        op = node['name']
        # Special handling for fft3d.
        if op == "fft3d":
            parent_name = eval_ast(node['args'][0])
            return parent_name + "_k_x_y_z"
        # Special handling for fft4d.
        if op == "fft4d":
            parent_name = eval_ast(node['args'][0])
            # Check if an operator has been defined for FFT4D (only operatorskspace applies)
            op_suffix = global_env.get("operatorskspace", "")
            return parent_name + "_k_x_y_z" + op_suffix + "_f"
        if op == "fft_t":
            parent_name = eval_ast(node['args'][0])
            # Check if an operator has been defined for FFT4D (only operatorskspace applies)
            op_suffix = global_env.get("operatorskspace", "")
            return parent_name + "_f"
        # Operator functions for FFT4D – these return a suffix string.
        if op == "cropxoperator" or op == "expandxoperator":
            x1 = node['args'][0]
            x2 = node['args'][1]
            return "_xrange" + range_str(x1, x2)
        if op == "cropyoperator" or op == "expandyoperator":
            y1 = node['args'][0]
            y2 = node['args'][1]
            return "_yrange" + range_str(y1, y2)
        if op == "cropzoperator" or op == "expandzoperator":
            z1 = node['args'][0]
            z2 = node['args'][1]
            return "_zrange" + range_str(z1, z2)
        if op == "mergeoperators":
            # Evaluate all arguments.
            evaluated_args = [eval_ast(arg) for arg in node['args']]
            # If the only argument is emptyoperator (i.e. empty string), return empty string.
            if len(evaluated_args) == 1 and evaluated_args[0] == "":
                return ""
            merged = ""
            for arg in evaluated_args:
                merged += arg
            return merged
        # Standard crop/expand functions (apply to a quantity).
        parent_name = eval_ast(node['args'][0])
        suffix = ""
        if op in ["cropx", "expandx"]:
            x1 = node['args'][1]
            x2 = node['args'][2]
            suffix = "_xrange" + range_str(x1, x2)
        elif op in ["cropy", "expandy"]:
            y1 = node['args'][1]
            y2 = node['args'][2]
            suffix = "_yrange" + range_str(y1, y2)
        elif op in ["cropz", "expandz"]:
            z1 = node['args'][1]
            z2 = node['args'][2]
            suffix = "_zrange" + range_str(z1, z2)
        elif op == "croplayer":
            layer = node['args'][1]
            suffix = "_zrange" + range_str(layer, layer + 1)
        elif op in ["crop", "expand"]:
            if len(node['args']) < 7:
                raise Exception(f"Not enough arguments for {op}")
            x1, x2, y1, y2, z1, z2 = node['args'][1:7]
            suffix = (
                "_xrange" + range_str(x1, x2) +
                "_yrange" + range_str(y1, y2) +
                "_zrange" + range_str(z1, z2)
            )
        else:
            raise Exception(f"Unknown function: {op}")
        return parent_name + suffix

    elif node['type'] == 'member_call':
        obj_name = eval_ast(node['object'])
        member = node['name']
        if member == "comp":
            if len(node['args']) != 1:
                raise Exception("comp expects exactly one argument")
            comp_index = eval_ast(node['args'][0])
            mapping = {0: "_x", 1: "_y", 2: "_z"}
            suffix = mapping.get(comp_index, f"_comp{comp_index}")
            return obj_name + suffix
        elif member in ["abs", "phi", "real", "imag"]:
            if len(node['args']) != 0:
                raise Exception(f"Member function {member} expects no arguments")
            return obj_name + "_" + member
        elif member in ["saveabs", "savephi", "topolar"]:
            if len(node['args']) != 0:
                raise Exception(f"Member function {member} expects no arguments")
            mapping = {"saveabs": "abs", "savephi": "phi", "topolar": "polar"}
            return obj_name + "_" + mapping[member]
        else:
            raise Exception(f"Unknown member function: {member}")

    elif node['type'] == 'member_access':
        obj_name = eval_ast(node['object'])
        member = node['name']
        return obj_name + "_" + member

    else:
        raise Exception(f"Unknown node type: {node['type']}")

# Process a command string.
# If the command contains an assignment (=), the variable is stored in global_env.
def process_command(cmd):
    cmd = cmd.strip()
    if "=" in cmd:
        # Split at the first '='.
        varname, expr_str = cmd.split("=", 1)
        varname = varname.strip().lower()
        expr_str = expr_str.strip()
        ast = parse_expression(expr_str)
        value = eval_ast(ast)
        global_env[varname] = value
        return f"{varname} defined as {value}"
    else:
        ast = parse_expression(cmd)
        return eval_ast(ast)

def get_globalenv():
    return global_env
