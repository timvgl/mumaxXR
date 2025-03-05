import re

# Global environment to hold assigned variables (e.g. operatorskspace, nx, etc.)
global_env = {}

# Allowed arithmetic functions (safe ones)
allowed_functions = {
    "int": int,
    "float": float,
    "abs": abs,
    "max": max,
    "min": min
}

# Set of custom operator names that our language uses (and should not be treated as arithmetic calls)
custom_ops = {
    "fft3d", "fft4d", "fft_t", "cropx", "expandx", "cropy", "expandy", "cropz", "expandz",
    "croplayer", "crop", "expand", "cropxoperator", "expandxoperator", "cropyoperator",
    "expandyoperator", "cropzoperator", "expandzoperator", "mergeoperators"
}

# Helper function: Generates a range string similar to mumax.
def range_str(a, b):
    # If a+1 equals b, return just a as string; otherwise, return "a-b".
    return str(a) if a + 1 == b else f"{a}-{b}"

# Tokenizer: Recognizes numbers (including scientific notation), identifiers (lowercase only),
# arithmetic operators, and punctuation (including dot, comma, parentheses, and equal sign).
def tokenize(s):
    tokens = re.findall(r'\d+(?:\.\d+)?(?:e-?\d+)?|[a-z_]\w*|[+\-*/(),.=]', s)
    return tokens

# Recursive descent parser for our language.
def parse_expression(s):
    tokens = tokenize(s)
    pos = 0

    # --- Arithmetic parser functions ---
    def parse_primary():
        nonlocal pos
        # If an expression is parenthesized, consume '(' expression ')'
        if pos < len(tokens) and tokens[pos] == '(':
            pos += 1
            node = parse_expr()
            if pos >= len(tokens) or tokens[pos] != ')':
                raise Exception("Expected closing parenthesis")
            pos += 1
            node = parse_postfix(node)
            return node
        # Otherwise, the token must be a number or an identifier.
        token = tokens[pos]
        pos += 1
        # Try to interpret as number.
        try:
            if '.' in token or 'e' in token:
                node = float(token)
            else:
                node = int(token)
        except ValueError:
            node = token  # identifier
        node = parse_postfix(node)
        return node

    def parse_factor():
        nonlocal pos
        # Handle unary minus.
        if pos < len(tokens) and tokens[pos] == '-':
            op = tokens[pos]
            pos += 1
            operand = parse_factor()
            return {"type": "unary", "op": op, "operand": operand}
        else:
            return parse_primary()

    def parse_term():
        nonlocal pos
        node = parse_factor()
        while pos < len(tokens) and tokens[pos] in ['*', '/']:
            op = tokens[pos]
            pos += 1
            right = parse_factor()
            node = {"type": "binary", "op": op, "left": node, "right": right}
        return node

    def parse_additive():
        nonlocal pos
        node = parse_term()
        while pos < len(tokens) and tokens[pos] in ['+', '-']:
            op = tokens[pos]
            pos += 1
            right = parse_term()
            node = {"type": "binary", "op": op, "left": node, "right": right}
        return node

    def parse_expr():
        return parse_additive()

    # --- Postfix processing (for function calls and member accesses) ---
    def parse_postfix(node):
        nonlocal pos
        # Process function calls: if the next token is '(' then it's a call.
        while pos < len(tokens):
            if tokens[pos] == '(':
                pos += 1  # Skip '('.
                args = []
                if pos < len(tokens) and tokens[pos] != ')':
                    args.append(parse_expr())
                    while pos < len(tokens) and tokens[pos] == ',':
                        pos += 1
                        args.append(parse_expr())
                if pos >= len(tokens) or tokens[pos] != ')':
                    raise Exception("Expected closing parenthesis in function call")
                pos += 1  # Skip ')'.
                # Build a call node.
                node = {"type": "call", "name": node, "args": args} if isinstance(node, str) else {"type": "call", "name": node["value"] if isinstance(node, dict) and node.get("type")=="identifier" else node, "args": args}
            elif tokens[pos] == '.':
                pos += 1  # Skip dot.
                if pos >= len(tokens):
                    raise Exception("Expected member name after dot")
                member = tokens[pos]
                pos += 1
                # Check if this is a member call.
                if pos < len(tokens) and tokens[pos] == '(':
                    pos += 1  # Skip '('.
                    args = []
                    if pos < len(tokens) and tokens[pos] != ')':
                        args.append(parse_expr())
                        while pos < len(tokens) and tokens[pos] == ',':
                            pos += 1
                            args.append(parse_expr())
                    if pos >= len(tokens) or tokens[pos] != ')':
                        raise Exception("Expected closing parenthesis in member call")
                    pos += 1  # Skip ')'.
                    node = {"type": "member_call", "object": node, "name": member, "args": args}
                else:
                    node = {"type": "member_access", "object": node, "name": member}
            else:
                break
        return node

    # --- Begin parsing ---
    node = parse_expr()
    if pos != len(tokens):
        raise Exception("Unexpected tokens remaining: " + " ".join(tokens[pos:]))
    return node

# Evaluator: Recursively processes the AST to build the final name or evaluate arithmetic.
def eval_ast(node):
    # If node is a number, return it.
    if isinstance(node, (int, float)):
        return node
    # If node is a string, check global environment or return as is.
    if isinstance(node, str):
        if node == "emptyoperator":
            return ""
        return global_env[node] if node in global_env else node

    # Binary operator node.
    if isinstance(node, dict) and node.get("type") == "binary":
        left = eval_ast(node["left"])
        right = eval_ast(node["right"])
        op = node["op"]
        if op == '+':
            return left + right
        elif op == '-':
            return left - right
        elif op == '*':
            return left * right
        elif op == '/':
            return left / right
        else:
            raise Exception(f"Unknown binary operator: {op}")

    # Unary operator node.
    if isinstance(node, dict) and node.get("type") == "unary":
        operand = eval_ast(node["operand"])
        op = node["op"]
        if op == '-':
            return -operand
        else:
            raise Exception(f"Unknown unary operator: {op}")

    # Handle our call nodes.
    if isinstance(node, dict) and node.get("type") == "call":
        op = node["name"]
        # If op is not a string (could be an arithmetic result) then just evaluate the call
        if not isinstance(op, str):
            op = eval_ast(op)
        # Check if this is one of our custom operators.
        if op in custom_ops:
            # Custom functions:
            if op == "fft3d":
                parent_name = eval_ast(node["args"][0])
                return parent_name + "_k_x_y_z"
            if op == "fft4d":
                parent_name = eval_ast(node["args"][0])
                op_suffix = global_env.get("operatorskspace", "")
                return parent_name + "_k_x_y_z" + op_suffix + "_f"
            if op == "fft_t":
                parent_name = eval_ast(node["args"][0])
                return parent_name + "_f"
            if op in ["cropx", "expandx"]:
                x1 = eval_ast(node["args"][1])
                x2 = eval_ast(node["args"][2])
                return eval_ast(node["args"][0]) + "_xrange" + range_str(x1, x2)
            if op in ["cropy", "expandy"]:
                y1 = eval_ast(node["args"][1])
                y2 = eval_ast(node["args"][2])
                return eval_ast(node["args"][0]) + "_yrange" + range_str(y1, y2)
            if op in ["cropz", "expandz"]:
                z1 = eval_ast(node["args"][1])
                z2 = eval_ast(node["args"][2])
                return eval_ast(node["args"][0]) + "_zrange" + range_str(z1, z2)
            if op == "croplayer":
                layer = eval_ast(node["args"][1])
                return eval_ast(node["args"][0]) + "_zrange" + range_str(layer, layer + 1)
            if op in ["crop", "expand"]:
                if len(node["args"]) < 7:
                    raise Exception(f"Not enough arguments for {op}")
                x1 = eval_ast(node["args"][1])
                x2 = eval_ast(node["args"][2])
                y1 = eval_ast(node["args"][3])
                y2 = eval_ast(node["args"][4])
                z1 = eval_ast(node["args"][5])
                z2 = eval_ast(node["args"][6])
                suffix = "_xrange" + range_str(x1, x2) + "_yrange" + range_str(y1, y2) + "_zrange" + range_str(z1, z2)
                return eval_ast(node["args"][0]) + suffix
            if op in ["cropxoperator", "expandxoperator"]:
                x1 = eval_ast(node["args"][0])
                x2 = eval_ast(node["args"][1])
                return "_xrange" + range_str(x1, x2)
            if op in ["cropyoperator", "expandyoperator"]:
                y1 = eval_ast(node["args"][0])
                y2 = eval_ast(node["args"][1])
                return "_yrange" + range_str(y1, y2)
            if op in ["cropzoperator", "expandzoperator"]:
                z1 = eval_ast(node["args"][0])
                z2 = eval_ast(node["args"][1])
                return "_zrange" + range_str(z1, z2)
            if op == "mergeoperators":
                evaluated_args = [eval_ast(arg) for arg in node["args"]]
                # If the only argument is emptyoperator (empty string), return empty string.
                if len(evaluated_args) == 1 and evaluated_args[0] == "":
                    return ""
                merged = ""
                for arg in evaluated_args:
                    merged += arg
                return merged
            raise Exception(f"Unknown custom function: {op}")
        else:
            # If not a custom op, assume it is an arithmetic function call.
            # Evaluate all arguments.
            evaluated_args = [eval_ast(arg) for arg in node["args"]]
            if op in allowed_functions:
                return allowed_functions[op](*evaluated_args)
            else:
                raise Exception(f"Unknown function: {op}")

    # Member call.
    if isinstance(node, dict) and node.get("type") == "member_call":
        obj_name = eval_ast(node["object"])
        member = node["name"]
        if member == "comp":
            if len(node["args"]) != 1:
                raise Exception("comp expects exactly one argument")
            comp_index = eval_ast(node["args"][0])
            mapping = {0: "_x", 1: "_y", 2: "_z"}
            return obj_name + mapping.get(comp_index, f"_comp{comp_index}")
        elif member in ["abs", "phi", "real", "imag"]:
            if len(node["args"]) != 0:
                raise Exception(f"Member function {member} expects no arguments")
            return obj_name + "_" + member
        elif member in ["saveabs", "savephi", "topolar"]:
            if len(node["args"]) != 0:
                raise Exception(f"Member function {member} expects no arguments")
            mapping = {"saveabs": "abs", "savephi": "phi", "topolar": "polar"}
            return obj_name + "_" + mapping[member]
        else:
            raise Exception(f"Unknown member function: {member}")

    # Member access.
    if isinstance(node, dict) and node.get("type") == "member_access":
        return eval_ast(node["object"]) + "_" + node["name"]

    raise Exception(f"Unknown node structure: {node}")

# Process a command string.
# If the command contains an assignment (=), the variable is stored in global_env.
def process_command(cmd):
    cmd = cmd.strip()
    if ":=" in cmd:
        varname, expr_str = cmd.split(":=", 1)
        varname = varname.strip().lower()
        expr_str = expr_str.strip()
        ast = parse_expression(expr_str)
        value = eval_ast(ast)
        global_env[varname] = value
        return f"{varname} defined as {value}"
    elif "=" in cmd:
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
