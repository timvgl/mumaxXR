import re
import numpy as np

# Global environment to hold assigned variables (e.g. operatorskspace, nx, bx, etc.)
global_env = {}

# Allowed arithmetic functions (safe ones)
allowed_functions = {
    "int": int,
    "float": float,
    "abs": abs,
    "max": max,
    "min": min,
    "iceil": lambda x: int(np.ceil(x)),
    "ifloor": lambda x: int(np.floor(x)),
    "ceil": np.ceil,
    "floor": np.floor
}

# Set of custom operator names that our language uses (and should not be treated as arithmetic calls)
custom_ops = {
    "fft3d", "fft4d", "fft_t", "cropx", "expandx", "cropy", "expandy", "cropz", "expandz",
    "croplayer", "crop", "cropoperator", "expand", "expandoperator", "cropxoperator", "expandxoperator", "cropyoperator",
    "expandyoperator", "cropzoperator", "expandzoperator", "mergeoperators", "cropk","cropkx","cropky","cropkz","cropkxy",
    "cropkoperator","cropkxoperator","cropkyoperator","cropkzoperator","cropkxyoperator"
}

constants = {
    "pi":    np.pi
}

class OperatorSpec:
    """Hold the type (e.g. 'cropkxy') and its float parameters."""
    def __init__(self, op: str, params: tuple):
        self.op = op
        self.params = params

mesh_sizes = {}

def iceil(x): return int(np.ceil(x))
def ifloor(x): return int(np.floor(x))

def init_mesh_for(name):
    # whenever a base quantity like "m" is first seen, we assume a full mesh:
    # it must have been defined by Tx,dx etc → we already put nx,ny,nz in global_env
    if "nx" in global_env and "ny" in global_env and "nz" in global_env:
        mesh_sizes[name] = [global_env["nx"], global_env["ny"], global_env["nz"]]

# Call this right after you see an assignment to Tx,Ty,Tz,dx,dy,dz:
#    init_mesh_for("m")    # or whatever your base quantity is called

# --- 2) Whenever you crop or expand, update mesh_sizes[newName]: -----------
def update_mesh_after_crop(parent_name, new_name, x1,x2,y1,y2,z1,z2):
    mesh_sizes[new_name] = [ x2-x1, y2-y1, z2-z1 ]

def update_mesh_after_expand(parent_name, new_name, x1,x2,y1,y2,z1,z2):
    mesh_sizes[new_name] = [ x2-x1, y2-y1, z2-z1 ]

# Helper function: Generates a range string similar to mumax.
def range_str(a, b):
    return str(a) if a + 1 == b else f"{a}-{b}"

# Tokenizer: Recognizes numbers (including scientific notation), identifiers (lowercase only),
# arithmetic operators, and punctuation (including dot, comma, parentheses, equal sign, and colon).
def tokenize(s):
    # The regex now will match numbers, identifiers, and the operators.
    tokens = re.findall(r'\d+(?:\.\d+)?(?:e-?\d+)?|[a-z_]\w*|[:=+\-*/(),.]', s)
    return tokens

# Recursive descent parser that handles both arithmetic and our custom function/member calls.
def parse_expression(s):
    tokens = tokenize(s)
    pos = 0

    # --- Arithmetic parser functions ---
    def parse_primary():
        nonlocal pos
        if pos < len(tokens) and tokens[pos] == '(':
            pos += 1
            node = parse_expr()
            if pos >= len(tokens) or tokens[pos] != ')':
                raise Exception("Expected closing parenthesis")
            pos += 1
            node = parse_postfix(node)
            return node
        token = tokens[pos]
        pos += 1
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
                # If the node is an identifier string, treat it as a function call.
                if isinstance(node, str):
                    node = {"type": "call", "name": node, "args": args}
                else:
                    # If node is already a node (e.g., from arithmetic), use its evaluated value as the function name.
                    node = {"type": "call", "name": node, "args": args}
            elif tokens[pos] == '.':
                pos += 1  # Skip dot.
                if pos >= len(tokens):
                    raise Exception("Expected member name after dot")
                member = tokens[pos]
                pos += 1
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

    node = parse_expr()
    if pos != len(tokens):
        raise Exception("Unexpected tokens remaining: " + " ".join(tokens[pos:]))
    return node

# Evaluator: Recursively processes the AST to build the final name or evaluate arithmetic.
def eval_ast(node):
    if isinstance(node, (int, float)):
        return node
    if isinstance(node, str):
        if node == "emptyoperator":
            return ""
        if node in constants:
            return constants[node]
        return global_env[node] if node in global_env else node

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
        elif op == '%':
            return left % right
        else:
            raise Exception(f"Unknown binary operator: {op}")

    if isinstance(node, dict) and node.get("type") == "unary":
        operand = eval_ast(node["operand"])
        op = node["op"]
        if op == '-':
            return -operand
        else:
            raise Exception(f"Unknown unary operator: {op}")

    if isinstance(node, dict) and node.get("type") == "call":
        op = node["name"]
        if not isinstance(op, str):
            op = eval_ast(op)
        if op in custom_ops:
            if op in ["cropk","cropkx","cropky","cropkz","cropkxy"]:
                parent = eval_ast(node["args"][0])

                nx, ny, nz = mesh_sizes[parent]
                if global_env.get("negativekx", "true") == "true":
                    nx_real = nx - 2
                else:
                    nx_real = nx - 1
                dx, dy, dz = global_env["dx"], global_env["dy"], global_env["dz"]
                # helper to compute ceil/floor indices
                symmetricX = global_env.get("negativekx", "true") == "true"
                # now dispatch
                if op == "cropk":
                    kx0, kx1, ky0, ky1, kz0, kz1 = [eval_ast(a) for a in node["args"][1:7]]
                    # mumax uses startIndexX = symmetricX? nx/2 : (NegativeKX? nx/2 : 0)
                    startX = float(nx/2)
                    x1 = iceil(startX + kx0*float(nx_real)*dx)
                    x2 = None
                    if kx0 == kx1:
                        x2 = x1 + 1
                    else:
                        x2 = ifloor(startX + kx1*float(nx_real)*dx)
                    y1 = iceil(float(ny)/2 + ky0*float(ny)*dy)
                    y2 = None
                    if ky0 == ky1:
                        y2 = y1 + 1
                    else:
                        y2 = ifloor(float(ny)/2 + ky1*float(ny)*dy)
                    z1 = iceil(float(nz)/2 + kz0*float(nz)*dz)
                    z2 = None
                    if kz0 == kz1:
                        z2 = z1 + 1
                    else:
                        z2 = ifloor(float(nz)/2 + kz1*float(nz)*dz)
                    name = f"{parent}_xrange{range_str(x1, x2)}yrange{range_str(y1, y2)}zrange{range_str(z1, z2)}"
                    if not parent in mesh_sizes:
                        init_mesh_for(parent)
                    update_mesh_after_crop(parent, name, x1,x2,y1,y2,z1,z2)
                    return name
    
                if op == "cropkx":
                    kx0, kx1 = [eval_ast(a) for a in node["args"][1:3]]
                    startX = float(nx/2)
                    x1 = iceil(startX + kx0*float(nx_real)*dx)
                    x2 = None
                    if kx0 == kx1:
                        x2 = x1 + 1
                    else:
                        x2 = ifloor(startX + kx1*float(nx_real)*dx)
                    name = f"{parent}_xrange{range_str(x1, x2)}"
                    if not parent in mesh_sizes:
                        init_mesh_for(parent)
                    update_mesh_after_crop(parent, name, x1,x2, 0,ny, 0,nz)
                    return name
    
                if op == "cropky":
                    ky0, ky1 = [eval_ast(a) for a in node["args"][1:3]]
                    y1 = iceil(float(ny)/2 + ky0*float(ny)*dy)
                    y2 = None
                    if ky0 == ky1:
                        y2 = y1 + 1
                    else:
                        y2 = ifloor(float(ny)/2 + ky1*float(ny)*dy)
                    name = f"{parent}_yrange{range_str(y1, y2)}"
                    if not parent in mesh_sizes:
                        init_mesh_for(parent)
                    update_mesh_after_crop(parent, name, 0, nx, y1,y2, 0,nz)
                    return name
    
                if op == "cropkz":
                    kz0, kz1 = [eval_ast(a) for a in node["args"][1:3]]
                    z1 = iceil(float(nz)/2 + kz0*float(nz)*dz)
                    z2 = None
                    if kz0 == kz1:
                        z2 = z1 + 1
                    else:
                        z2 = ifloor(float(nz)/2 + kz1*float(nz)*dz)
                    name = f"{parent}_zrange{range_str(z1, z2)}"
                    if not parent in mesh_sizes:
                        init_mesh_for(parent)
                    update_mesh_after_crop(parent, name, 0,nx, 0,ny, z1,z2)
                    return name
    
                if op == "cropkxy":
                    kx0, kx1, ky0, ky1 = [eval_ast(a) for a in node["args"][1:5]]
                    startX = float(nx)/2
                    x1 = iceil(startX + kx0*float(nx_real)*dx)
                    x2 = None
                    if kx0 == kx1:
                        x2 = x1 + 1
                    else:
                        x2 = ifloor(startX + kx1*float(nx_real)*dx)
                    y1 = iceil(float(ny)/2 + ky0*float(ny)*dy)
                    y2 = None
                    if ky0 == ky1:
                        y2 = y1 + 1
                    else:
                        y2 = ifloor(float(ny)/2 + ky1*float(ny)*dy)
                    if not parent in mesh_sizes:
                        init_mesh_for(parent)
                    name = f"{parent}_xrange{range_str(x1, x2)}yrange{range_str(y1, y2)}"
                    update_mesh_after_crop(parent, name, x1,x2, y1,y2, 0,nz)
                    return name
            if op in ["cropkoperator","cropkxoperator","cropkyoperator","cropkzoperator","cropkxyoperator"]:
                args = [eval_ast(a) for a in node["args"]]
                return OperatorSpec(op.replace("operator", ""), tuple(args))
                
            if op == "fft3d":
                parent_name = eval_ast(node["args"][0])
                if not parent_name in mesh_sizes:
                    init_mesh_for(parent_name)
                new_name = parent_name + "_k_x_y_z"
                parent_mesh = mesh_sizes[parent_name]
                nx_real, ny, nz = parent_mesh
                if global_env.get("negativekx", "true") == "true":
                    nx = 2 * (nx_real // 2 + 1)
                else:
                    nx = nx_real + 1
                update_mesh_after_expand(parent_name, new_name, 0, nx, 0, ny, 0, nz)
                return new_name
            if op == "fft4d":
                parent = eval_ast(node["args"][0])
                if parent not in mesh_sizes:
                    init_mesh_for(parent)
                parent_mesh = mesh_sizes[parent]
                nx_real, ny, nz = parent_mesh
                if global_env.get("negativekx", "true") == "true":
                    nx = 2 * (nx_real // 2 + 1)
                else:
                    nx = nx_real + 1
                dx, dy, dz = global_env["dx"], global_env["dy"], global_env["dz"]
            
                final_suffix = []
                for item in global_env.get("operatorskspace", []):
                    if isinstance(item, OperatorSpec):
                        # compute exactly as before, based on spec.op and spec.params
                        spec = item
                        if spec.op=="cropk":
                            kx0, kx1, ky0, ky1, kz0, kz1 = spec.params
                            startX = 0
                            if global_env.get("negativekx", "true") == "true":
                                startX = float(nx)/2
                            x1 = iceil(startX + kx0*float(nx_real)*dx)
                            x2 = None
                            if kx0 == kx1:
                                x2 = x1 + 1
                            else:
                                x2 = ifloor(startX + kx1*float(nx_real)*dx)
                            y1 = iceil(float(ny)/2 + ky0*float(ny)*dy)
                            y2 = None
                            if ky0 == ky1:
                                y2 = y1 + 1
                            else:
                                y2 = ifloor(float(ny)/2 + ky1*float(ny)*dy)
                            z1 = iceil(float(nz)/2 + kz0*float(nz)*dz)
                            z2 = None
                            if kz0 == kz1:
                                z2 = z1 + 1
                            else:
                                z2 = ifloor(float(nz)/2 + kz1*float(nz)*dz)
                            final_suffix.append(f"_xrange{range_str(x1, x2)}yrange{range_str(y1, y2)}")
                        if spec.op=="cropkx":
                            kx0, kx1 = spec.params
                            startX = 0
                            if global_env.get("negativekx", "true") == "true":
                                startX = float(nx)/2
                            x1 = iceil(startX + kx0*float(nx_real)*dx)
                            x2 = None
                            if kx0 == kx1:
                                x2 = x1 + 1
                            else:
                                x2 = ifloor(startX + kx1*float(nx_real)*dx)
                            final_suffix.append(f"_xrange{range_str(x1, x2)}")
                        if spec.op=="cropky":
                            ky0, ky1 = spec.params
                            y1 = iceil(float(ny)/2 + ky0*float(ny)*dy)
                            y2 = None
                            if ky0 == ky1:
                                y2 = y1 + 1
                            else:
                                y2 = ifloor(float(ny)/2 + ky1*float(ny)*dy)
                            final_suffix.append(f"_yrange{range_str(y1, y2)}")
                        if spec.op=="cropkz":
                            kz0, kz1 = spec.params
                            z1 = iceil(float(nz)/2 + kz0*float(nz)*dz)
                            z2 = None
                            if kz0 == kz1:
                                z2 = z1 + 1
                            else:
                                z2 = ifloor(float(nz)/2 + kz1*float(nz)*dz)
                            final_suffix.append(f"_zrange{range_str(z1, z2)}")
                        if spec.op=="cropkxy":
                            kx0, kx1, ky0, ky1 = spec.params
                            startX = 0
                            if global_env.get("negativekx", "true") == "true":
                                startX = float(nx)/2
                            x1 = iceil(startX + kx0*float(nx_real)*dx)
                            x2 = None
                            if kx0 == kx1:
                                x2 = x1 + 1
                            else:
                                x2 = ifloor(startX + kx1*float(nx_real)*dx)
                            y1 = iceil(float(ny)/2 + ky0*float(ny)*dy)
                            y2 = None
                            if ky0 == ky1:
                                y2 = y1 + 1
                            else:
                                y2 = ifloor(float(ny)/2 + ky1*float(ny)*dy)
                            final_suffix.append(f"_xrange{range_str(x1, x2)}yrange{range_str(y1, y2)}")
                    else:
                        final_suffix.append(item)
            
                return parent + "_k_x_y_z" + "".join(final_suffix) + "_f"
            if op == "fft_t":
                parent_name = eval_ast(node["args"][0])
                return parent_name + "_f"
            if op in ["cropx", "expandx"]:
                parent_name = eval_ast(node["args"][0])
                x1 = eval_ast(node["args"][1])
                x2 = eval_ast(node["args"][2])
                new_name = parent_name + "_xrange" + range_str(x1, x2)
                if not parent_name in mesh_sizes:
                    init_mesh_for(parent_name)
                update_mesh_after_crop(parent_name, new_name, x1, x2, 0, mesh_sizes[parent_name][1], 0, mesh_sizes[parent_name][2])
                return new_name
            if op in ["cropy", "expandy"]:
                parent_name = eval_ast(node["args"][0])
                y1 = eval_ast(node["args"][1])
                y2 = eval_ast(node["args"][2])
                new_name = parent_name + "_yrange" + range_str(y1, y2)
                if not parent_name in mesh_sizes:
                    init_mesh_for(parent_name)
                update_mesh_after_crop(parent_name, new_name, 0, mesh_sizes[parent_name][0], y1, y2, 0, mesh_sizes[parent_name][2])
                return new_name
            if op in ["cropz", "expandz"]:
                parent_name = eval_ast(node["args"][0])
                z1 = eval_ast(node["args"][1])
                z2 = eval_ast(node["args"][2])
                new_name = parent_name + "_zrange" + range_str(z1, z2)
                if not parent_name in mesh_sizes:
                    init_mesh_for(parent_name)
                update_mesh_after_crop(parent_name, new_name, 0, mesh_sizes[parent_name][0], 0, mesh_sizes[parent_name][1], z1, z2)
                return new_name
            if op == "croplayer":
                layer = eval_ast(node["args"][1])
                return eval_ast(node["args"][0]) + "_zrange" + range_str(layer, layer + 1)
            if op in ["crop", "expand"]:
                if len(node["args"]) < 7:
                    raise Exception(f"Not enough arguments for {op}")
                parent_name = eval_ast(node["args"][0])
                x1 = eval_ast(node["args"][1])
                x2 = eval_ast(node["args"][2])
                y1 = eval_ast(node["args"][3])
                y2 = eval_ast(node["args"][4])
                z1 = eval_ast(node["args"][5])
                z2 = eval_ast(node["args"][6])
                new_name = parent_name + "_xrange" + range_str(x1, x2) + "yrange" + range_str(y1, y2) + "zrange" + range_str(z1, z2)
                if not parent_name in mesh_sizes:
                    init_mesh_for(parent_name)
                update_mesh_after_crop(parent_name, new_name, x1, x2, y1, y2, z1, z2)
                return new_name
            if op in ["cropoperator", "expandoperator"]:
                if len(node["args"]) < 6:
                    raise Exception(f"Not enough arguments for {op}")
                x1 = eval_ast(node["args"][0])
                x2 = eval_ast(node["args"][1])
                y1 = eval_ast(node["args"][2])
                y2 = eval_ast(node["args"][3])
                z1 = eval_ast(node["args"][4])
                z2 = eval_ast(node["args"][5])
                if "nz" in global_env and global_env["nz"] > 1 or "tz" in global_env and "dz" in global_env and int(global_env["tz"]/global_env["dz"]) > 1:
                    suffix = "_xrange" + range_str(x1, x2) + "yrange" + range_str(y1, y2) + "zrange" + range_str(z1, z2)
                else:
                    suffix = "_xrange" + range_str(x1, x2) + "yrange" + range_str(y1, y2)
                return suffix
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
                specs = []
                for a in node["args"]:
                    v = eval_ast(a)
                    if isinstance(v, OperatorSpec):
                        specs.append(v)
                    elif isinstance(v, str):
                        # ignore empty, or plain suffix strings if you still use those
                        specs.append(v)
                # store the list for FFT4D:
                global_env["operatorskspace"] = specs
                return specs  # mergeoperators itself yields no direct suffix here
            raise Exception(f"Unknown custom function: {op}")
        else:
            evaluated_args = [eval_ast(arg) for arg in node["args"]]
            if op in allowed_functions:
                return allowed_functions[op](*evaluated_args)
            else:
                raise Exception(f"Unknown function: {op}")

    if isinstance(node, dict) and node.get("type") == "member_call":
        obj_name = eval_ast(node["object"])
        member = node["name"]
        if member == "comp":
            if len(node["args"]) != 1:
                raise Exception("comp expects exactly one argument")
            comp_index = eval_ast(node["args"][0])
            mapping = {0: "_x", 1: "_y", 2: "_z"}
            new_name = obj_name + mapping[comp_index]
            if not obj_name in mesh_sizes:
                init_mesh_for(obj_name)
            mesh_sizes[new_name] = mesh_sizes[obj_name]
            return new_name
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

    if isinstance(node, dict) and node.get("type") == "member_access":
        return eval_ast(node["object"]) + "_" + node["name"]

    raise Exception(f"Unknown node structure: {node}")

# Process a command string.
# If the command contains an assignment using either "=" or ":=", the variable is stored in global_env.
def process_command(cmd):
    cmd = cmd.strip()
    # Check for ":=" assignment first.
    if ":=" in cmd:
        varname, expr_str = cmd.split(":=", 1)
        varname = varname.strip().lower()
        expr_str = expr_str.strip()
        ast = parse_expression(expr_str)
        value = eval_ast(ast)
        global_env[varname] = value
        if varname in ["tx", "dx"] and "tx" in global_env and "dx" in global_env:
            global_env["nx"] = int(global_env["tx"] / global_env["dx"])
        if varname in ["ty", "dy"] and "ty" in global_env and "dy" in global_env:
            global_env["ny"] = int(global_env["ty"] / global_env["dy"])
        if varname in ["tz", "dz"] and "tz" in global_env and "dz" in global_env:
            global_env["nz"] = int(global_env["tz"] / global_env["dz"])
        return f"{varname} defined as {value}"
    elif "=" in cmd:
        # Otherwise, use "=" assignment.
        varname, expr_str = cmd.split("=", 1)
        varname = varname.strip().lower()
        expr_str = expr_str.strip()
        ast = parse_expression(expr_str)
        value = eval_ast(ast)
        global_env[varname] = value
        if varname in ["tx", "dx"] and "tx" in global_env and "dx" in global_env:
            global_env["nx"] = int(global_env["tx"] / global_env["dx"])
        if varname in ["ty", "dy"] and "ty" in global_env and "dy" in global_env:
            global_env["ny"] = int(global_env["ty"] / global_env["dy"])
        if varname in ["tz", "dz"] and "tz" in global_env and "dz" in global_env:
            global_env["nz"] = int(global_env["tz"] / global_env["dz"])
        if varname in ["nx", "dx"] and "nx" in global_env and "dx" in global_env:
            global_env["tx"] = float(global_env["nx"]) * global_env["dx"]
        if varname in ["ny", "dy"] and "ny" in global_env and "dy" in global_env:
            global_env["ty"] = float(global_env["ny"]) * global_env["dy"]
        if varname in ["nz", "dz"] and "nz" in global_env and "dz" in global_env:
            global_env["tz"] = float(global_env["nz"]) * global_env["dz"]
        if varname in ["tx", "nx"] and "tx" in global_env and "nx" in global_env:
            global_env["dx"] = int(global_env["tx"] / global_env["nx"])
        if varname in ["ty", "ny"] and "ty" in global_env and "ny" in global_env:
            global_env["dy"] = int(global_env["ty"] / global_env["ny"])
        if varname in ["tz", "nz"] and "tz" in global_env and "nz" in global_env:
            global_env["dz"] = int(global_env["tz"] / global_env["nz"])
        return f"{varname} defined as {value}"
    else:
        ast = parse_expression(cmd)
        return eval_ast(ast)
