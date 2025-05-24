from __future__ import annotations
import argparse
import subprocess
import rply
import io
import os

class DebugPrint:
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({', '.join(f'{k}={v!r}' for k, v in self.__dict__.items())})"

TOKENS = {
    "FUNC": r"func",
    "EXTERN": r"extern",
    "DATA": r"data",
    "OP_EQ": r"eq",
    "OP_IF": r"if",
    "OP_RET": r"ret",
    "OP_SUB": r"sub",
    "OP_ADD": r"add",
    "OP_CALL": r"call",
    "OP_CAST": r"cast",
    "OP_EXT": r"ext",
    "FUNC_NAME": r"@[_a-zA-Z][_a-zA-Z0-9]*",
    "PARAM_NAME": r"\$[_a-zA-Z][_a-zA-Z0-9]*",
    "TEMP_NAME": r"%[0-9]+",
    "LOCAL_NAME": r"%[_a-zA-Z][_a-zA-Z0-9]*",
    "INT_TYPE": r"i[0-9]+",
    "UINT_TYPE": r"u[0-9]+",
    "VOID_TYPE": r"void",
    "INT": r"-?[0-9]+",
    "ARROW": r"->",
    "LPAREN": r"\(",
    "RPAREN": r"\)",
    "LBRACE": r"\{",
    "RBRACE": r"\}",
    "COMMA": r",",
    "COLON": r":",
    "EQ": r"=",
    "SEMICOLON": r";",
}

lg = rply.LexerGenerator()
for token, regex in TOKENS.items():
    lg.add(token, regex)

lg.ignore(r"\s+")
lg.ignore(r"//.*")

lexer = lg.build()

pg = rply.ParserGenerator(TOKENS.keys())

class Node(DebugPrint):
    ctx: Ctx

    def define_and_set_ctx(self, ctx: Ctx) -> None:
        raise NotImplementedError(self.__class__.__name__, "define_and_set_ctx")

    def check(self) -> None:
        raise NotImplementedError(self.__class__.__name__, "check")

class Top(Node):
    def to_x64(self, x64: X64) -> X64Top:
        raise NotImplementedError(self.__class__.__name__, "to_x64")

class X64Top(DebugPrint):
    def __init__(self, x64: X64) -> None:
        self.x64 = x64

    def write(self, stream: io.TextIOBase) -> None:
        raise NotImplementedError(self.__class__.__name__, "write")

@pg.production("tops : top tops")
@pg.production("tops : top")
@pg.production("tops :")
def _(p):
    if len(p) == 0:
        return []
    elif len(p) == 1:
        return [p[0]]
    else:
        return [p[0]] + p[1]

@pg.production("top : func_decl")
@pg.production("top : extern_decl")
@pg.production("top : data_decl")
def _(p):
    return p[0]

class ExternDecl(Top):
    def __init__(self, name: str):
        self.name = name

    def define_and_set_ctx(self, ctx: Ctx) -> None:
        self.ctx = ctx

        # TODO: Add types to externs to indicate whether theyre a function (use ptr type), or a variable (use the var type)
        ctx.define_global(self.name, IntType(64, False))

    def check(self) -> None:
        pass

    def to_x64(self, x64: X64) -> X64Top:
        return X64ExternDecl(x64, self.name)

class X64ExternDecl(X64Top):
    def __init__(self, x64: X64, name: str) -> None:
        super().__init__(x64)

        self.name = name

    def write(self, stream: io.TextIOBase) -> None:
        stream.write(f"extern {self.name}\n")

@pg.production("extern_decl : EXTERN FUNC_NAME SEMICOLON")
def _(p):
    return ExternDecl(p[1].getstr()[1:])

class DataDecl(Top):
    def __init__(self, name: str, values: list[Int]):
        self.name = name
        self.values = values

    def define_and_set_ctx(self, ctx: Ctx) -> None:
        self.ctx = ctx

        # TODO: Use pointer type
        self.ctx.define_global(self.name, IntType(64, False))

    def check(self) -> None:
        for value in self.values:
            value.check()

    def to_x64(self, x64: X64) -> X64Top:
        return X64DataDecl(x64, self.name, self.values)

class X64DataDecl(X64Top):
    def __init__(self, x64: X64, name: str, values: list[Int]) -> None:
        super().__init__(x64)

        self.name = name
        self.values = values

    def write(self, stream: io.TextIOBase) -> None:
        stream.write(f"section .data\n")
        stream.write(f"{self.name}:\n")
        for value in self.values:
            stream.write(f"    ")
            if value.bytes == 1:
                stream.write(f"db ")
            elif value.bytes == 2:
                stream.write(f"dw ")
            elif value.bytes == 4:
                stream.write(f"dd ")
            elif value.bytes == 8:
                stream.write(f"dq ")
            else:
                raise Exception(f"Unsupported size {value.bytes} for data declaration")
            stream.write(f"{value.value}\n")

@pg.production("data_decl : DATA FUNC_NAME EQ values SEMICOLON")
def _(p):
    return DataDecl(p[1].getstr()[1:], p[3])

@pg.production("values : int COMMA values")
@pg.production("values : int")
@pg.production("values :")
def _(p):
    if len(p) == 0:
        return []
    elif len(p) == 1:
        return [p[0]]
    else:
        return [p[0]] + p[2]

class FuncDecl(Top):
    def __init__(self, name: str, params: list[Param], ret_type: Type, body: Block):
        self.name = name
        self.params = params
        self.ret_type = ret_type
        self.body = body

    def define_and_set_ctx(self, ctx: Ctx) -> None:
        self.ctx = ctx

        # TODO: Introduce a pointer type thats the size of the target architectures pointer size. Use that here
        self.ctx.define_global(self.name, IntType(64, False))

    def check(self) -> None:
        child_ctx = self.ctx.create_child()

        for param in self.params:
            param.define_and_set_ctx(child_ctx)
        self.body.define_and_set_ctx(child_ctx)

        self.body.check()

    def to_x64(self, x64: X64) -> X64Top:
        if len(self.params) > len(X64.PARAM_REGS):
            raise NotImplementedError("Passing more than 6 parameters not implemented yet")
        params = [X64Param(reg.sized(param.type.bytes), param) for reg, param in zip(X64.PARAM_REGS, self.params)]

        ret_reg = None
        if not isinstance(self.ret_type, VoidType):
            ret_reg = X64.RET_REG.sized(self.ret_type.bytes)

        func = X64FuncDecl(x64, self.name, params, ret_reg)
        self.body.to_x64(func)
        return func

class X64Param(DebugPrint):
    def __init__(self, reg: X64Reg, info: Param):
        self.reg = reg
        self.info = info

class X64FuncDecl(X64Top):
    def __init__(self, x64: X64, name: str, params: list[X64Param], ret_reg: X64Reg | None) -> None:
        super().__init__(x64)

        self.name = name
        self.params = params
        self.ret_reg = ret_reg

        self.__used_stack = 0
        self.__used_labels = 0
        self.__temps = {}

        self.body: list[X64Instr] = []

        self.emit_comment(f"Moving parameters to stack")
        if not self.params:
            self.emit_comment(f"No parameters")

        self.param_values = {}
        for param in self.params:
            val = self.__alloc_param(param.info.name, param.info.type)
            self.emit_comment(f"- {param.info.name}")
            self.emit_mov(val, param.reg)

    def get_or_alloc_temp(self, name: str, type: Type) -> X64Value:
        if name in self.__temps:
            assert self.__temps[name][1] == type, f"Type mismatch for temp {name}: {self.__temps[name][1]} != {type}, should be prevented by type checking"
            return self.__temps[name][0]

        val = self.stack_alloc(type.bytes)
        self.__temps[name] = (val, type)
        return val

    def __alloc_param(self, name: str, type: Type) -> X64Value:
        if name in self.param_values:
            raise Exception(f"Parameter {name} already allocated")

        val = self.stack_alloc(type.bytes)
        self.param_values[name] = (val, type)
        return val

    def get_param(self, name: str, type: Type) -> X64Value:
        if name in self.param_values:
            assert self.param_values[name][1] == type, f"Type mismatch for param {name}: {self.param_values[name][1]} != {type}"
            return self.param_values[name][0]

        raise Exception(f"Parameter {name!r} not found")

    def write(self, stream: io.TextIOBase) -> None:
        stream.write(f"section .text\n")
        stream.write(f"global {self.name}\n")
        stream.write(f"{self.name}:\n")
        stream.write("    push rbp\n")
        stream.write("    mov rbp, rsp\n")
        stream.write(f"    sub rsp, {self.__used_stack}\n")
        for line in self.body:
            line.write(stream)
        stream.write(f".return:\n")
        stream.write(f"    mov rsp, rbp\n")
        stream.write(f"    pop rbp\n")
        stream.write(f"    ret\n")

    def temp_label(self) -> X64Label:
        label = f".L{self.__used_labels}"
        self.__used_labels += 1
        return X64Label(label)

    def stack_alloc(self, bytes: int) -> X64Stack:
        self.__used_stack += bytes
        return X64Stack(self.__used_stack, bytes)

    def emit_comment(self, comment: str) -> None:
        self.body.append(X64CommentInstr(comment))

    def emit_movsx(self, left: X64Value, right: X64Value) -> None:
        # TODO: Don't use a temp reg if left is already a register
        # Target always has to be a register
        temp = X64.R10.sized(left.bytes)
        self.body.append(X64MovsxInstr(temp, right))
        self.body.append(X64MovInstr(left, temp))

    def emit_movzx(self, left: X64Value, right: X64Value) -> None:
        # TODO: Don't use a temp reg if left is already a register
        # Target always has to be a register
        temp = X64.R10.sized(left.bytes)
        self.body.append(X64MovzxInstr(temp, right))
        self.body.append(X64MovInstr(left, temp))

    def emit_cmp(self, left: X64Value, right: X64Value) -> None:
        if left.bytes != right.bytes:
            raise Exception("Size mismatch in cmp")

        if isinstance(left, X64Memory) and isinstance(right, X64Memory):
            temp = X64.R10.sized(right.bytes)
            self.body.append(X64MovInstr(temp, right))
            right = temp

        self.body.append(X64CmpInstr(left, right))

    def emit_add(self, left: X64Value, right: X64Value) -> None:
        if left.bytes != right.bytes:
            raise Exception("Size mismatch in add")

        if isinstance(left, X64Memory) and isinstance(right, X64Memory):
            temp = X64.R10.sized(right.bytes)
            self.body.append(X64MovInstr(temp, right))
            right = temp

        self.body.append(X64AddInstr(left, right))

    def emit_sub(self, left: X64Value, right: X64Value) -> None:
        if left.bytes != right.bytes:
            raise Exception("Size mismatch in sub")

        if isinstance(left, X64Memory) and isinstance(right, X64Memory):
            temp = X64.R10.sized(right.bytes)
            self.body.append(X64MovInstr(temp, right))
            right = temp

        self.body.append(X64SubInstr(left, right))

    def emit_mov(self, left: X64Value, right: X64Value) -> None:
        if left.bytes != right.bytes:
            raise Exception("Size mismatch in mov")

        if isinstance(left, X64Memory) and isinstance(right, X64Memory):
            temp = X64.R10.sized(right.bytes)
            self.body.append(X64MovInstr(temp, right))
            right = temp

        self.body.append(X64MovInstr(left, right))

    def emit_call(self, func: X64Value) -> None:
        self.body.append(X64CallInstr(func))  # TODO: I'm pretty sure you cant call int literals or memory reads, so use a temp reg

    def emit_je(self, address: X64Value) -> None:
        self.body.append(X64JeInstr(address))

    def emit_jmp(self, address: X64Value) -> None:
        self.body.append(X64JmpInstr(address))

    def emit_label(self, label: X64Label) -> None:
        self.body.append(X64LabelInstr(label))

@pg.production("func_decl : FUNC FUNC_NAME LPAREN params RPAREN ARROW type block")
def _(p):
    return FuncDecl(p[1].getstr()[1:], p[3], p[6], p[7])

class Param(Node):
    def __init__(self, name: str, type: Type):
        self.name = name
        self.type = type

    def define_and_set_ctx(self, ctx: Ctx) -> None:
        self.ctx = ctx

        self.ctx.define_param(self.name, self.type)

@pg.production("params : param COMMA params")
@pg.production("params : param")
@pg.production("params :")
def _(p):
    if len(p) == 0:
        return []
    elif len(p) == 1:
        return [p[0]]
    else:
        return [p[0]] + p[2]

@pg.production("param : PARAM_NAME COLON type")
def _(p):
    return Param(p[0].getstr()[1:], p[2])

class Type:
    def __init__(self, bytes: int) -> None:
        self.bytes = bytes

    def __eq__(self, value: object) -> bool:
        if type(value) != type(self):
            return False
        return self.bytes == value.bytes

    def __repr__(self) -> str:
        raise NotImplementedError(self.__class__.__name__, "__repr__")

class VoidType(Type):
    def __init__(self):
        super().__init__(0)

@pg.production("type : VOID_TYPE")
def _(p):
    return VoidType()

class IntType(Type):
    def __init__(self, bits: int, signed: bool):
        assert bits % 8 == 0, "Int type with non-byte-aligned size"
        super().__init__(bits // 8)
        self.signed = signed

    def __repr__(self) -> str:
        return f"{'i' if self.signed else 'u'}{self.bytes * 8}"

@pg.production("type : INT_TYPE")
def _(p):
    return IntType(int(p[0].getstr()[1:]), True)

@pg.production("type : UINT_TYPE")
def _(p):
    return IntType(int(p[0].getstr()[1:]), False)

class Statement(Node):
    def to_x64(self, x64: X64FuncDecl) -> None:
        raise NotImplementedError(self.__class__.__name__, "to_x64")

class Block(Statement):
    def __init__(self, statements: list[Statement]):
        self.statements = statements

    def define_and_set_ctx(self, ctx: Ctx) -> None:
        self.ctx = ctx

        child_ctx = ctx.create_child()
        for statement in self.statements:
            statement.define_and_set_ctx(child_ctx)

    def check(self) -> None:
        for statement in self.statements:
            statement.check()

    def to_x64(self, x64: X64FuncDecl) -> None:
        for statement in self.statements:
            statement.to_x64(x64)

@pg.production("block : LBRACE statements RBRACE")
def _(p):
    return Block(p[1])

@pg.production("statements : statement statements")
@pg.production("statements : statement")
@pg.production("statements :")
def _(p):
    if len(p) == 0:
        return []
    elif len(p) == 1:
        return [p[0]]
    else:
        return [p[0]] + p[1]

class Expression(Node):
    def __init__(self, type: Type) -> None:
        self.type = type

    def to_x64(self, x64: X64FuncDecl) -> X64Value:
        raise NotImplementedError(self.__class__.__name__, "to_x64")

class If(Statement):
    def __init__(self, condition: Expression, block: Block):
        self.condition = condition
        self.block = block

    def define_and_set_ctx(self, ctx: Ctx) -> None:
        self.ctx = ctx

        self.condition.define_and_set_ctx(ctx)
        self.block.define_and_set_ctx(ctx)

    def check(self) -> None:
        self.condition.check()
        if not isinstance(self.condition.type, IntType):
            raise Exception(f"Condition must be an int type, got {self.condition.type}")
        if self.condition.type.bytes != 1:
            raise Exception(f"Condition must be 1 byte, got {self.condition.type.bytes} bytes")  # TODO: Maybe we should allow any size?

        self.block.check()

    def to_x64(self, x64: X64FuncDecl) -> None:
        x64.emit_comment("If statement")
        if_false = x64.temp_label()
        x64.emit_cmp(self.condition.to_x64(x64), X64Int(0, 1))
        x64.emit_je(if_false)
        x64.emit_comment("If true")
        self.block.to_x64(x64)
        x64.emit_comment("If false")
        x64.emit_label(if_false)

@pg.production("statement : OP_IF expression block")
def _(p):
    return If(p[1], p[2])

class Ref(Expression):
    def __init__(self, name: str, type: Type) -> None:
        super().__init__(type)
        self.name = name

class Assign(Statement):
    def __init__(self, ref: Ref, expr: Expression):
        self.ref = ref
        self.expr = expr

    def to_x64(self, x64: X64FuncDecl) -> None:
        x64.emit_comment("Assign statement")
        x64.emit_mov(self.ref.to_x64(x64), self.expr.to_x64(x64))

# @pg.production("statement : local_ref EQ op SEMICOLON")
# @pg.production("statement : temp_ref EQ op SEMICOLON")
# @pg.production("statement : local_ref EQ expression SEMICOLON")
# @pg.production("statement : temp_ref EQ expression SEMICOLON")
# def _(p):
#     return Assign(p[0], p[2])

@pg.production("ref : local_ref")
@pg.production("ref : temp_ref")
@pg.production("ref : param_ref")
@pg.production("ref : func_ref")
def _(p):
    return p[0]

class LocalRef(Ref):
    pass

@pg.production("local_ref : LOCAL_NAME COLON type")
def _(p):
    return LocalRef(p[0].getstr()[1:], p[2])

class TempRef(Ref):
    def define_and_set_ctx(self, ctx: Ctx) -> None:
        self.ctx = ctx

        self.ctx.ensure_temp_exists(self.name, self.type)

    def check(self) -> None:
        pass

    def to_x64(self, x64: X64FuncDecl) -> X64Value:
        return x64.get_or_alloc_temp(self.name, self.type)

@pg.production("temp_ref : TEMP_NAME COLON type")
def _(p):
    return TempRef(p[0].getstr()[1:], p[2])

class ParamRef(Ref):
    def define_and_set_ctx(self, ctx: Ctx) -> None:
        self.ctx = ctx

        self.ctx.ensure_param_exists(self.name, self.type)

    def check(self) -> None:
        pass

    def to_x64(self, x64: X64FuncDecl) -> X64Value:
        return x64.get_param(self.name, self.type)

@pg.production("param_ref : PARAM_NAME COLON type")
def _(p):
    return ParamRef(p[0].getstr()[1:], p[2])

class FuncRef(Ref):
    def __init__(self, name: str) -> None:
        super().__init__(name, IntType(64, False))

    def define_and_set_ctx(self, ctx: Ctx) -> None:
        self.ctx = ctx

        ctx.ensure_global_exists(self.name, IntType(64, False))  # TODO: Use pointer type

    def check(self) -> None:
        pass

    def to_x64(self, x64: X64FuncDecl) -> X64Value:
        return X64Label(self.name)

@pg.production("func_ref : FUNC_NAME")
def _(p):
    return FuncRef(p[0].getstr()[1:])

class Op(Statement):
    pass

class EqOp(Op):
    def __init__(self, dest: Ref, left: Expression, right: Expression):
        self.dest = dest
        self.left = left
        self.right = right

    def define_and_set_ctx(self, ctx: Ctx) -> None:
        self.ctx = ctx

        self.dest.define_and_set_ctx(ctx)
        self.left.define_and_set_ctx(ctx)
        self.right.define_and_set_ctx(ctx)

    def check(self) -> None:
        self.dest.check()
        self.left.check()
        self.right.check()

        if not isinstance(self.dest.type, IntType):
            raise Exception(f"Destination type must be int, got {self.dest.type}")
        if self.dest.type.bytes != 1:
            raise Exception(f"Destination type must be 1 byte, got {self.dest.type.bytes} bytes")

        if self.left.type != self.right.type:
            raise Exception(f"Type mismatch in equality operation, left: {self.left.type}, right: {self.right.type}")
        if not isinstance(self.left.type, IntType):
            raise NotImplementedError("Non-int equality not implemented yet")

    def to_x64(self, x64: X64FuncDecl) -> None:
        x64_dest = self.dest.to_x64(x64)
        x64_left = self.left.to_x64(x64)
        x64_right = self.right.to_x64(x64)

        # TODO: Use setcc instead of jcc + mov
        if_true = x64.temp_label()
        x64.emit_cmp(x64_left, x64_right)
        x64.emit_mov(x64_dest, X64Int(1, 1))
        x64.emit_je(if_true)
        x64.emit_mov(x64_dest, X64Int(0, 1))
        x64.emit_label(if_true)

@pg.production("op : OP_EQ ref COMMA expression COMMA expression")
def _(p):
    return EqOp(p[1], p[3], p[5])

class ExtOp(Op):
    def __init__(self, dest: Ref, expr: Expression):
        self.dest = dest
        self.expr = expr
    
    def define_and_set_ctx(self, ctx: Ctx) -> None:
        self.ctx = ctx

        self.dest.define_and_set_ctx(ctx)
        self.expr.define_and_set_ctx(ctx)

    def check(self) -> None:
        self.dest.check()
        self.expr.check()

        if not isinstance(self.dest.type, IntType):
            raise NotImplementedError("Non-int extension not implemented yet")
        if not isinstance(self.expr.type, IntType):
            raise NotImplementedError("Non-int extension not implemented yet")
        
        if self.dest.type.bytes < self.expr.type.bytes:
            raise Exception(f"Destination type {self.dest.type} must be larger than or equal to expression type {self.expr.type}")

    def to_x64(self, x64: X64FuncDecl) -> None:
        # TODO: Ensure destination has the same signedness as expr
        expr_val = self.expr.to_x64(x64)
        res = self.dest.to_x64(x64)
        if not isinstance(self.dest.type, IntType):
            raise Exception("Destination type must be int")

        if self.dest.type.signed:
            x64.emit_movsx(res, expr_val)
        else:
            x64.emit_movzx(res, expr_val)

@pg.production("op : OP_EXT ref COMMA expression")
def _(p):
    return ExtOp(p[1], p[3])

class CastOp(Op):
    def __init__(self, dest: Ref, expr: Expression):
        self.dest = dest
        self.expr = expr

    def to_x64(self, x64: X64FuncDecl) -> None:
        # return self.expr.to_x64(x64)
        return super().to_x64(x64)

@pg.production("op : OP_CAST ref COMMA expression")
def _(p):
    return CastOp(p[1], p[3])

class SubOp(Op):
    def __init__(self, dest: Ref, left: Expression, right: Expression):
        self.dest = dest
        self.left = left
        self.right = right

    def define_and_set_ctx(self, ctx: Ctx) -> None:
        self.ctx = ctx

        self.dest.define_and_set_ctx(ctx)
        self.left.define_and_set_ctx(ctx)
        self.right.define_and_set_ctx(ctx)

    def check(self) -> None:
        self.dest.check()
        self.left.check()
        self.right.check()

        if self.dest.type != self.left.type or self.dest.type != self.right.type:
            raise Exception(f"Type mismatch in subtraction operation, all operands have to be the same type")
        
        if not isinstance(self.dest.type, IntType):
            raise NotImplementedError("Non-int subtraction not implemented yet")

    def to_x64(self, x64: X64FuncDecl) -> None:
        # TODO: Type check operands

        if not type(self.dest.type) == IntType:
            raise NotImplementedError("Non-int subtraction not implemented yet")

        x64_dest = self.dest.to_x64(x64)
        x64_left = self.left.to_x64(x64)
        x64_right = self.right.to_x64(x64)

        x64.emit_mov(x64_dest, x64_left)
        x64.emit_sub(x64_dest, x64_right)

@pg.production("op : OP_SUB ref COMMA expression COMMA expression")
def _(p):
    return SubOp(p[1], p[3], p[5])

class CallOp(Op):
    def __init__(self, dest: Ref, func: Expression, args: list[Expression]):
        self.dest = dest
        self.func = func
        self.args = args

    def define_and_set_ctx(self, ctx: Ctx) -> None:
        self.ctx = ctx

        self.dest.define_and_set_ctx(ctx)
        self.func.define_and_set_ctx(ctx)

        for arg in self.args:
            arg.define_and_set_ctx(ctx)

    def check(self) -> None:
        self.dest.check()
        self.func.check()
        for arg in self.args:
            arg.check()

        if len(self.args) > 6:
            raise NotImplementedError("Passing more than 6 parameters not implemented yet")

    def to_x64(self, x64: X64FuncDecl) -> None:
        x64.emit_comment("Call")
        for i, arg in enumerate(self.args):
            x64_arg = arg.to_x64(x64)
            x64.emit_mov(X64.PARAM_REGS[i].sized(x64_arg.bytes), x64_arg)

        # TODO: Handle this better. It's for variadic functions where rax needs to be set to the number of floating point args
        x64.emit_mov(X64.RAX.sized(8), X64Int(0, 8))

        x64.emit_call(self.func.to_x64(x64))

        # TODO: Implement a void literal that lets you explicitly discard results of operations
        # if isinstance(self.dest.type, VoidType):
        #     return

        # TODO: Decide which reg to use based on ret type, if its a float, we shouldnt use rax
        x64.emit_mov(self.dest.to_x64(x64), X64.RET_REG.sized(self.dest.type.bytes))

@pg.production("op : OP_CALL ref COMMA expression COMMA args")
@pg.production("op : OP_CALL ref COMMA expression")
def _(p):
    if len(p) == 6:
        return CallOp(p[1], p[3], p[5])
    else:
        return CallOp(p[1], p[3], [])

@pg.production("args : expression COMMA args")
@pg.production("args : expression")
@pg.production("args :")
def _(p):
    if len(p) == 0:
        return []
    elif len(p) == 1:
        return [p[0]]
    else:
        return [p[0]] + p[2]

class AddOp(Op):
    def __init__(self, dest: Ref, left: Expression, right: Expression):
        self.dest = dest
        self.left = left
        self.right = right

    def define_and_set_ctx(self, ctx: Ctx) -> None:
        self.ctx = ctx

        self.dest.define_and_set_ctx(ctx)
        self.left.define_and_set_ctx(ctx)
        self.right.define_and_set_ctx(ctx)

    def check(self) -> None:
        self.dest.check()
        self.left.check()
        self.right.check()

        if self.dest.type != self.left.type or self.dest.type != self.right.type:
            raise Exception(f"Type mismatch in add operation, all operands have to be the same type")
        
        if not isinstance(self.dest.type, IntType):
            raise NotImplementedError("Non-int addition not implemented yet")

    def to_x64(self, x64: X64FuncDecl) -> None:
        x64_dest = self.dest.to_x64(x64)
        x64_left = self.left.to_x64(x64)
        x64_right = self.right.to_x64(x64)

        x64.emit_mov(x64_dest, x64_left)
        x64.emit_add(x64_dest, x64_right)

@pg.production("op : OP_ADD ref COMMA expression COMMA expression")
def _(p):
    return AddOp(p[1], p[3], p[5])

class RetOp(Statement):
    def __init__(self, expr: Expression | None):
        self.expr = expr

    def define_and_set_ctx(self, ctx: Ctx) -> None:
        self.ctx = ctx

        if self.expr is not None:
            self.expr.define_and_set_ctx(ctx)

    def check(self) -> None:
        # TODO: Check that the return type matches the function's return type instead of doing that in the x64 pass
        if self.expr is not None:
            self.expr.check()

    def to_x64(self, x64: X64FuncDecl) -> None:
        x64.emit_comment("Return statement")

        has_expr = self.expr is not None
        has_ret_reg = x64.ret_reg is not None
        if not has_expr and has_ret_reg:
            raise Exception("Cannot return without an expression")
        if has_expr and not has_ret_reg:
            raise Exception("Cannot return an expression without a return register")

        if self.expr is not None:
            assert x64.ret_reg is not None, "Null checked earlier"
            x64.emit_mov(x64.ret_reg, self.expr.to_x64(x64))
        x64.emit_jmp(X64Label(".return"))

@pg.production("op : OP_RET expression")
@pg.production("op : OP_RET")
def _(p):
    if len(p) == 2:
        return RetOp(p[1])
    else:
        return RetOp(None)

@pg.production("statement : op SEMICOLON")
def _(p):
    return p[0]

@pg.production("expression : ref")
@pg.production("expression : int")
def _(p):
    return p[0]

class Int(Expression):
    def __init__(self, value: int, bits: int, signed: bool):
        super().__init__(IntType(bits, signed))

        assert bits % 8 == 0, "Int type with non-byte-aligned size"

        self.value = value
        self.bytes = bits // 8
        self.signed = signed

    def define_and_set_ctx(self, ctx: Ctx) -> None:
        self.ctx = ctx

    def check(self) -> None:
        try:
            self.value.to_bytes(self.bytes, "little", signed=self.signed)
        except OverflowError:
            raise Exception(f"Value {self.value} does not fit in {self.bytes} byte(s)")

    def to_x64(self, x64: X64FuncDecl) -> X64Value:
        return X64Int(self.value, self.bytes)

@pg.production("int : INT COLON INT_TYPE")
def _(p):
    return Int(int(p[0].getstr()), int(p[2].getstr()[1:]), signed=True)

@pg.production("int : INT COLON UINT_TYPE")
def _(p):
    return Int(int(p[0].getstr()), int(p[2].getstr()[1:]), signed=False)

parser = pg.build()

class Ctx:
    class Storage:
        TEMP = "Temp"
        PARAM = "Param"
        LOCAL = "Local"
        GLOBAL = "Global"

    def __init__(self, parent: Ctx | None = None) -> None:
        self.__parent = parent
        self.__symbols: dict[tuple[str, str], Type] = {}

    def create_child(self) -> Ctx:
        return Ctx(self)

    def __define_symbol(self, storage: str, name: str, type: Type) -> None:
        if (storage, name) in self.__symbols:
            raise Exception(f"{storage} {name!r} already defined")
        self.__symbols[(storage, name)] = type

    def define_global(self, name: str, type: Type) -> None:
        assert self.__parent is None, "Global symbols should only be defined in the root context"

        self.__define_symbol(Ctx.Storage.GLOBAL, name, type)

    def define_param(self, name: str, type: Type) -> None:
        self.__define_symbol(Ctx.Storage.PARAM, name, type)

    def define_temp(self, name: str, type: Type) -> None:
        self.__define_symbol(Ctx.Storage.TEMP, name, type)

    def define_local(self, name: str, type: Type) -> None:
        self.__define_symbol(Ctx.Storage.LOCAL, name, type)

    def __try_get_symbol(self, storage: str, name: str) -> Type | None:
        if (storage, name) in self.__symbols:
            return self.__symbols[(storage, name)]

        if self.__parent is not None:
            return self.__parent.__try_get_symbol(storage, name)

        return None

    def __ensure_symbol_exists(self, storage: str, name: str, type: Type) -> None:
        existing = self.__try_get_symbol(storage, name)
        if existing is not None:
            if existing != type:
                raise Exception(f"{storage} {name!r} already defined with type {existing!r}, cannot redefine as {type!r}")
            return

        self.__define_symbol(storage, name, type)

    def ensure_temp_exists(self, name: str, type: Type) -> None:
        self.__ensure_symbol_exists(Ctx.Storage.TEMP, name, type)

    def ensure_param_exists(self, name: str, type: Type) -> None:
        self.__ensure_symbol_exists(Ctx.Storage.PARAM, name, type)

    def ensure_global_exists(self, name: str, type: Type) -> None:
        existing = self.__try_get_symbol(Ctx.Storage.GLOBAL, name)
        if existing is None:
            raise Exception(f"Global {name!r} not defined")

        if existing != type:
            raise Exception(f"Global {name!r} already defined with type {existing}, cannot redefine as {type}")

def x64_ptr_size(bytes: int) -> str:
    if bytes == 8:
        return "qword"
    elif bytes == 4:
        return "dword"
    elif bytes == 2:
        return "word"
    elif bytes == 1:
        return "byte"
    else:
        raise NotImplementedError(f"Unsupported size {bytes} for x64")

class X64Value(DebugPrint):
    def __init__(self, bytes: int) -> None:
        self.bytes = bytes

    def write(self, stream: io.TextIOBase) -> None:
        raise NotImplementedError(self.__class__.__name__, "write")

class X64Int(X64Value):
    def __init__(self, value: int, bytes: int):
        super().__init__(bytes)
        self.value = value

    def write(self, stream: io.TextIOBase) -> None:
        stream.write(f"{x64_ptr_size(self.bytes)} {self.value}")

class X64Label(X64Value):
    def __init__(self, name: str):
        super().__init__(8)
        self.name = name

    def write(self, stream: io.TextIOBase) -> None:
        stream.write(f"{self.name}")

class X64Reg(X64Value):
    def __init__(self, name: str, bytes: int):
        super().__init__(bytes)
        self.name = name

    def write(self, stream: io.TextIOBase) -> None:
        stream.write(self.name)

class X64RegSet:
    def __init__(self, qword: str, dword: str, word: str, high_byte: str | None, low_byte: str):
        self.qword = X64Reg(qword, 8)
        self.dword = X64Reg(dword, 4)
        self.word = X64Reg(word, 2)
        self.high_byte = X64Reg(high_byte, 1) if high_byte is not None else None
        self.low_byte = X64Reg(low_byte, 1)

    def sized(self, bytes: int) -> X64Reg:
        if bytes == 8:
            return self.qword
        elif bytes == 4:
            return self.dword
        elif bytes == 2:
            return self.word
        elif bytes == 1:
            return self.low_byte
        else:
            raise NotImplementedError(f"Unsupported size {bytes} for {self.qword}")

class X64Memory(X64Value):
    pass

class X64Stack(X64Memory):
    def __init__(self, offset: int, bytes: int):
        self.offset = offset
        self.bytes = bytes

    def write(self, stream: io.TextIOBase) -> None:
        stream.write(f"{x64_ptr_size(self.bytes)} [rbp - {self.offset}]")

class X64Instr(DebugPrint):
    def write(self, stream: io.TextIOBase) -> None:
        raise NotImplementedError(self.__class__.__name__, "write")

class X64CommentInstr(X64Instr):
    def __init__(self, comment: str):
        self.comment = comment

    def write(self, stream: io.TextIOBase) -> None:
        stream.write(f"    ; {self.comment}\n")

class X64AddInstr(X64Instr):
    def __init__(self, left: X64Value, right: X64Value):
        self.left = left
        self.right = right

    def write(self, stream: io.TextIOBase) -> None:
        stream.write(f"    add ")
        self.left.write(stream)
        stream.write(", ")
        self.right.write(stream)
        stream.write("\n")

class X64SubInstr(X64Instr):
    def __init__(self, left: X64Value, right: X64Value):
        self.left = left
        self.right = right

    def write(self, stream: io.TextIOBase) -> None:
        stream.write(f"    sub ")
        self.left.write(stream)
        stream.write(", ")
        self.right.write(stream)
        stream.write("\n")

class X64MovInstr(X64Instr):
    def __init__(self, left: X64Value, right: X64Value):
        self.left = left
        self.right = right

    def write(self, stream: io.TextIOBase) -> None:
        stream.write(f"    mov ")
        self.left.write(stream)
        stream.write(", ")
        self.right.write(stream)
        stream.write("\n")

class X64MovsxInstr(X64Instr):
    def __init__(self, left: X64Value, right: X64Value):
        self.left = left
        self.right = right

    def write(self, stream: io.TextIOBase) -> None:
        stream.write(f"    movsx ")
        self.left.write(stream)
        stream.write(", ")
        self.right.write(stream)
        stream.write("\n")

class X64MovzxInstr(X64Instr):
    def __init__(self, left: X64Value, right: X64Value):
        self.left = left
        self.right = right

    def write(self, stream: io.TextIOBase) -> None:
        stream.write(f"    movzx ")
        self.left.write(stream)
        stream.write(", ")
        self.right.write(stream)
        stream.write("\n")

class X64CmpInstr(X64Instr):
    def __init__(self, left: X64Value, right: X64Value):
        self.left = left
        self.right = right

    def write(self, stream: io.TextIOBase) -> None:
        stream.write(f"    cmp ")
        self.left.write(stream)
        stream.write(", ")
        self.right.write(stream)
        stream.write("\n")

class X64CallInstr(X64Instr):
    def __init__(self, func: X64Value):
        self.func = func

    def write(self, stream: io.TextIOBase) -> None:
        stream.write(f"    call ")
        self.func.write(stream)
        stream.write("\n")

class X64JeInstr(X64Instr):
    def __init__(self, label: X64Value):
        self.label = label

    def write(self, stream: io.TextIOBase) -> None:
        stream.write(f"    je ")
        self.label.write(stream)
        stream.write("\n")

class X64JmpInstr(X64Instr):
    def __init__(self, label: X64Value):
        self.label = label

    def write(self, stream: io.TextIOBase) -> None:
        stream.write(f"    jmp ")
        self.label.write(stream)
        stream.write("\n")

class X64LabelInstr(X64Instr):
    def __init__(self, label: X64Label):
        self.label = label

    def write(self, stream: io.TextIOBase) -> None:
        stream.write(f"{self.label.name}:\n")

class X64(DebugPrint):
    RAX = X64RegSet("rax", "eax", "ax", "ah", "al")
    RDI = X64RegSet("rdi", "edi", "di", None, "dil")
    RSI = X64RegSet("rsi", "esi", "si", None, "sil")
    RDX = X64RegSet("rdx", "edx", "dx", "dh", "dl")
    RCX = X64RegSet("rcx", "ecx", "cx", "ch", "cl")
    R8 = X64RegSet("r8", "r8d", "r8w", None, "r8b")
    R9 = X64RegSet("r9", "r9d", "r9w", None, "r9b")
    R10 = X64RegSet("r10", "r10d", "r10w", None, "r10b")  # Used as temporary
    R11 = X64RegSet("r11", "r11d", "r11w", None, "r11b")  # Used as temporary

    PARAM_REGS = [RDI, RSI, RDX, RCX, R8, R9]
    RET_REG = RAX

class Target:
    def __init__(self, name: str) -> None:
        self.name = name

    def run(self, tops: list[Top], output_file: str) -> bool:
        raise NotImplementedError(self.__class__.__name__, "run")

class X64Target(Target):
    def __init__(self) -> None:
        super().__init__("x64")

    def run(self, tops: list[Top], output_file: str) -> bool:
        x64 = X64()

        s_file = get_with_ext(output_file, "s")
        o_file = get_with_ext(output_file, "o")

        with open(s_file, "w") as f:
            for top in tops:
                assert isinstance(top, Top)
                top.to_x64(x64).write(f)
                f.write("\n")
            f.write("section .note.GNU-stack\n")

        if subprocess.run(["nasm", "-f", "elf64", "-o", o_file, s_file]).returncode != 0:
            return False
        if subprocess.run(["cc", "-no-pie", "-o", output_file, o_file]).returncode != 0:
            return False

        return True

TARGETS = [
    X64Target(),
]

def get_with_ext(file: str, ext: str | None) -> str:
    base, _ = os.path.splitext(file)
    if ext is None:
        return base
    return f"{base}.{ext}"

arg_parser = argparse.ArgumentParser(description="TAC compiler")
arg_parser.add_argument("-o", "--output", type=str, help="Output executable file")
arg_parser.add_argument("-t", "--target", choices=[t.name for t in TARGETS], default=TARGETS[0].name, help="Target architecture")
arg_parser.add_argument("input", type=str, help="Input TAC file")
args = arg_parser.parse_args()

try:
    with open(args.input, "r") as f:
        source = f.read()
except FileNotFoundError:
    print(f"Error: Input file {args.input} not found")
    exit(1)

tops = parser.parse(lexer.lex(source))
assert isinstance(tops, list)

ctx = Ctx()
for top in tops:
    assert isinstance(top, Top)
    top.define_and_set_ctx(ctx)
for top in tops:
    assert isinstance(top, Top)
    top.check()

for target in TARGETS:
    if target.name == args.target:
        break
else:
    print(f"Error: Target {args.target} not found")
    exit(1)

if not target.run(tops, args.output or get_with_ext(args.input, None)):
    print(f"Error: Compilation failed")
    exit(1)
