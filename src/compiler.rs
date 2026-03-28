use crate::ast::{self, Literal, Node, Operator, Type, UnaryOperator};
use std::collections::HashMap;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Clone, Debug, PartialEq, Eq)]
enum LlvmType {
    I8,
    I16,
    I32,
    I64,
    F32,
    F64,
    Char16,
    I1,
    PtrI8,
    Void,
}

impl LlvmType {
    fn ir(&self) -> &'static str {
        match self {
            LlvmType::I8 => "i8",
            LlvmType::I16 => "i16",
            LlvmType::I32 => "i32",
            LlvmType::I64 => "i64",
            LlvmType::F32 => "float",
            LlvmType::F64 => "double",
            LlvmType::Char16 => "i16",
            LlvmType::I1 => "i1",
            LlvmType::PtrI8 => "ptr",
            LlvmType::Void => "void",
        }
    }
}

#[derive(Clone, Debug)]
struct FunctionSignature {
    return_type: LlvmType,
    parameters: Vec<LlvmType>,
}

#[derive(Clone, Debug)]
struct LocalVar {
    ptr: String,
    llvm_type: LlvmType,
}

#[derive(Clone, Debug)]
struct ExprValue {
    llvm_type: LlvmType,
    value: String,
}

#[derive(Clone, Debug)]
struct GlobalString {
    name: String,
    bytes: Vec<u8>,
}

impl GlobalString {
    fn ir_decl(&self) -> String {
        format!(
            "@{} = private unnamed_addr constant [{} x i8] c\"{}\", align 1",
            self.name,
            self.bytes.len(),
            escape_llvm_bytes(&self.bytes)
        )
    }
}

fn escape_llvm_bytes(bytes: &[u8]) -> String {
    let mut out = String::new();
    for &byte in bytes {
        match byte {
            b' '..=b'~' if byte != b'\\' && byte != b'"' => out.push(byte as char),
            _ => {
                let _ = write!(out, "\\{:02X}", byte);
            }
        }
    }
    out
}

fn llvm_type(sk_type: &Type) -> Result<LlvmType, String> {
    match sk_type {
        Type::Byte => Ok(LlvmType::I8),
        Type::Short => Ok(LlvmType::I16),
        Type::Int => Ok(LlvmType::I32),
        Type::Long => Ok(LlvmType::I64),
        Type::Float => Ok(LlvmType::F32),
        Type::Double => Ok(LlvmType::F64),
        Type::Boolean => Ok(LlvmType::I1),
        Type::String => Ok(LlvmType::PtrI8),
        Type::Char => Ok(LlvmType::Char16),
        Type::Void => Ok(LlvmType::Void),
        other => Err(format!(
            "LLVM backend does not support type `{}` yet",
            ast::type_to_string(other)
        )),
    }
}

fn is_integer_llvm_type(llvm_type: &LlvmType) -> bool {
    matches!(
        llvm_type,
        LlvmType::I8 | LlvmType::I16 | LlvmType::I32 | LlvmType::I64 | LlvmType::Char16
    )
}

fn is_numeric_llvm_type(llvm_type: &LlvmType) -> bool {
    is_integer_llvm_type(llvm_type) || matches!(llvm_type, LlvmType::F32 | LlvmType::F64)
}

fn promoted_numeric_llvm_type(left: &LlvmType, right: &LlvmType) -> Option<LlvmType> {
    match (left, right) {
        (LlvmType::F64, _) | (_, LlvmType::F64) => Some(LlvmType::F64),
        (LlvmType::F32, _) | (_, LlvmType::F32) => Some(LlvmType::F32),
        (LlvmType::I64, _) | (_, LlvmType::I64) => Some(LlvmType::I64),
        (LlvmType::I8, LlvmType::I8)
        | (LlvmType::I8, LlvmType::I16)
        | (LlvmType::I16, LlvmType::I8)
        | (LlvmType::I16, LlvmType::I16)
        | (LlvmType::I8, LlvmType::I32)
        | (LlvmType::I32, LlvmType::I8)
        | (LlvmType::I16, LlvmType::I32)
        | (LlvmType::I32, LlvmType::I16)
        | (LlvmType::I32, LlvmType::I32) => Some(LlvmType::I32),
        _ => None,
    }
}

struct FunctionCompiler<'a> {
    function_name: &'a str,
    return_type: LlvmType,
    signatures: &'a HashMap<String, FunctionSignature>,
    globals: &'a mut Vec<GlobalString>,
    scopes: Vec<HashMap<String, LocalVar>>,
    lines: Vec<String>,
    temp_counter: usize,
    label_counter: usize,
    terminated: bool,
}

impl<'a> FunctionCompiler<'a> {
    fn new(
        function_name: &'a str,
        return_type: LlvmType,
        signatures: &'a HashMap<String, FunctionSignature>,
        globals: &'a mut Vec<GlobalString>,
    ) -> Self {
        Self {
            function_name,
            return_type,
            signatures,
            globals,
            scopes: vec![HashMap::new()],
            lines: Vec::new(),
            temp_counter: 0,
            label_counter: 0,
            terminated: false,
        }
    }

    fn compile(
        mut self,
        parameters: &[(String, Type)],
        body: &[Node],
    ) -> Result<Vec<String>, String> {
        for (index, (name, sk_type)) in parameters.iter().enumerate() {
            let llvm_type = llvm_type(sk_type)?;
            let arg_name = format!("%arg{}", index);
            let ptr = self.emit_alloca(llvm_type.clone(), name);
            self.emit_line(format!(
                "store {} {}, ptr {}, align {}",
                llvm_type.ir(),
                arg_name,
                ptr,
                Self::align_of(&llvm_type)
            ));
            self.declare_local(name.clone(), ptr, llvm_type);
        }

        self.compile_statements(body)?;

        if !self.terminated {
            match self.return_type {
                LlvmType::Void => self.emit_line("ret void".to_string()),
                _ => {
                    return Err(format!(
                        "function `{}` can reach the end without returning a value in LLVM backend",
                        self.function_name
                    ));
                }
            }
        }

        Ok(self.lines)
    }

    fn compile_statements(&mut self, statements: &[Node]) -> Result<(), String> {
        for statement in statements {
            if self.terminated {
                break;
            }
            self.compile_statement(statement)?;
        }
        Ok(())
    }

    fn compile_statement(&mut self, node: &Node) -> Result<(), String> {
        match node {
            Node::VariableDeclaration {
                var_type,
                name,
                value,
                ..
            } => {
                let llvm_type = llvm_type(var_type)?;
                let ptr = self.emit_alloca(llvm_type.clone(), name);
                let init = match value {
                    Some(value) => self.compile_expr(value)?,
                    None => self.default_value(&llvm_type),
                };
                let init = self.coerce_expr(init, &llvm_type, "variable declaration")?;
                self.emit_store(&ptr, &init);
                self.declare_local(name.clone(), ptr, llvm_type);
                Ok(())
            }
            Node::Assignment { var, value, .. } => {
                let expr = self.compile_expr(value)?;
                let local = self.resolve_local_from_access(var)?;
                let expr = self.coerce_expr(expr, &local.llvm_type, "assignment")?;
                self.emit_store(&local.ptr, &expr);
                Ok(())
            }
            Node::Block { statements } => {
                self.push_scope();
                self.compile_statements(statements)?;
                self.pop_scope();
                Ok(())
            }
            Node::If {
                condition,
                body,
                else_if_blocks,
                else_block,
            } => self.compile_if(condition, body, else_if_blocks, else_block.as_deref()),
            Node::For {
                init,
                condition,
                update,
                body,
            } => self.compile_for(init.as_deref(), condition.as_deref(), update.as_deref(), body),
            Node::Return(value) => {
                match value {
                    Some(value) => {
                        let expr = self.compile_expr(value)?;
                        let return_type = self.return_type.clone();
                        let expr = self.coerce_expr(expr, &return_type, "return")?;
                        self.emit_line(format!("ret {} {}", expr.llvm_type.ir(), expr.value));
                    }
                    None => {
                        if self.return_type != LlvmType::Void {
                            return Err(format!(
                                "function `{}` must return `{}`",
                                self.function_name,
                                self.return_type.ir()
                            ));
                        }
                        self.emit_line("ret void".to_string());
                    }
                }
                self.terminated = true;
                Ok(())
            }
            Node::Print(value) => self.compile_print(value),
            Node::FunctionCall { .. } => {
                let _ = self.compile_expr(node)?;
                Ok(())
            }
            Node::EOI => Ok(()),
            unsupported => Err(format!(
                "LLVM backend does not support statement `{:?}` yet",
                unsupported
            )),
        }
    }

    fn compile_if(
        &mut self,
        condition: &Node,
        body: &[Node],
        else_if_blocks: &[Node],
        else_block: Option<&[Node]>,
    ) -> Result<(), String> {
        let after_label = self.next_label("if_end");
        let else_entry = self.next_label("if_else");
        let then_label = self.next_label("if_then");
        let cond = self.compile_expr(condition)?;
        if cond.llvm_type != LlvmType::I1 {
            return Err("if condition must be boolean in LLVM backend".to_string());
        }
        self.emit_line(format!(
            "br i1 {}, label %{}, label %{}",
            cond.value, then_label, else_entry
        ));

        self.emit_label(&then_label);
        self.push_scope();
        self.compile_statements(body)?;
        self.pop_scope();
        if !self.terminated {
            self.emit_line(format!("br label %{}", after_label));
        }
        let then_terminated = self.terminated;
        self.terminated = false;

        self.emit_label(&else_entry);
        for else_if in else_if_blocks {
            self.compile_statement(else_if)?;
        }
        if let Some(else_block) = else_block {
            self.push_scope();
            self.compile_statements(else_block)?;
            self.pop_scope();
        }
        let else_terminated = self.terminated;
        if !else_terminated {
            self.emit_line(format!("br label %{}", after_label));
        }

        if then_terminated && else_terminated {
            self.terminated = true;
        } else {
            self.emit_label(&after_label);
            self.terminated = false;
        }
        Ok(())
    }

    fn compile_for(
        &mut self,
        init: Option<&Node>,
        condition: Option<&Node>,
        update: Option<&Node>,
        body: &[Node],
    ) -> Result<(), String> {
        self.push_scope();
        if let Some(init) = init {
            self.compile_statement(init)?;
        }

        let cond_label = self.next_label("for_cond");
        let body_label = self.next_label("for_body");
        let update_label = self.next_label("for_update");
        let end_label = self.next_label("for_end");

        self.emit_line(format!("br label %{}", cond_label));
        self.emit_label(&cond_label);
        if let Some(condition) = condition {
            let cond = self.compile_expr(condition)?;
            if cond.llvm_type != LlvmType::I1 {
                return Err("for condition must be boolean in LLVM backend".to_string());
            }
            self.emit_line(format!(
                "br i1 {}, label %{}, label %{}",
                cond.value, body_label, end_label
            ));
        } else {
            self.emit_line(format!("br label %{}", body_label));
        }

        self.emit_label(&body_label);
        self.compile_statements(body)?;
        let body_terminated = self.terminated;
        self.terminated = false;
        if !body_terminated {
            self.emit_line(format!("br label %{}", update_label));
        }

        self.emit_label(&update_label);
        if let Some(update) = update {
            self.compile_statement(update)?;
            self.terminated = false;
        }
        self.emit_line(format!("br label %{}", cond_label));

        self.emit_label(&end_label);
        self.pop_scope();
        Ok(())
    }

    fn compile_print(&mut self, value: &Node) -> Result<(), String> {
        let expr = self.compile_expr(value)?;
        match expr.llvm_type.clone() {
            LlvmType::I8 | LlvmType::I16 | LlvmType::I32 => {
                let fmt = self.global_c_string("fmt_i32", "%d\n");
                let fmt_ptr = self.string_ptr(&fmt);
                let printed = self.coerce_expr(expr, &LlvmType::I32, "print")?;
                self.emit_line(format!(
                    "call i32 (ptr, ...) @printf(ptr {}, i32 {})",
                    fmt_ptr, printed.value
                ));
                Ok(())
            }
            LlvmType::I64 => {
                let fmt = self.global_c_string("fmt_int", "%lld\n");
                let fmt_ptr = self.string_ptr(&fmt);
                self.emit_line(format!(
                    "call i32 (ptr, ...) @printf(ptr {}, i64 {})",
                    fmt_ptr, expr.value
                ));
                Ok(())
            }
            LlvmType::I1 => {
                let fmt = self.global_c_string("fmt_str", "%s\n");
                let fmt_ptr = self.string_ptr(&fmt);
                let true_str = self.global_c_string("bool_true", "true");
                let false_str = self.global_c_string("bool_false", "false");
                let true_ptr = self.string_ptr(&true_str);
                let false_ptr = self.string_ptr(&false_str);
                let bool_str = self.next_temp();
                self.emit_line(format!(
                    "{} = select i1 {}, ptr {}, ptr {}",
                    bool_str, expr.value, true_ptr, false_ptr
                ));
                self.emit_line(format!(
                    "call i32 (ptr, ...) @printf(ptr {}, ptr {})",
                    fmt_ptr, bool_str
                ));
                Ok(())
            }
            LlvmType::F32 => {
                let fmt = self.global_c_string("fmt_float", "%f\n");
                let fmt_ptr = self.string_ptr(&fmt);
                let printed = self.coerce_expr(expr, &LlvmType::F64, "print")?;
                self.emit_line(format!(
                    "call i32 (ptr, ...) @printf(ptr {}, double {})",
                    fmt_ptr, printed.value
                ));
                Ok(())
            }
            LlvmType::F64 => {
                let fmt = self.global_c_string("fmt_float", "%f\n");
                let fmt_ptr = self.string_ptr(&fmt);
                self.emit_line(format!(
                    "call i32 (ptr, ...) @printf(ptr {}, double {})",
                    fmt_ptr, expr.value
                ));
                Ok(())
            }
            LlvmType::Char16 => {
                let fmt = self.global_c_string("fmt_char", "%lc\n");
                let fmt_ptr = self.string_ptr(&fmt);
                let printed = self.coerce_expr(expr, &LlvmType::I32, "print")?;
                self.emit_line(format!(
                    "call i32 (ptr, ...) @printf(ptr {}, i32 {})",
                    fmt_ptr, printed.value
                ));
                Ok(())
            }
            LlvmType::PtrI8 => {
                let fmt = self.global_c_string("fmt_str", "%s\n");
                let fmt_ptr = self.string_ptr(&fmt);
                self.emit_line(format!(
                    "call i32 (ptr, ...) @printf(ptr {}, ptr {})",
                    fmt_ptr, expr.value
                ));
                Ok(())
            }
            LlvmType::Void => Err("cannot print a void value".to_string()),
        }
    }

    fn compile_expr(&mut self, node: &Node) -> Result<ExprValue, String> {
        match node {
            Node::Literal(Literal::Integer(value)) => Ok(ExprValue {
                llvm_type: LlvmType::I32,
                value: value.to_string(),
            }),
            Node::Literal(Literal::Long(value)) => Ok(ExprValue {
                llvm_type: LlvmType::I64,
                value: value.to_string(),
            }),
            Node::Literal(Literal::Float(value)) => Ok(ExprValue {
                llvm_type: LlvmType::F32,
                value: value.to_string(),
            }),
            Node::Literal(Literal::Double(value)) => Ok(ExprValue {
                llvm_type: LlvmType::F64,
                value: value.to_string(),
            }),
            Node::Literal(Literal::Boolean(value)) => Ok(ExprValue {
                llvm_type: LlvmType::I1,
                value: if *value { "1" } else { "0" }.to_string(),
            }),
            Node::Literal(Literal::Char(value)) => Ok(ExprValue {
                llvm_type: LlvmType::Char16,
                value: (*value as u32 as u16).to_string(),
            }),
            Node::Literal(Literal::StringLiteral(value)) => {
                let global = self.global_c_string("str", value);
                Ok(ExprValue {
                    llvm_type: LlvmType::PtrI8,
                    value: self.string_ptr(&global),
                })
            }
            Node::Identifier(name) => self.load_local(name),
            Node::Access { nodes } => {
                if nodes.len() != 1 {
                    return Err(
                        "LLVM backend currently supports only plain variable access".to_string()
                    );
                }
                match &nodes[0] {
                    Node::Identifier(name) => self.load_local(name),
                    _ => Err("LLVM backend currently supports only plain variable access"
                        .to_string()),
                }
            }
            Node::UnaryOp { operator, operand } => {
                let value = self.compile_expr(operand)?;
                match operator {
                    UnaryOperator::Plus => Ok(value),
                    UnaryOperator::Minus => {
                        if !is_numeric_llvm_type(&value.llvm_type) || value.llvm_type == LlvmType::Char16 {
                            return Err("unary `-` requires a numeric operand".to_string());
                        }
                        let target = if matches!(value.llvm_type, LlvmType::I8 | LlvmType::I16) {
                            LlvmType::I32
                        } else {
                            value.llvm_type.clone()
                        };
                        let value = self.coerce_expr(value, &target, "unary `-`")?;
                        let temp = self.next_temp();
                        let op = if matches!(target, LlvmType::F32 | LlvmType::F64) {
                            "fsub"
                        } else {
                            "sub"
                        };
                        let zero = if matches!(target, LlvmType::F32 | LlvmType::F64) {
                            if target == LlvmType::F32 { "0.0" } else { "0.0" }
                        } else {
                            "0"
                        };
                        self.emit_line(format!(
                            "{} = {} {} {}, {}",
                            temp,
                            op,
                            target.ir(),
                            zero,
                            value.value
                        ));
                        Ok(ExprValue {
                            llvm_type: target,
                            value: temp,
                        })
                    }
                    UnaryOperator::Negate => {
                        if value.llvm_type != LlvmType::I1 {
                            return Err("unary `!` requires a boolean operand".to_string());
                        }
                        let temp = self.next_temp();
                        self.emit_line(format!("{} = xor i1 {}, true", temp, value.value));
                        Ok(ExprValue {
                            llvm_type: LlvmType::I1,
                            value: temp,
                        })
                    }
                }
            }
            Node::BinaryOp {
                left,
                operator,
                right,
            } => self.compile_binary_expr(left, operator, right),
            Node::FunctionCall {
                name, arguments, ..
            } => self.compile_function_call(name, arguments),
            unsupported => Err(format!(
                "LLVM backend does not support expression `{:?}` yet",
                unsupported
            )),
        }
    }

    fn compile_function_call(
        &mut self,
        name: &str,
        arguments: &[Vec<Node>],
    ) -> Result<ExprValue, String> {
        if arguments.len() != 1 {
            return Err(
                "LLVM backend does not support chained closure-style function calls yet"
                    .to_string(),
            );
        }

        let signature = self
            .signatures
            .get(name)
            .cloned()
            .ok_or_else(|| format!("unknown function `{}` in LLVM backend", name))?;

        let provided_args = &arguments[0];
        if provided_args.len() != signature.parameters.len() {
            return Err(format!(
                "function `{}` expects {} arguments, got {}",
                name,
                signature.parameters.len(),
                provided_args.len()
            ));
        }

        let mut arg_parts = Vec::with_capacity(provided_args.len());
        for (arg_node, expected_type) in provided_args.iter().zip(signature.parameters.iter()) {
            let arg = self.compile_expr(arg_node)?;
            let arg = self.coerce_expr(arg, expected_type, "function argument")?;
            arg_parts.push(format!("{} {}", arg.llvm_type.ir(), arg.value));
        }

        if signature.return_type == LlvmType::Void {
            self.emit_line(format!(
                "call void @skunk_{}({})",
                name,
                arg_parts.join(", ")
            ));
            Ok(ExprValue {
                llvm_type: LlvmType::Void,
                value: "void".to_string(),
            })
        } else {
            let temp = self.next_temp();
            self.emit_line(format!(
                "{} = call {} @skunk_{}({})",
                temp,
                signature.return_type.ir(),
                name,
                arg_parts.join(", ")
            ));
            Ok(ExprValue {
                llvm_type: signature.return_type,
                value: temp,
            })
        }
    }

    fn compile_binary_expr(
        &mut self,
        left: &Node,
        operator: &Operator,
        right: &Node,
    ) -> Result<ExprValue, String> {
        let left = self.compile_expr(left)?;
        let right = self.compile_expr(right)?;

        if let Some(promoted) = promoted_numeric_llvm_type(&left.llvm_type, &right.llvm_type) {
            let left = self.coerce_expr(left, &promoted, "binary operand")?;
            let right = self.coerce_expr(right, &promoted, "binary operand")?;
            let temp = self.next_temp();

            if matches!(promoted, LlvmType::F32 | LlvmType::F64) {
                let op = match operator {
                    Operator::Add => "fadd",
                    Operator::Subtract => "fsub",
                    Operator::Multiply => "fmul",
                    Operator::Divide => "fdiv",
                    Operator::Equals => "fcmp oeq",
                    Operator::NotEquals => "fcmp one",
                    Operator::LessThan => "fcmp olt",
                    Operator::LessThanOrEqual => "fcmp ole",
                    Operator::GreaterThan => "fcmp ogt",
                    Operator::GreaterThanOrEqual => "fcmp oge",
                    _ => {
                        return Err(format!(
                            "LLVM backend does not support `{:?}` for floating-point values",
                            operator
                        ))
                    }
                };
                self.emit_line(format!(
                    "{} = {} {} {}, {}",
                    temp,
                    op,
                    promoted.ir(),
                    left.value,
                    right.value
                ));
                return Ok(ExprValue {
                    llvm_type: if matches!(
                        operator,
                        Operator::Equals
                            | Operator::NotEquals
                            | Operator::LessThan
                            | Operator::LessThanOrEqual
                            | Operator::GreaterThan
                            | Operator::GreaterThanOrEqual
                    ) {
                        LlvmType::I1
                    } else {
                        promoted
                    },
                    value: temp,
                });
            }

            let op = match operator {
                Operator::Add => "add",
                Operator::Subtract => "sub",
                Operator::Multiply => "mul",
                Operator::Divide => "sdiv",
                Operator::Mod => "srem",
                Operator::Equals => "icmp eq",
                Operator::NotEquals => "icmp ne",
                Operator::LessThan => "icmp slt",
                Operator::LessThanOrEqual => "icmp sle",
                Operator::GreaterThan => "icmp sgt",
                Operator::GreaterThanOrEqual => "icmp sge",
                _ => {
                    return Err(format!(
                        "LLVM backend does not support `{:?}` for numeric values",
                        operator
                    ))
                }
            };
            self.emit_line(format!(
                "{} = {} {} {}, {}",
                temp,
                op,
                promoted.ir(),
                left.value,
                right.value
            ));
            return Ok(ExprValue {
                llvm_type: if matches!(
                    operator,
                    Operator::Equals
                        | Operator::NotEquals
                        | Operator::LessThan
                        | Operator::LessThanOrEqual
                        | Operator::GreaterThan
                        | Operator::GreaterThanOrEqual
                ) {
                    LlvmType::I1
                } else {
                    promoted
                },
                value: temp,
            });
        }

        match (&left.llvm_type, operator, &right.llvm_type) {
            (LlvmType::I1, Operator::And, LlvmType::I1)
            | (LlvmType::I1, Operator::Or, LlvmType::I1) => {
                let op = match operator {
                    Operator::And => "and",
                    Operator::Or => "or",
                    _ => unreachable!(),
                };
                let temp = self.next_temp();
                self.emit_line(format!(
                    "{} = {} i1 {}, {}",
                    temp, op, left.value, right.value
                ));
                Ok(ExprValue {
                    llvm_type: LlvmType::I1,
                    value: temp,
                })
            }
            (LlvmType::I1, Operator::Equals, LlvmType::I1)
            | (LlvmType::I1, Operator::NotEquals, LlvmType::I1) => {
                let pred = match operator {
                    Operator::Equals => "eq",
                    Operator::NotEquals => "ne",
                    _ => unreachable!(),
                };
                let temp = self.next_temp();
                self.emit_line(format!(
                    "{} = icmp {} i1 {}, {}",
                    temp, pred, left.value, right.value
                ));
                Ok(ExprValue {
                    llvm_type: LlvmType::I1,
                    value: temp,
                })
            }
            (LlvmType::Char16, Operator::Equals, LlvmType::Char16)
            | (LlvmType::Char16, Operator::NotEquals, LlvmType::Char16)
            | (LlvmType::Char16, Operator::LessThan, LlvmType::Char16)
            | (LlvmType::Char16, Operator::LessThanOrEqual, LlvmType::Char16)
            | (LlvmType::Char16, Operator::GreaterThan, LlvmType::Char16)
            | (LlvmType::Char16, Operator::GreaterThanOrEqual, LlvmType::Char16) => {
                let pred = match operator {
                    Operator::Equals => "eq",
                    Operator::NotEquals => "ne",
                    Operator::LessThan => "ult",
                    Operator::LessThanOrEqual => "ule",
                    Operator::GreaterThan => "ugt",
                    Operator::GreaterThanOrEqual => "uge",
                    _ => unreachable!(),
                };
                let temp = self.next_temp();
                self.emit_line(format!(
                    "{} = icmp {} i16 {}, {}",
                    temp, pred, left.value, right.value
                ));
                Ok(ExprValue {
                    llvm_type: LlvmType::I1,
                    value: temp,
                })
            }
            (LlvmType::PtrI8, Operator::Equals, LlvmType::PtrI8)
            | (LlvmType::PtrI8, Operator::NotEquals, LlvmType::PtrI8) => {
                let pred = match operator {
                    Operator::Equals => "eq",
                    Operator::NotEquals => "ne",
                    _ => unreachable!(),
                };
                let temp = self.next_temp();
                self.emit_line(format!(
                    "{} = icmp {} ptr {}, {}",
                    temp, pred, left.value, right.value
                ));
                Ok(ExprValue {
                    llvm_type: LlvmType::I1,
                    value: temp,
                })
            }
            _ => Err(format!(
                "LLVM backend does not support binary op `{:?}` for `{}` and `{}` yet",
                operator,
                left.llvm_type.ir(),
                right.llvm_type.ir()
            )),
        }
    }

    fn load_local(&mut self, name: &str) -> Result<ExprValue, String> {
        let local = self
            .lookup_local(name)
            .cloned()
            .ok_or_else(|| format!("unknown variable `{}` in LLVM backend", name))?;
        let temp = self.next_temp();
        self.emit_line(format!(
            "{} = load {}, ptr {}, align {}",
            temp,
            local.llvm_type.ir(),
            local.ptr,
            Self::align_of(&local.llvm_type)
        ));
        Ok(ExprValue {
            llvm_type: local.llvm_type,
            value: temp,
        })
    }

    fn resolve_local_from_access(&self, node: &Node) -> Result<LocalVar, String> {
        match node {
            Node::Access { nodes } if nodes.len() == 1 => match &nodes[0] {
                Node::Identifier(name) => self
                    .lookup_local(name)
                    .cloned()
                    .ok_or_else(|| format!("unknown variable `{}` in LLVM backend", name)),
                _ => Err("LLVM backend currently supports only plain variable assignments"
                    .to_string()),
            },
            _ => Err("LLVM backend currently supports only plain variable assignments"
                .to_string()),
        }
    }

    fn default_value(&self, llvm_type: &LlvmType) -> ExprValue {
        match llvm_type {
            LlvmType::I8 => ExprValue {
                llvm_type: LlvmType::I8,
                value: "0".to_string(),
            },
            LlvmType::I16 => ExprValue {
                llvm_type: LlvmType::I16,
                value: "0".to_string(),
            },
            LlvmType::I32 => ExprValue {
                llvm_type: LlvmType::I32,
                value: "0".to_string(),
            },
            LlvmType::I64 => ExprValue {
                llvm_type: LlvmType::I64,
                value: "0".to_string(),
            },
            LlvmType::F32 => ExprValue {
                llvm_type: LlvmType::F32,
                value: "0.0".to_string(),
            },
            LlvmType::F64 => ExprValue {
                llvm_type: LlvmType::F64,
                value: "0.0".to_string(),
            },
            LlvmType::Char16 => ExprValue {
                llvm_type: LlvmType::Char16,
                value: "0".to_string(),
            },
            LlvmType::I1 => ExprValue {
                llvm_type: LlvmType::I1,
                value: "0".to_string(),
            },
            LlvmType::PtrI8 => ExprValue {
                llvm_type: LlvmType::PtrI8,
                value: "null".to_string(),
            },
            LlvmType::Void => ExprValue {
                llvm_type: LlvmType::Void,
                value: "void".to_string(),
            },
        }
    }

    fn coerce_expr(
        &mut self,
        value: ExprValue,
        expected: &LlvmType,
        context: &str,
    ) -> Result<ExprValue, String> {
        if &value.llvm_type == expected {
            return Ok(value);
        }

        let temp = self.next_temp();
        let line = match (&value.llvm_type, expected) {
            (LlvmType::I8, LlvmType::I16) => format!("{} = sext i8 {} to i16", temp, value.value),
            (LlvmType::I8, LlvmType::I32) => format!("{} = sext i8 {} to i32", temp, value.value),
            (LlvmType::I8, LlvmType::I64) => format!("{} = sext i8 {} to i64", temp, value.value),
            (LlvmType::I16, LlvmType::I32) => format!("{} = sext i16 {} to i32", temp, value.value),
            (LlvmType::I16, LlvmType::I64) => format!("{} = sext i16 {} to i64", temp, value.value),
            (LlvmType::I32, LlvmType::I64) => format!("{} = sext i32 {} to i64", temp, value.value),
            (LlvmType::Char16, LlvmType::I32) => format!("{} = zext i16 {} to i32", temp, value.value),
            (LlvmType::I32, LlvmType::I16) => format!("{} = trunc i32 {} to i16", temp, value.value),
            (LlvmType::I32, LlvmType::I8) => format!("{} = trunc i32 {} to i8", temp, value.value),
            (LlvmType::I64, LlvmType::I32) => format!("{} = trunc i64 {} to i32", temp, value.value),
            (LlvmType::I64, LlvmType::I16) => format!("{} = trunc i64 {} to i16", temp, value.value),
            (LlvmType::I64, LlvmType::I8) => format!("{} = trunc i64 {} to i8", temp, value.value),
            (LlvmType::I8, LlvmType::F32) => format!("{} = sitofp i8 {} to float", temp, value.value),
            (LlvmType::I8, LlvmType::F64) => format!("{} = sitofp i8 {} to double", temp, value.value),
            (LlvmType::I16, LlvmType::F32) => format!("{} = sitofp i16 {} to float", temp, value.value),
            (LlvmType::I16, LlvmType::F64) => format!("{} = sitofp i16 {} to double", temp, value.value),
            (LlvmType::I32, LlvmType::F32) => format!("{} = sitofp i32 {} to float", temp, value.value),
            (LlvmType::I32, LlvmType::F64) => format!("{} = sitofp i32 {} to double", temp, value.value),
            (LlvmType::I64, LlvmType::F32) => format!("{} = sitofp i64 {} to float", temp, value.value),
            (LlvmType::I64, LlvmType::F64) => format!("{} = sitofp i64 {} to double", temp, value.value),
            (LlvmType::F32, LlvmType::F64) => format!("{} = fpext float {} to double", temp, value.value),
            (LlvmType::F64, LlvmType::F32) => format!("{} = fptrunc double {} to float", temp, value.value),
            _ => {
                return Err(format!(
                    "type mismatch in {}: expected `{}`, got `{}`",
                    context,
                    expected.ir(),
                    value.llvm_type.ir()
                ))
            }
        };

        self.emit_line(line);
        Ok(ExprValue {
            llvm_type: expected.clone(),
            value: temp,
        })
    }

    fn global_c_string(&mut self, prefix: &str, value: &str) -> String {
        let mut bytes = value.as_bytes().to_vec();
        bytes.push(0);
        if let Some(existing) = self.globals.iter().find(|g| g.bytes == bytes) {
            return existing.name.clone();
        }
        let name = format!("{}.{}", prefix, self.globals.len());
        self.globals.push(GlobalString { name: name.clone(), bytes });
        name
    }

    fn string_ptr(&self, global: &str) -> String {
        format!("getelementptr inbounds (i8, ptr @{}, i64 0)", global)
    }

    fn emit_store(&mut self, ptr: &str, expr: &ExprValue) {
        self.emit_line(format!(
            "store {} {}, ptr {}, align {}",
            expr.llvm_type.ir(),
            expr.value,
            ptr,
            Self::align_of(&expr.llvm_type)
        ));
    }

    fn emit_alloca(&mut self, llvm_type: LlvmType, hint: &str) -> String {
        let ptr = format!("%{}_{}", sanitize_name(hint), self.temp_counter);
        self.temp_counter += 1;
        self.emit_line(format!(
            "{} = alloca {}, align {}",
            ptr,
            llvm_type.ir(),
            Self::align_of(&llvm_type)
        ));
        ptr
    }

    fn emit_label(&mut self, label: &str) {
        self.lines.push(format!("{}:", label));
    }

    fn emit_line(&mut self, line: String) {
        self.lines.push(format!("  {}", line));
    }

    fn next_temp(&mut self) -> String {
        let name = format!("%t{}", self.temp_counter);
        self.temp_counter += 1;
        name
    }

    fn next_label(&mut self, prefix: &str) -> String {
        let label = format!("{}_{}_{}", sanitize_name(self.function_name), prefix, self.label_counter);
        self.label_counter += 1;
        label
    }

    fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    fn declare_local(&mut self, name: String, ptr: String, llvm_type: LlvmType) {
        self.scopes
            .last_mut()
            .expect("scope stack should never be empty")
            .insert(name, LocalVar { ptr, llvm_type });
    }

    fn lookup_local(&self, name: &str) -> Option<&LocalVar> {
        self.scopes.iter().rev().find_map(|scope| scope.get(name))
    }

    fn align_of(llvm_type: &LlvmType) -> usize {
        match llvm_type {
            LlvmType::I8 | LlvmType::I1 => 1,
            LlvmType::I16 | LlvmType::Char16 => 2,
            LlvmType::I32 | LlvmType::F32 => 4,
            LlvmType::I64 | LlvmType::F64 | LlvmType::PtrI8 => 8,
            LlvmType::Void => 1,
        }
    }
}

fn sanitize_name(name: &str) -> String {
    name.chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
        .collect()
}

pub struct CompiledArtifact {
    pub llvm_ir_path: PathBuf,
    pub binary_path: PathBuf,
}

pub fn compile_to_executable(
    program: &Node,
    source_path: &Path,
    output_path: &Path,
) -> Result<CompiledArtifact, String> {
    let llvm_ir = compile_to_llvm_ir(program)?;
    let llvm_ir_path = output_path.with_extension("ll");
    fs::write(&llvm_ir_path, llvm_ir)
        .map_err(|err| format!("failed to write LLVM IR to {}: {}", llvm_ir_path.display(), err))?;

    let status = Command::new("clang")
        .arg(&llvm_ir_path)
        .arg("-O2")
        .arg("-o")
        .arg(output_path)
        .status()
        .map_err(|err| {
            format!(
                "failed to invoke clang while compiling {}: {}",
                source_path.display(),
                err
            )
        })?;

    if !status.success() {
        return Err(format!(
            "clang failed while compiling {} to {}",
            source_path.display(),
            output_path.display()
        ));
    }

    Ok(CompiledArtifact {
        llvm_ir_path,
        binary_path: output_path.to_path_buf(),
    })
}

pub fn compile_to_llvm_ir(program: &Node) -> Result<String, String> {
    let statements = match program {
        Node::Program { statements } => statements,
        other => {
            return Err(format!(
                "expected a program root for LLVM compilation, found `{:?}`",
                other
            ))
        }
    };

    let mut signatures = HashMap::<String, FunctionSignature>::new();
    let mut functions = Vec::<&Node>::new();

    for statement in statements {
        match statement {
            Node::FunctionDeclaration {
                name,
                parameters,
                return_type,
                ..
            } => {
                let params = parameters
                    .iter()
                    .map(|(_, ty)| llvm_type(ty))
                    .collect::<Result<Vec<_>, _>>()?;
                let return_type = llvm_type(return_type)?;
                signatures.insert(
                    name.clone(),
                    FunctionSignature {
                        return_type,
                        parameters: params,
                    },
                );
                functions.push(statement);
            }
            Node::EOI => {}
            Node::StructDeclaration { .. } => {
                return Err(
                    "LLVM backend does not support structs yet; the interpreter still does"
                        .to_string(),
                );
            }
            other => {
                return Err(format!(
                    "LLVM backend currently expects top-level function declarations only, found `{:?}`",
                    other
                ))
            }
        }
    }

    if !signatures.contains_key("main") {
        return Err("LLVM backend currently requires `function main(): ... {}`".to_string());
    }

    let mut globals = Vec::<GlobalString>::new();
    let mut function_irs = Vec::<String>::new();

    for function in functions {
        if let Node::FunctionDeclaration {
            name,
            parameters,
            return_type,
            body,
            ..
        } = function
        {
            let return_type = llvm_type(return_type)?;
            let param_defs = parameters
                .iter()
                .enumerate()
                .map(|(index, (_, ty))| Ok(format!("{} %arg{}", llvm_type(ty)?.ir(), index)))
                .collect::<Result<Vec<_>, String>>()?;
            let compiler = FunctionCompiler::new(name, return_type.clone(), &signatures, &mut globals);
            let body_lines = compiler.compile(parameters, body)?;
            let mut function_ir = String::new();
            let _ = writeln!(
                function_ir,
                "define {} @skunk_{}({}) {{",
                return_type.ir(),
                name,
                param_defs.join(", ")
            );
            let _ = writeln!(function_ir, "entry:");
            for line in body_lines {
                let _ = writeln!(function_ir, "{}", line);
            }
            let _ = writeln!(function_ir, "}}");
            function_irs.push(function_ir);
        }
    }

    let main_signature = signatures
        .get("main")
        .expect("validated above");
    if !main_signature.parameters.is_empty() {
        return Err("LLVM backend requires `main` to take no parameters".to_string());
    }

    let c_main_body = match main_signature.return_type {
        LlvmType::I8 => {
            "  %result = call i8 @skunk_main()\n  %exit_code = sext i8 %result to i32\n  ret i32 %exit_code\n"
        }
        LlvmType::I16 => {
            "  %result = call i16 @skunk_main()\n  %exit_code = sext i16 %result to i32\n  ret i32 %exit_code\n"
        }
        LlvmType::I32 => "  %result = call i32 @skunk_main()\n  ret i32 %result\n",
        LlvmType::I64 => "  %result = call i64 @skunk_main()\n  %exit_code = trunc i64 %result to i32\n  ret i32 %exit_code\n",
        LlvmType::F32 => {
            "  %result = call float @skunk_main()\n  %exit_code = fptosi float %result to i32\n  ret i32 %exit_code\n"
        }
        LlvmType::F64 => {
            "  %result = call double @skunk_main()\n  %exit_code = fptosi double %result to i32\n  ret i32 %exit_code\n"
        }
        LlvmType::Char16 => {
            "  %result = call i16 @skunk_main()\n  %exit_code = zext i16 %result to i32\n  ret i32 %exit_code\n"
        }
        LlvmType::I1 => "  %result = call i1 @skunk_main()\n  %exit_code = zext i1 %result to i32\n  ret i32 %exit_code\n",
        LlvmType::Void => "  call void @skunk_main()\n  ret i32 0\n",
        LlvmType::PtrI8 => {
            return Err("LLVM backend does not support `main` returning string yet".to_string())
        }
    };

    let mut ir = String::new();
    let _ = writeln!(ir, "declare i32 @printf(ptr, ...)");
    let _ = writeln!(ir);
    for global in &globals {
        let _ = writeln!(ir, "{}", global.ir_decl());
    }
    if !globals.is_empty() {
        let _ = writeln!(ir);
    }
    for function_ir in function_irs {
        let _ = writeln!(ir, "{}", function_ir);
    }
    let _ = writeln!(ir, "define i32 @main() {{");
    let _ = write!(ir, "{}", c_main_body);
    let _ = writeln!(ir, "}}");

    Ok(ir)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compiles_basic_program_to_ir() {
        let program = ast::parse(
            r#"
            function add(a: int, b: int): int {
                return a + b;
            }

            function main(): void {
                total: int = add(2, 3);
                print(total);
            }
            "#,
        );

        let ir = compile_to_llvm_ir(&program).unwrap();
        assert!(ir.contains("define i32 @skunk_add(i32 %arg0, i32 %arg1)"));
        assert!(ir.contains("define void @skunk_main()"));
        assert!(ir.contains("call i32 (ptr, ...) @printf"));
    }

    #[test]
    fn rejects_structs_for_now() {
        let program = ast::parse(
            r#"
            struct Point {
                x: int;
            }

            function main(): void {}
            "#,
        );

        let err = compile_to_llvm_ir(&program).unwrap_err();
        assert!(err.contains("does not support structs yet"));
    }

    #[test]
    fn compiles_new_primitives_to_ir() {
        let program = ast::parse(
            r#"
            function main(): double {
                b: byte = 10;
                s: short = 20;
                l: long = 30L;
                f: float = 1.5f;
                c: char = 'A';
                print(c);
                return l + f;
            }
            "#,
        );

        let ir = compile_to_llvm_ir(&program).unwrap();
        assert!(ir.contains("define double @skunk_main()"));
        assert!(ir.contains("sitofp i64"));
        assert!(ir.contains("fadd float"));
    }
}
