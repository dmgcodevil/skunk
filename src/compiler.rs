use crate::ast::{self, Literal, Node, Operator, Type, UnaryOperator};
use std::collections::HashMap;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Clone, Debug, PartialEq, Eq)]
enum LlvmType {
    I64,
    I1,
    PtrI8,
    Void,
}

impl LlvmType {
    fn ir(&self) -> &'static str {
        match self {
            LlvmType::I64 => "i64",
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
        Type::Int => Ok(LlvmType::I64),
        Type::Boolean => Ok(LlvmType::I1),
        Type::String => Ok(LlvmType::PtrI8),
        Type::Void => Ok(LlvmType::Void),
        other => Err(format!(
            "LLVM backend does not support type `{}` yet",
            ast::type_to_string(other)
        )),
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
                self.ensure_type(&init, &llvm_type, "variable declaration")?;
                self.emit_store(&ptr, &init);
                self.declare_local(name.clone(), ptr, llvm_type);
                Ok(())
            }
            Node::Assignment { var, value, .. } => {
                let expr = self.compile_expr(value)?;
                let local = self.resolve_local_from_access(var)?;
                self.ensure_type(&expr, &local.llvm_type, "assignment")?;
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
                        self.ensure_type(&expr, &self.return_type, "return")?;
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
        self.ensure_type(&cond, &LlvmType::I1, "if condition")?;
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
            self.ensure_type(&cond, &LlvmType::I1, "for condition")?;
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
        match expr.llvm_type {
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
                llvm_type: LlvmType::I64,
                value: value.to_string(),
            }),
            Node::Literal(Literal::Boolean(value)) => Ok(ExprValue {
                llvm_type: LlvmType::I1,
                value: if *value { "1" } else { "0" }.to_string(),
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
                        self.ensure_type(&value, &LlvmType::I64, "unary `-`")?;
                        let temp = self.next_temp();
                        self.emit_line(format!("{} = sub i64 0, {}", temp, value.value));
                        Ok(ExprValue {
                            llvm_type: LlvmType::I64,
                            value: temp,
                        })
                    }
                    UnaryOperator::Negate => {
                        self.ensure_type(&value, &LlvmType::I1, "unary `!`")?;
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
            self.ensure_type(&arg, expected_type, "function argument")?;
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

        match (&left.llvm_type, operator, &right.llvm_type) {
            (LlvmType::I64, Operator::Add, LlvmType::I64)
            | (LlvmType::I64, Operator::Subtract, LlvmType::I64)
            | (LlvmType::I64, Operator::Multiply, LlvmType::I64)
            | (LlvmType::I64, Operator::Divide, LlvmType::I64)
            | (LlvmType::I64, Operator::Mod, LlvmType::I64) => {
                let op = match operator {
                    Operator::Add => "add",
                    Operator::Subtract => "sub",
                    Operator::Multiply => "mul",
                    Operator::Divide => "sdiv",
                    Operator::Mod => "srem",
                    _ => unreachable!(),
                };
                let temp = self.next_temp();
                self.emit_line(format!(
                    "{} = {} i64 {}, {}",
                    temp, op, left.value, right.value
                ));
                Ok(ExprValue {
                    llvm_type: LlvmType::I64,
                    value: temp,
                })
            }
            (LlvmType::I64, Operator::Equals, LlvmType::I64)
            | (LlvmType::I64, Operator::NotEquals, LlvmType::I64)
            | (LlvmType::I64, Operator::LessThan, LlvmType::I64)
            | (LlvmType::I64, Operator::LessThanOrEqual, LlvmType::I64)
            | (LlvmType::I64, Operator::GreaterThan, LlvmType::I64)
            | (LlvmType::I64, Operator::GreaterThanOrEqual, LlvmType::I64) => {
                let pred = match operator {
                    Operator::Equals => "eq",
                    Operator::NotEquals => "ne",
                    Operator::LessThan => "slt",
                    Operator::LessThanOrEqual => "sle",
                    Operator::GreaterThan => "sgt",
                    Operator::GreaterThanOrEqual => "sge",
                    _ => unreachable!(),
                };
                let temp = self.next_temp();
                self.emit_line(format!(
                    "{} = icmp {} i64 {}, {}",
                    temp, pred, left.value, right.value
                ));
                Ok(ExprValue {
                    llvm_type: LlvmType::I1,
                    value: temp,
                })
            }
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
            LlvmType::I64 => ExprValue {
                llvm_type: LlvmType::I64,
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

    fn ensure_type(
        &self,
        value: &ExprValue,
        expected: &LlvmType,
        context: &str,
    ) -> Result<(), String> {
        if &value.llvm_type == expected {
            Ok(())
        } else {
            Err(format!(
                "type mismatch in {}: expected `{}`, got `{}`",
                context,
                expected.ir(),
                value.llvm_type.ir()
            ))
        }
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
            LlvmType::I64 | LlvmType::PtrI8 => 8,
            LlvmType::I1 => 1,
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
        LlvmType::I64 => "  %result = call i64 @skunk_main()\n  %exit_code = trunc i64 %result to i32\n  ret i32 %exit_code\n",
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
        assert!(ir.contains("define i64 @skunk_add(i64 %arg0, i64 %arg1)"));
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
}
