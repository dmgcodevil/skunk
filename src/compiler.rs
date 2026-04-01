use crate::ast::{self, Literal, Node, Operator, Type, UnaryOperator};
use std::collections::{BTreeMap, HashMap};
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
    Allocator,
    Arena,
    Window,
    TraitObject(String),
    Struct(String),
    Enum(String),
    Reference {
        target_type: Box<LlvmType>,
        mutable: bool,
    },
    Pointer {
        target_type: Box<LlvmType>,
    },
    Function {
        parameters: Vec<LlvmType>,
        return_type: Box<LlvmType>,
    },
    Slice {
        elem_type: Box<LlvmType>,
    },
    Array {
        elem_type: Box<LlvmType>,
        len: usize,
    },
    Void,
}

impl LlvmType {
    fn ir(&self) -> String {
        match self {
            LlvmType::I8 => "i8".to_string(),
            LlvmType::I16 => "i16".to_string(),
            LlvmType::I32 => "i32".to_string(),
            LlvmType::I64 => "i64".to_string(),
            LlvmType::F32 => "float".to_string(),
            LlvmType::F64 => "double".to_string(),
            LlvmType::Char16 => "i16".to_string(),
            LlvmType::I1 => "i1".to_string(),
            LlvmType::PtrI8 => "ptr".to_string(),
            LlvmType::Allocator => "ptr".to_string(),
            LlvmType::Arena => "ptr".to_string(),
            LlvmType::Window => "ptr".to_string(),
            LlvmType::TraitObject(name) => format!("%trait.{}", sanitize_name(name)),
            LlvmType::Struct(name) => format!("%struct.{}", sanitize_name(name)),
            LlvmType::Enum(name) => format!("%enum.{}", sanitize_name(name)),
            LlvmType::Reference { .. } => "ptr".to_string(),
            LlvmType::Pointer { .. } => "ptr".to_string(),
            LlvmType::Function { .. } => "{ ptr, ptr }".to_string(),
            LlvmType::Slice { .. } => "{ ptr, i32 }".to_string(),
            LlvmType::Array { elem_type, len } => format!("[{} x {}]", len, elem_type.ir()),
            LlvmType::Void => "void".to_string(),
        }
    }
}

fn is_pointer_like_llvm_type(llvm_type: &LlvmType) -> bool {
    matches!(
        llvm_type,
        LlvmType::Pointer { .. } | LlvmType::Reference { .. }
    )
}

#[derive(Clone, Debug)]
struct FunctionSignature {
    symbol_name: String,
    return_type: LlvmType,
    parameters: Vec<LlvmType>,
}

#[derive(Clone, Debug)]
struct FunctionPlan {
    symbol_name: String,
    parameters: Vec<(String, Type)>,
    return_type: Type,
    body: Vec<Node>,
    is_method: bool,
}

#[derive(Clone, Debug)]
struct StructLayout {
    name: String,
    fields: Vec<(String, LlvmType)>,
}

#[derive(Clone, Debug)]
struct EnumVariantLayout {
    name: String,
    tag: usize,
    payload_types: Vec<LlvmType>,
    field_indices: Vec<usize>,
}

#[derive(Clone, Debug)]
struct EnumLayout {
    name: String,
    variants: Vec<EnumVariantLayout>,
}

#[derive(Clone, Debug)]
struct TraitMethodLayout {
    name: String,
    receiver_is_mut: bool,
    return_type: LlvmType,
    parameters: Vec<LlvmType>,
}

#[derive(Clone, Debug)]
struct TraitLayout {
    name: String,
    methods: Vec<TraitMethodLayout>,
}

#[derive(Clone, Debug)]
struct ClosureEnv {
    type_name: String,
    captures: Vec<(String, LlvmType)>,
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

fn llvm_type(
    sk_type: &Type,
    structs: &HashMap<String, StructLayout>,
    enums: &HashMap<String, EnumLayout>,
    traits: &HashMap<String, TraitLayout>,
) -> Result<LlvmType, String> {
    match sk_type {
        Type::Const { inner } | Type::BindingConst { inner } => {
            llvm_type(inner, structs, enums, traits)
        }
        Type::Byte => Ok(LlvmType::I8),
        Type::Short => Ok(LlvmType::I16),
        Type::Int => Ok(LlvmType::I32),
        Type::Long => Ok(LlvmType::I64),
        Type::Float => Ok(LlvmType::F32),
        Type::Double => Ok(LlvmType::F64),
        Type::Boolean => Ok(LlvmType::I1),
        Type::String => Ok(LlvmType::PtrI8),
        Type::Char => Ok(LlvmType::Char16),
        Type::Allocator => Ok(LlvmType::Allocator),
        Type::Arena => Ok(LlvmType::Arena),
        Type::Array {
            elem_type,
            dimensions,
        } => {
            let mut llvm_elem = llvm_type(elem_type, structs, enums, traits)?;
            for dimension in dimensions.iter().rev() {
                llvm_elem = LlvmType::Array {
                    elem_type: Box::new(llvm_elem),
                    len: array_len_from_dimension(dimension)?,
                };
            }
            Ok(llvm_elem)
        }
        Type::Slice { elem_type } => Ok(LlvmType::Slice {
            elem_type: Box::new(llvm_type(elem_type, structs, enums, traits)?),
        }),
        Type::Reference {
            target_type,
            mutable,
        } => Ok(LlvmType::Reference {
            target_type: Box::new(llvm_type(target_type, structs, enums, traits)?),
            mutable: *mutable,
        }),
        Type::Pointer { target_type } => Ok(LlvmType::Pointer {
            target_type: Box::new(llvm_type(target_type, structs, enums, traits)?),
        }),
        Type::Function {
            parameters,
            return_type,
        } => Ok(LlvmType::Function {
            parameters: parameters
                .iter()
                .map(|param| llvm_type(param, structs, enums, traits))
                .collect::<Result<Vec<_>, _>>()?,
            return_type: Box::new(llvm_type(return_type, structs, enums, traits)?),
        }),
        Type::Custom(name) => {
            if name == "Color" {
                Ok(LlvmType::I32)
            } else if name == "Window" {
                Ok(LlvmType::Window)
            } else if structs.contains_key(name) {
                Ok(LlvmType::Struct(name.clone()))
            } else if enums.contains_key(name) {
                Ok(LlvmType::Enum(name.clone()))
            } else if traits.contains_key(name) {
                Ok(LlvmType::TraitObject(name.clone()))
            } else {
                Err(format!("unknown nominal type `{}` in LLVM backend", name))
            }
        }
        Type::Void => Ok(LlvmType::Void),
        other => Err(format!(
            "LLVM backend does not support type `{}` yet",
            ast::type_to_string(other)
        )),
    }
}

fn array_len_from_dimension(dimension: &Node) -> Result<usize, String> {
    let value = match dimension {
        Node::Literal(Literal::Integer(value)) => *value,
        Node::Literal(Literal::Long(value)) => *value,
        other => {
            return Err(format!(
                "LLVM backend requires array dimensions to be integer literals, found `{:?}`",
                other
            ))
        }
    };
    usize::try_from(value).map_err(|_| {
        format!(
            "LLVM backend requires non-negative array dimensions, found `{}`",
            value
        )
    })
}

fn collect_struct_layouts(statements: &[Node]) -> Result<HashMap<String, StructLayout>, String> {
    let mut raw_fields = HashMap::<String, Vec<(String, Type)>>::new();
    let enum_placeholders = collect_enum_placeholders(statements);
    for statement in statements {
        if let Node::StructDeclaration { name, fields, .. } = statement {
            raw_fields.insert(name.clone(), fields.clone());
        }
    }

    let mut layouts = HashMap::<String, StructLayout>::new();
    for (name, fields) in &raw_fields {
        let llvm_fields = fields
            .iter()
            .map(|(field_name, field_type)| {
                Ok((
                    field_name.clone(),
                        llvm_type(
                            field_type,
                            &layouts_with_raw(raw_fields.keys(), &layouts),
                            &enum_placeholders,
                            &HashMap::new(),
                        )?,
                ))
            })
            .collect::<Result<Vec<_>, String>>()?;
        layouts.insert(
            name.clone(),
            StructLayout {
                name: name.clone(),
                fields: llvm_fields,
            },
        );
    }
    Ok(layouts)
}

fn collect_enum_layouts(
    statements: &[Node],
    structs: &HashMap<String, StructLayout>,
) -> Result<HashMap<String, EnumLayout>, String> {
    let mut layouts = HashMap::<String, EnumLayout>::new();
    let enum_placeholders = collect_enum_placeholders(statements);

    for statement in statements {
        let Node::EnumDeclaration { name, variants } = statement else {
            continue;
        };

        let mut variant_layouts = Vec::new();
        let mut next_field_index = 1usize;
        for (tag, variant) in variants.iter().enumerate() {
            let payload_types = variant
                .payload_types
                .iter()
                .map(|payload_type| llvm_type(payload_type, structs, &enum_placeholders, &HashMap::new()))
                .collect::<Result<Vec<_>, String>>()?;
            let field_indices = (0..payload_types.len())
                .map(|_| {
                    let current = next_field_index;
                    next_field_index += 1;
                    current
                })
                .collect::<Vec<_>>();
            variant_layouts.push(EnumVariantLayout {
                name: variant.name.clone(),
                tag,
                payload_types,
                field_indices,
            });
        }

        layouts.insert(
            name.clone(),
            EnumLayout {
                name: name.clone(),
                variants: variant_layouts,
            },
        );
    }

    Ok(layouts)
}

fn collect_enum_placeholders(statements: &[Node]) -> HashMap<String, EnumLayout> {
    statements
        .iter()
        .filter_map(|statement| match statement {
            Node::EnumDeclaration { name, .. } => Some((
                name.clone(),
                EnumLayout {
                    name: name.clone(),
                    variants: Vec::new(),
                },
            )),
            _ => None,
        })
        .collect()
}

fn collect_trait_layouts(
    statements: &[Node],
    structs: &HashMap<String, StructLayout>,
    enums: &HashMap<String, EnumLayout>,
) -> Result<HashMap<String, TraitLayout>, String> {
    let trait_decls = statements
        .iter()
        .filter_map(|statement| match statement {
            Node::TraitDeclaration {
                name,
                supertraits,
                methods,
            } => Some((name.clone(), (supertraits.clone(), methods.clone()))),
            _ => None,
        })
        .collect::<HashMap<_, _>>();
    fn build_trait_layout(
        trait_name: &str,
        trait_decls: &HashMap<String, (Vec<String>, Vec<ast::TraitMethodSignature>)>,
        structs: &HashMap<String, StructLayout>,
        enums: &HashMap<String, EnumLayout>,
        layouts: &mut HashMap<String, TraitLayout>,
        visiting: &mut Vec<String>,
    ) -> Result<TraitLayout, String> {
        if let Some(layout) = layouts.get(trait_name) {
            return Ok(layout.clone());
        }
        if visiting.iter().any(|name| name == trait_name) {
            visiting.push(trait_name.to_string());
            return Err(format!(
                "cyclic supertrait relationship detected: {}",
                visiting.join(" -> ")
            ));
        }
        let (supertraits, methods) = trait_decls
            .get(trait_name)
            .cloned()
            .ok_or_else(|| format!("unknown trait `{}` in LLVM backend", trait_name))?;
        visiting.push(trait_name.to_string());
        let mut method_layouts = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for supertrait in supertraits {
            let layout = build_trait_layout(
                &supertrait,
                trait_decls,
                structs,
                enums,
                layouts,
                visiting,
            )?;
            for method in layout.methods {
                if seen.insert(method.name.clone()) {
                    method_layouts.push(method);
                }
            }
        }
        for method in methods {
            let receiver_type = method
                .parameters
                .first()
                .map(|(_, ty)| ty.clone())
                .ok_or_else(|| format!("trait method `{}` is missing self", method.name))?;
            if !seen.insert(method.name.clone()) {
                return Err(format!(
                    "trait `{}` declares duplicate inherited method `{}`",
                    trait_name, method.name
                ));
            }
            method_layouts.push(TraitMethodLayout {
                name: method.name.clone(),
                receiver_is_mut: matches!(receiver_type, Type::MutSelf),
                return_type: llvm_type(&method.return_type, structs, enums, layouts)?,
                parameters: method
                    .parameters
                    .iter()
                    .skip(1)
                    .map(|(_, ty)| llvm_type(ty, structs, enums, layouts))
                    .collect::<Result<Vec<_>, String>>()?,
            });
        }
        visiting.pop();
        let layout = TraitLayout {
            name: trait_name.to_string(),
            methods: method_layouts,
        };
        layouts.insert(trait_name.to_string(), layout.clone());
        Ok(layout)
    }

    let mut layouts = HashMap::new();
    for trait_name in trait_decls.keys() {
        build_trait_layout(
            trait_name,
            &trait_decls,
            structs,
            enums,
            &mut layouts,
            &mut Vec::new(),
        )?;
    }
    Ok(layouts)
}

fn layouts_with_raw<'a>(
    raw_names: impl Iterator<Item = &'a String>,
    layouts: &HashMap<String, StructLayout>,
) -> HashMap<String, StructLayout> {
    let mut merged = layouts.clone();
    for name in raw_names {
        merged.entry(name.clone()).or_insert_with(|| StructLayout {
            name: name.clone(),
            fields: Vec::new(),
        });
    }
    merged
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
    structs: &'a HashMap<String, StructLayout>,
    enums: &'a HashMap<String, EnumLayout>,
    traits: &'a HashMap<String, TraitLayout>,
    trait_vtables: &'a HashMap<String, String>,
    globals: &'a mut Vec<GlobalString>,
    extra_type_decls: &'a mut Vec<String>,
    extra_function_irs: &'a mut Vec<String>,
    lambda_counter: &'a mut usize,
    closure_env: Option<ClosureEnv>,
    scopes: Vec<HashMap<String, LocalVar>>,
    lines: Vec<String>,
    temp_counter: usize,
    label_counter: usize,
    terminated: bool,
    unsafe_depth: usize,
}

impl<'a> FunctionCompiler<'a> {
    fn new(
        function_name: &'a str,
        return_type: LlvmType,
        signatures: &'a HashMap<String, FunctionSignature>,
        structs: &'a HashMap<String, StructLayout>,
        enums: &'a HashMap<String, EnumLayout>,
        traits: &'a HashMap<String, TraitLayout>,
        trait_vtables: &'a HashMap<String, String>,
        globals: &'a mut Vec<GlobalString>,
        extra_type_decls: &'a mut Vec<String>,
        extra_function_irs: &'a mut Vec<String>,
        lambda_counter: &'a mut usize,
        closure_env: Option<ClosureEnv>,
    ) -> Self {
        Self {
            function_name,
            return_type,
            signatures,
            structs,
            enums,
            traits,
            trait_vtables,
            globals,
            extra_type_decls,
            extra_function_irs,
            lambda_counter,
            closure_env,
            scopes: vec![HashMap::new()],
            lines: Vec::new(),
            temp_counter: 0,
            label_counter: 0,
            terminated: false,
            unsafe_depth: 0,
        }
    }

    fn compile(
        mut self,
        parameters: &[(String, Type)],
        body: &[Node],
    ) -> Result<Vec<String>, String> {
        let mut arg_index = 0usize;
        if let Some(env) = self.closure_env.clone() {
            for (capture_index, (name, llvm_type)) in env.captures.iter().enumerate() {
                let field_ptr = self.next_temp();
                self.emit_line(format!(
                    "{} = getelementptr inbounds %env.{}, ptr %env, i32 0, i32 {}",
                    field_ptr,
                    sanitize_name(&env.type_name),
                    capture_index
                ));
                let capture_ptr = self.next_temp();
                self.emit_line(format!(
                    "{} = load ptr, ptr {}, align 8",
                    capture_ptr, field_ptr
                ));
                self.declare_local(name.clone(), capture_ptr, llvm_type.clone());
            }
            arg_index = 1;
        }

        for (index, (name, sk_type)) in parameters.iter().enumerate() {
            let arg_name = format!("%arg{}", arg_index);
            if index == 0 && name == "self" {
                let llvm_type = llvm_type(sk_type, self.structs, self.enums, self.traits)?;
                self.declare_local(name.clone(), arg_name, llvm_type);
                arg_index += 1;
                continue;
            }
            let llvm_type = llvm_type(sk_type, self.structs, self.enums, self.traits)?;
            let ptr = self.emit_heap_alloc(llvm_type.clone(), name);
            self.emit_line(format!(
                "store {} {}, ptr {}, align {}",
                llvm_type.ir(),
                arg_name,
                ptr,
                self.align_of(&llvm_type)
            ));
            self.declare_local(name.clone(), ptr, llvm_type);
            arg_index += 1;
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
                let llvm_type = llvm_type(var_type, self.structs, self.enums, self.traits)?;
                let ptr = self.emit_heap_alloc(llvm_type.clone(), name);
                self.declare_local(name.clone(), ptr.clone(), llvm_type.clone());
                let init = match value {
                    Some(value) => self.compile_expr_with_expected(value, Some(&llvm_type))?,
                    None => self.default_value(&llvm_type),
                };
                let init = self.coerce_expr(init, &llvm_type, "variable declaration")?;
                self.emit_store(&ptr, &init);
                Ok(())
            }
            Node::Assignment { var, value, .. } => {
                let local = self.resolve_local_from_access(var)?;
                let expr = self.compile_expr_with_expected(value, Some(&local.llvm_type))?;
                let expr = self.coerce_expr(expr, &local.llvm_type, "assignment")?;
                self.emit_store(&local.ptr, &expr);
                Ok(())
            }
            Node::StructDestructure {
                struct_type,
                fields,
                value,
                ..
            } => {
                let expected = llvm_type(struct_type, self.structs, self.enums, self.traits)?;
                let value = self.compile_expr_with_expected(value, Some(&expected))?;
                let value = self.coerce_expr(value, &expected, "struct destructure")?;
                let LlvmType::Struct(struct_name) = &value.llvm_type else {
                    return Err(
                        "struct destructure expects a struct value in LLVM backend".to_string()
                    );
                };
                self.bind_struct_pattern_fields(struct_name, &value, fields)?;
                Ok(())
            }
            Node::Block { statements } => {
                self.push_scope();
                self.compile_statements(statements)?;
                self.pop_scope();
                Ok(())
            }
            Node::UnsafeBlock { statements } => {
                self.push_scope();
                self.enter_unsafe();
                let result = self.compile_statements(statements);
                self.exit_unsafe();
                self.pop_scope();
                result
            }
            Node::If {
                condition,
                body,
                else_if_blocks,
                else_block,
            } => self.compile_if(condition, body, else_if_blocks, else_block.as_deref()),
            Node::Match { value, cases } => self.compile_match(value, cases),
            Node::For {
                init,
                condition,
                update,
                body,
            } => self.compile_for(
                init.as_deref(),
                condition.as_deref(),
                update.as_deref(),
                body,
            ),
            Node::Return(value) => {
                match value {
                    Some(value) => {
                        let return_type = self.return_type.clone();
                        let expr = self.compile_expr_with_expected(value, Some(&return_type))?;
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
            Node::StaticFunctionCall { .. } => {
                let _ = self.compile_expr(node)?;
                Ok(())
            }
            Node::Access { .. } => {
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

    fn compile_match(&mut self, value: &Node, cases: &[ast::MatchCase]) -> Result<(), String> {
        let matched = self.compile_expr(value)?;
        match &matched.llvm_type {
            LlvmType::Enum(enum_name) => {
                let enum_layout = self
                    .enums
                    .get(enum_name)
                    .ok_or_else(|| format!("unknown enum `{}` in LLVM backend", enum_name))?;
                let tag = self.next_temp();
                self.emit_line(format!(
                    "{} = extractvalue {} {}, 0",
                    tag,
                    matched.llvm_type.ir(),
                    matched.value
                ));

                let after_label = self.next_label("match_end");
                let default_label = self.next_label("match_default");
                let case_labels = cases
                    .iter()
                    .map(|_| self.next_label("match_case"))
                    .collect::<Vec<_>>();
                let targets = cases
                    .iter()
                    .zip(case_labels.iter())
                    .map(|(case, label)| {
                        let ast::MatchPattern::EnumVariant { variant, .. } = &case.pattern else {
                            return Err("enum match case expected enum pattern".to_string());
                        };
                        let variant_layout = enum_layout
                            .variants
                            .iter()
                            .find(|candidate| candidate.name == *variant)
                            .ok_or_else(|| {
                                format!("unknown variant `{}` on enum `{}`", variant, enum_name)
                            })?;
                        Ok(format!("i32 {}, label %{}", variant_layout.tag, label))
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                self.emit_line(format!(
                    "switch i32 {}, label %{} [ {} ]",
                    tag,
                    default_label,
                    targets.join(" ")
                ));

                let mut all_terminated = true;
                for ((case, label), index) in cases.iter().zip(case_labels.iter()).zip(0..) {
                    self.emit_label(label);
                    self.push_scope();
                    let ast::MatchPattern::EnumVariant {
                        variant, bindings, ..
                    } = &case.pattern
                    else {
                        return Err("enum match case expected enum pattern".to_string());
                    };
                    let variant_layout = enum_layout
                        .variants
                        .iter()
                        .find(|candidate| candidate.name == *variant)
                        .ok_or_else(|| {
                            format!("unknown variant `{}` on enum `{}`", variant, enum_name)
                        })?;
                    if bindings.len() != variant_layout.payload_types.len() {
                        return Err(format!(
                            "variant `{}` binding arity does not match payload arity",
                            variant
                        ));
                    }
                    let payload_values = self.extract_enum_payloads(&matched, variant_layout)?;
                    for ((binding, payload_type), payload_value) in bindings
                        .iter()
                        .zip(variant_layout.payload_types.iter())
                        .zip(payload_values.into_iter())
                    {
                        let payload_ptr = self.emit_heap_alloc(payload_type.clone(), binding);
                        self.emit_store(&payload_ptr, &payload_value);
                        self.declare_local(binding.clone(), payload_ptr, payload_type.clone());
                    }
                    self.compile_statements(&case.body)?;
                    self.pop_scope();
                    if !self.terminated {
                        all_terminated = false;
                        self.emit_line(format!("br label %{}", after_label));
                    }
                    if index + 1 < cases.len() {
                        self.terminated = false;
                    }
                }

                self.emit_label(&default_label);
                self.emit_line("unreachable".to_string());

                if all_terminated {
                    self.terminated = true;
                } else {
                    self.emit_label(&after_label);
                    self.terminated = false;
                }
                Ok(())
            }
            LlvmType::Struct(struct_name) => {
                if cases.len() != 1 {
                    return Err(format!(
                        "match on struct `{}` expects exactly one case in LLVM backend",
                        struct_name
                    ));
                }
                let case = &cases[0];
                let ast::MatchPattern::Struct { fields, .. } = &case.pattern else {
                    return Err("struct match case expected struct pattern".to_string());
                };
                self.push_scope();
                self.bind_struct_pattern_fields(struct_name, &matched, fields)?;
                self.compile_statements(&case.body)?;
                self.pop_scope();
                Ok(())
            }
            _ => Err("match expects an enum or struct value in LLVM backend".to_string()),
        }
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
            LlvmType::Allocator
            | LlvmType::Arena
            | LlvmType::Window
            | LlvmType::TraitObject(_)
            | LlvmType::Reference { .. }
            | LlvmType::Pointer { .. } => {
                Err("cannot print a pointer-like value directly".to_string())
            }
            LlvmType::Struct(_) => Err("cannot print a struct value directly".to_string()),
            LlvmType::Enum(_) => Err("cannot print an enum value directly".to_string()),
            LlvmType::Function { .. } => Err("cannot print a function value directly".to_string()),
            LlvmType::Slice { .. } => Err("cannot print a slice value directly".to_string()),
            LlvmType::Array { .. } => Err("cannot print an array value directly".to_string()),
            LlvmType::Void => Err("cannot print a void value".to_string()),
        }
    }

    fn compile_expr(&mut self, node: &Node) -> Result<ExprValue, String> {
        self.compile_expr_with_expected(node, None)
    }

    /// Compiles an expression while optionally using an expected target type to
    /// guide lowering decisions.
    ///
    /// This is the main expression entry point used by variable initializers,
    /// assignments, returns, and struct field construction.
    fn compile_expr_with_expected(
        &mut self,
        node: &Node,
        expected: Option<&LlvmType>,
    ) -> Result<ExprValue, String> {
        if let Some(LlvmType::TraitObject(trait_name)) = expected {
            if let Some(value) =
                self.try_compile_borrowed_trait_object(node, trait_name, "trait object coercion")?
            {
                return Ok(value);
            }
        }

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
                value: {
                    let mut formatted = value.to_string();
                    if !formatted.contains('.') && !formatted.contains('e') && !formatted.contains('E') {
                        formatted.push_str(".0");
                    }
                    formatted
                },
            }),
            Node::Literal(Literal::Double(value)) => Ok(ExprValue {
                llvm_type: LlvmType::F64,
                value: {
                    let mut formatted = value.to_string();
                    if !formatted.contains('.') && !formatted.contains('e') && !formatted.contains('E') {
                        formatted.push_str(".0");
                    }
                    formatted
                },
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
                let parsed = ast::parse_string_literal(value)?;
                let global = self.global_c_string("str", &parsed);
                Ok(ExprValue {
                    llvm_type: LlvmType::PtrI8,
                    value: self.string_ptr(&global),
                })
            }
            Node::Identifier(name) => self.load_local(name),
            Node::Access { nodes } => self.compile_access_expr(nodes),
            Node::ArrayInit { elements } => {
                let expected = expected.ok_or_else(|| {
                    "LLVM backend needs a concrete array type for array literals".to_string()
                })?;
                match expected {
                    LlvmType::Slice { .. } => self.compile_slice_literal(elements, expected),
                    _ => self.compile_array_literal(elements, expected),
                }
            }
            Node::StructInitialization { _type, fields } => {
                let struct_name = match _type {
                    Type::Custom(name) => name.as_str(),
                    other => {
                        return Err(format!(
                        "LLVM backend currently requires concrete struct literal types, found `{}`",
                        ast::type_to_string(other)
                    ))
                    }
                };
                match expected {
                    Some(LlvmType::TraitObject(_)) => {
                        let inferred = LlvmType::Struct(struct_name.to_string());
                        self.compile_struct_literal(struct_name, fields, &inferred)
                    }
                    Some(expected) => self.compile_struct_literal(struct_name, fields, expected),
                    None => {
                        let inferred = LlvmType::Struct(struct_name.to_string());
                        self.compile_struct_literal(struct_name, fields, &inferred)
                    }
                }
            }
            Node::StaticFunctionCall {
                _type,
                name,
                arguments,
                ..
            } => self.compile_static_function_call(_type, name, arguments),
            Node::FunctionDeclaration {
                parameters,
                return_type,
                body,
                lambda: true,
                ..
            } => self.compile_lambda_expr(parameters, return_type, body, expected),
            Node::UnaryOp { operator, operand } => {
                match operator {
                    UnaryOperator::AddressOf | UnaryOperator::AddressOfMut => {
                        let Node::Access { nodes } = operand.as_ref() else {
                            return Err(
                                "address-of requires an addressable access expression".to_string()
                            );
                        };
                        let (ptr, llvm_type) = self.resolve_access_ptr(nodes)?;
                        if matches!(expected, Some(LlvmType::Pointer { .. })) {
                            if !self.unsafe_allowed() {
                                return Err("address-of requires an unsafe block".to_string());
                            }
                            return Ok(ExprValue {
                                llvm_type: LlvmType::Pointer {
                                    target_type: Box::new(llvm_type),
                                },
                                value: ptr,
                            });
                        }
                        Ok(ExprValue {
                            llvm_type: LlvmType::Reference {
                                target_type: Box::new(llvm_type),
                                mutable: matches!(operator, UnaryOperator::AddressOfMut),
                            },
                            value: ptr,
                        })
                    }
                    UnaryOperator::Plus => {
                        let value = self.compile_expr_with_expected(operand, expected)?;
                        Ok(value)
                    }
                    UnaryOperator::Minus => {
                        let value = self.compile_expr_with_expected(operand, expected)?;
                        if !is_numeric_llvm_type(&value.llvm_type)
                            || value.llvm_type == LlvmType::Char16
                        {
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
                            if target == LlvmType::F32 {
                                "0.0"
                            } else {
                                "0.0"
                            }
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
                        let value = self.compile_expr_with_expected(operand, expected)?;
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

    fn try_compile_borrowed_trait_object(
        &mut self,
        node: &Node,
        trait_name: &str,
        context: &str,
    ) -> Result<Option<ExprValue>, String> {
        let addressable = match node {
            Node::Identifier(name) => self
                .lookup_local(name)
                .cloned()
                .map(|local| (local.ptr, local.llvm_type)),
            Node::Access { nodes } => self.resolve_access_ptr(nodes).ok(),
            _ => None,
        };

        let Some((ptr, llvm_type)) = addressable else {
            return Ok(None);
        };

        let LlvmType::Struct(concrete_name) = llvm_type else {
            return Ok(None);
        };

        let trait_value =
            self.trait_object_from_ptr(trait_name, &concrete_name, ptr, context)?;
        Ok(Some(trait_value))
    }

    fn compile_static_function_call(
        &mut self,
        sk_type: &Type,
        name: &str,
        arguments: &[Node],
    ) -> Result<ExprValue, String> {
        if name == "size_of" || name == "align_of" {
            if !arguments.is_empty() {
                return Err(format!("{} expects no arguments", name));
            }
            let measured_type = llvm_type(sk_type, self.structs, self.enums, self.traits)?;
            let value = if name == "size_of" {
                self.size_of(&measured_type)
            } else {
                self.align_of(&measured_type)
            };
            return Ok(ExprValue {
                llvm_type: LlvmType::I32,
                value: value.to_string(),
            });
        }
        match sk_type {
            Type::Custom(memory_name) if memory_name == "Memory" => {
                if !self.unsafe_allowed() {
                    return Err(format!("Memory::{} requires an unsafe block", name));
                }
                match name {
                    "copy" => {
                        if arguments.len() != 3 {
                            return Err(
                                "Memory::copy expects dst, src, and byte count".to_string()
                            );
                        }
                        let dst = self.compile_expr(&arguments[0])?;
                        let src = self.compile_expr(&arguments[1])?;
                        let count =
                            self.compile_expr_with_expected(&arguments[2], Some(&LlvmType::I32))?;
                        let count = self.coerce_expr(count, &LlvmType::I64, "Memory::copy")?;
                        if !matches!(
                            dst.llvm_type,
                            LlvmType::Pointer { .. } | LlvmType::Reference { .. }
                        ) {
                            return Err("Memory::copy expects a pointer destination".to_string());
                        }
                        if !matches!(
                            src.llvm_type,
                            LlvmType::Pointer { .. } | LlvmType::Reference { .. }
                        ) {
                            return Err("Memory::copy expects a pointer source".to_string());
                        }
                        self.emit_line(format!(
                            "call ptr @memcpy(ptr {}, ptr {}, i64 {})",
                            dst.value, src.value, count.value
                        ));
                        return Ok(ExprValue {
                            llvm_type: LlvmType::Void,
                            value: "void".to_string(),
                        });
                    }
                    "set" => {
                        if arguments.len() != 3 {
                            return Err(
                                "Memory::set expects dst, value, and byte count".to_string()
                            );
                        }
                        let dst = self.compile_expr(&arguments[0])?;
                        let value =
                            self.compile_expr_with_expected(&arguments[1], Some(&LlvmType::I8))?;
                        let value = self.coerce_expr(value, &LlvmType::I32, "Memory::set")?;
                        let count =
                            self.compile_expr_with_expected(&arguments[2], Some(&LlvmType::I32))?;
                        let count = self.coerce_expr(count, &LlvmType::I64, "Memory::set")?;
                        if !matches!(
                            dst.llvm_type,
                            LlvmType::Pointer { .. } | LlvmType::Reference { .. }
                        ) {
                            return Err("Memory::set expects a pointer destination".to_string());
                        }
                        self.emit_line(format!(
                            "call ptr @memset(ptr {}, i32 {}, i64 {})",
                            dst.value, value.value, count.value
                        ));
                        return Ok(ExprValue {
                            llvm_type: LlvmType::Void,
                            value: "void".to_string(),
                        });
                    }
                    _ => {
                        return Err(format!(
                            "LLVM backend does not support static function call `Memory::{}` yet",
                            name
                        ))
                    }
                }
            }
            Type::Custom(color_name) if color_name == "Color" => match name {
                "black" | "white" | "red" | "green" | "blue" => {
                    if !arguments.is_empty() {
                        return Err(format!("Color::{} expects no arguments", name));
                    }
                    let value = match name {
                        "black" => 0xFF000000u32,
                        "white" => 0xFFFFFFFFu32,
                        "red" => 0xFFFF0000u32,
                        "green" => 0xFF00FF00u32,
                        "blue" => 0xFF0000FFu32,
                        _ => unreachable!(),
                    };
                    Ok(ExprValue {
                        llvm_type: LlvmType::I32,
                        value: value.to_string(),
                    })
                }
                "rgb" | "rgba" => {
                    let expected_args = if name == "rgb" { 3 } else { 4 };
                    if arguments.len() != expected_args {
                        return Err(format!(
                            "Color::{} expects {} integer argument(s)",
                            name, expected_args
                        ));
                    }
                    let mut channels = Vec::with_capacity(expected_args);
                    for argument in arguments {
                        let channel =
                            self.compile_expr_with_expected(argument, Some(&LlvmType::I32))?;
                        let channel = self.coerce_expr(channel, &LlvmType::I32, "Color channel")?;
                        let masked = self.next_temp();
                        self.emit_line(format!(
                            "{} = and i32 {}, 255",
                            masked, channel.value
                        ));
                        channels.push(masked);
                    }
                    let alpha = if name == "rgb" {
                        "255".to_string()
                    } else {
                        channels[3].clone()
                    };
                    let shifted_alpha = self.next_temp();
                    self.emit_line(format!(
                        "{} = shl i32 {}, 24",
                        shifted_alpha, alpha
                    ));
                    let shifted_red = self.next_temp();
                    self.emit_line(format!(
                        "{} = shl i32 {}, 16",
                        shifted_red, channels[0]
                    ));
                    let shifted_green = self.next_temp();
                    self.emit_line(format!(
                        "{} = shl i32 {}, 8",
                        shifted_green, channels[1]
                    ));
                    let with_red = self.next_temp();
                    self.emit_line(format!(
                        "{} = or i32 {}, {}",
                        with_red, shifted_alpha, shifted_red
                    ));
                    let with_green = self.next_temp();
                    self.emit_line(format!(
                        "{} = or i32 {}, {}",
                        with_green, with_red, shifted_green
                    ));
                    let full = self.next_temp();
                    self.emit_line(format!(
                        "{} = or i32 {}, {}",
                        full, with_green, channels[2]
                    ));
                    Ok(ExprValue {
                        llvm_type: LlvmType::I32,
                        value: full,
                    })
                }
                _ => Err(format!(
                    "LLVM backend does not support static function call `Color::{}` yet",
                    name
                )),
            },
            Type::Custom(window_name) if window_name == "Window" => {
                if name != "create" {
                    return Err(format!(
                        "LLVM backend does not support static function call `Window::{}` yet",
                        name
                    ));
                }
                if arguments.len() != 3 {
                    return Err("Window::create expects width, height, and title".to_string());
                }
                let width =
                    self.compile_expr_with_expected(&arguments[0], Some(&LlvmType::I32))?;
                let width = self.coerce_expr(width, &LlvmType::I32, "Window::create width")?;
                let height =
                    self.compile_expr_with_expected(&arguments[1], Some(&LlvmType::I32))?;
                let height =
                    self.coerce_expr(height, &LlvmType::I32, "Window::create height")?;
                let title =
                    self.compile_expr_with_expected(&arguments[2], Some(&LlvmType::PtrI8))?;
                let title = self.coerce_expr(title, &LlvmType::PtrI8, "Window::create title")?;
                let temp = self.next_temp();
                self.emit_line(format!(
                    "{} = call ptr @skunk_window_create(i32 {}, i32 {}, ptr {})",
                    temp, width.value, height.value, title.value
                ));
                Ok(ExprValue {
                    llvm_type: LlvmType::Window,
                    value: temp,
                })
            }
            Type::Custom(keyboard_name) if keyboard_name == "Keyboard" && name == "is_down" => {
                if arguments.len() != 2 {
                    return Err("Keyboard::is_down expects window and key".to_string());
                }
                let window =
                    self.compile_expr_with_expected(&arguments[0], Some(&LlvmType::Window))?;
                let window = self.coerce_expr(window, &LlvmType::Window, "Keyboard::is_down window")?;
                let key = self.compile_expr_with_expected(&arguments[1], Some(&LlvmType::Char16))?;
                let key = self.coerce_expr(key, &LlvmType::Char16, "Keyboard::is_down key")?;
                let temp = self.next_temp();
                self.emit_line(format!(
                    "{} = call i1 @skunk_keyboard_is_down(ptr {}, i16 {})",
                    temp, window.value, key.value
                ));
                Ok(ExprValue {
                    llvm_type: LlvmType::I1,
                    value: temp,
                })
            }
            Type::Pointer { target_type } if name == "cast" => {
                if !self.unsafe_allowed() {
                    return Err("pointer cast requires an unsafe block".to_string());
                }
                if arguments.len() != 1 {
                    return Err("pointer cast expects exactly one pointer argument".to_string());
                }
                let value = self.compile_expr(&arguments[0])?;
                let target_type = llvm_type(sk_type, self.structs, self.enums, self.traits)?;
                match value.llvm_type {
                    LlvmType::Pointer { .. } | LlvmType::Reference { .. } => Ok(ExprValue {
                        llvm_type: target_type,
                        value: value.value,
                    }),
                    other => Err(format!(
                        "pointer cast expects a pointer argument, found `{}`",
                        other.ir()
                    )),
                }
            }
            Type::Pointer { target_type } if name == "offset" => {
                if !self.unsafe_allowed() {
                    return Err("pointer offset requires an unsafe block".to_string());
                }
                if !matches!(ast::unwrap_const_view(target_type.as_ref()), Type::Byte) {
                    return Err("pointer offset currently requires a byte pointer type".to_string());
                }
                if arguments.len() != 2 {
                    return Err("pointer offset expects pointer and integer offset".to_string());
                }
                let base = self.compile_expr(&arguments[0])?;
                let offset =
                    self.compile_expr_with_expected(&arguments[1], Some(&LlvmType::I32))?;
                let offset = self.coerce_expr(offset, &LlvmType::I64, "pointer offset")?;
                match base.llvm_type {
                    LlvmType::Pointer { .. } | LlvmType::Reference { .. } => {
                        let temp = self.next_temp();
                        self.emit_line(format!(
                            "{} = getelementptr inbounds i8, ptr {}, i64 {}",
                            temp, base.value, offset.value
                        ));
                        Ok(ExprValue {
                            llvm_type: LlvmType::Pointer {
                                target_type: Box::new(LlvmType::I8),
                            },
                            value: temp,
                        })
                    }
                    other => Err(format!(
                        "pointer offset expects a pointer argument, found `{}`",
                        other.ir()
                    )),
                }
            }
            Type::Custom(system_name) if system_name == "System" && name == "allocator" => {
                if !arguments.is_empty() {
                    return Err("System::allocator expects no arguments".to_string());
                }
                let temp = self.next_temp();
                self.emit_line(format!("{} = call ptr @skunk_system_allocator()", temp));
                Ok(ExprValue {
                    llvm_type: LlvmType::Allocator,
                    value: temp,
                })
            }
            Type::Arena if name == "init" => {
                if arguments.len() != 1 {
                    return Err("Arena::init expects exactly one argument".to_string());
                }
                let allocator =
                    self.compile_expr_with_expected(&arguments[0], Some(&LlvmType::Allocator))?;
                let allocator = self.coerce_expr(allocator, &LlvmType::Allocator, "Arena::init")?;
                let temp = self.next_temp();
                self.emit_line(format!(
                    "{} = call ptr @skunk_arena_init(ptr {})",
                    temp, allocator.value
                ));
                Ok(ExprValue {
                    llvm_type: LlvmType::Arena,
                    value: temp,
                })
            }
            Type::Slice { .. } if name == "alloc" => {
                if arguments.len() != 2 {
                    return Err("slice alloc expects allocator and length".to_string());
                }
                let allocator =
                    self.compile_expr_with_expected(&arguments[0], Some(&LlvmType::Allocator))?;
                let allocator =
                    self.coerce_expr(allocator, &LlvmType::Allocator, "slice alloc allocator")?;
                let len = self.compile_expr_with_expected(&arguments[1], Some(&LlvmType::I32))?;
                let len = self.coerce_expr(len, &LlvmType::I32, "slice alloc length")?;
                let llvm_slice_type = llvm_type(sk_type, self.structs, self.enums, self.traits)?;
                let elem_type = match &llvm_slice_type {
                    LlvmType::Slice { elem_type } => elem_type.as_ref().clone(),
                    _ => unreachable!(),
                };
                let elem_size = self.size_of(&elem_type);
                let data_ptr = self.next_temp();
                self.emit_line(format!(
                    "{} = call ptr @skunk_alloc_buffer(ptr {}, i64 {}, i32 {})",
                    data_ptr, allocator.value, elem_size, len.value
                ));
                let with_ptr = self.next_temp();
                self.emit_line(format!(
                    "{} = insertvalue {} zeroinitializer, ptr {}, 0",
                    with_ptr,
                    llvm_slice_type.ir(),
                    data_ptr
                ));
                let full = self.next_temp();
                self.emit_line(format!(
                    "{} = insertvalue {} {}, i32 {}, 1",
                    full,
                    llvm_slice_type.ir(),
                    with_ptr,
                    len.value
                ));
                Ok(ExprValue {
                    llvm_type: llvm_slice_type,
                    value: full,
                })
            }
            Type::Array { .. } if name == "fill" || name == "new" => {
                if arguments.len() != 1 {
                    return Err(format!(
                        "array `{}` expects exactly one argument in LLVM backend",
                        name
                    ));
                }
                let llvm_type = llvm_type(sk_type, self.structs, self.enums, self.traits)?;
                self.compile_array_fill(&llvm_type, &arguments[0])
            }
            Type::Custom(enum_name) if self.enums.contains_key(enum_name) => {
                self.compile_enum_constructor(enum_name, name, arguments)
            }
            Type::Custom(struct_name) if name != "create" && self.structs.contains_key(struct_name) => {
                let signature_key = format!("{}::{}", struct_name, name);
                let signature = self
                    .signatures
                    .get(&signature_key)
                    .cloned()
                    .ok_or_else(|| {
                        format!(
                            "unknown static function `{}` on `{}` in LLVM backend",
                            name, struct_name
                        )
                    })?;
                self.compile_direct_call(&signature_key, &signature, arguments, "static function")
            }
            Type::Custom(_)
            | Type::Byte
            | Type::Short
            | Type::Int
            | Type::Long
            | Type::Float
            | Type::Double
            | Type::Boolean
            | Type::Char
            | Type::String
            | Type::Pointer { .. }
                if name == "create" =>
            {
                if arguments.len() != 1 {
                    return Err(format!(
                        "{}::create expects exactly one allocator argument",
                        ast::type_to_string(sk_type)
                    ));
                }
                let allocator =
                    self.compile_expr_with_expected(&arguments[0], Some(&LlvmType::Allocator))?;
                let allocator = self.coerce_expr(allocator, &LlvmType::Allocator, "type create")?;
                let target_type = llvm_type(sk_type, self.structs, self.enums, self.traits)?;
                let temp = self.next_temp();
                self.emit_line(format!(
                    "{} = call ptr @skunk_alloc_create(ptr {}, i64 {})",
                    temp,
                    allocator.value,
                    self.size_of(&target_type)
                ));
                Ok(ExprValue {
                    llvm_type: LlvmType::Pointer {
                        target_type: Box::new(target_type),
                    },
                    value: temp,
                })
            }
            Type::Array { .. } => Err(format!(
                "array static method `{}` is not supported in LLVM backend",
                name
            )),
            _ => Err(format!(
                "LLVM backend does not support static function call `{}::{}` yet",
                ast::type_to_string(sk_type),
                name
            )),
        }
    }

    fn compile_enum_constructor(
        &mut self,
        enum_name: &str,
        variant_name: &str,
        arguments: &[Node],
    ) -> Result<ExprValue, String> {
        let enum_layout = self
            .enums
            .get(enum_name)
            .ok_or_else(|| format!("unknown enum `{}` in LLVM backend", enum_name))?;
        let variant = enum_layout
            .variants
            .iter()
            .find(|candidate| candidate.name == variant_name)
            .ok_or_else(|| format!("unknown variant `{}` on enum `{}`", variant_name, enum_name))?;
        let llvm_enum_type = LlvmType::Enum(enum_name.to_string());
        let with_tag = self.next_temp();
        self.emit_line(format!(
            "{} = insertvalue {} zeroinitializer, i32 {}, 0",
            with_tag,
            llvm_enum_type.ir(),
            variant.tag
        ));
        if arguments.len() != variant.payload_types.len() {
            return Err(format!(
                "enum variant `{}::{}` expects {} argument(s), got {}",
                enum_name,
                variant_name,
                variant.payload_types.len(),
                arguments.len()
            ));
        }
        let mut current = with_tag;
        for ((argument, payload_type), field_index) in arguments
            .iter()
            .zip(variant.payload_types.iter())
            .zip(variant.field_indices.iter())
        {
            let payload = self.compile_expr_with_expected(argument, Some(payload_type))?;
            let payload = self.coerce_expr(payload, payload_type, "enum variant payload")?;
            let next = self.next_temp();
            self.emit_line(format!(
                "{} = insertvalue {} {}, {} {}, {}",
                next,
                llvm_enum_type.ir(),
                current,
                payload_type.ir(),
                payload.value,
                field_index
            ));
            current = next;
        }
        Ok(ExprValue {
            llvm_type: llvm_enum_type,
            value: current,
        })
    }

    fn compile_array_literal(
        &mut self,
        elements: &[Node],
        expected: &LlvmType,
    ) -> Result<ExprValue, String> {
        match expected {
            LlvmType::Array { elem_type, len } => {
                if elements.len() != *len {
                    return Err(format!(
                        "array literal expected {} elements, got {}",
                        len,
                        elements.len()
                    ));
                }
                let mut aggregate = ExprValue {
                    llvm_type: expected.clone(),
                    value: "zeroinitializer".to_string(),
                };
                for (index, element) in elements.iter().enumerate() {
                    let element =
                        self.compile_expr_with_expected(element, Some(elem_type.as_ref()))?;
                    let element =
                        self.coerce_expr(element, elem_type.as_ref(), "array literal element")?;
                    let temp = self.next_temp();
                    self.emit_line(format!(
                        "{} = insertvalue {} {}, {} {}, {}",
                        temp,
                        expected.ir(),
                        aggregate.value,
                        element.llvm_type.ir(),
                        element.value,
                        index
                    ));
                    aggregate = ExprValue {
                        llvm_type: expected.clone(),
                        value: temp,
                    };
                }
                Ok(aggregate)
            }
            other => Err(format!(
                "LLVM backend cannot use an array literal to initialize `{}`",
                other.ir()
            )),
        }
    }

    fn compile_array_fill(
        &mut self,
        expected: &LlvmType,
        fill_node: &Node,
    ) -> Result<ExprValue, String> {
        match expected {
            LlvmType::Array { elem_type, len } => {
                let mut aggregate = ExprValue {
                    llvm_type: expected.clone(),
                    value: "zeroinitializer".to_string(),
                };
                for index in 0..*len {
                    let element = self.compile_array_fill(elem_type.as_ref(), fill_node)?;
                    let temp = self.next_temp();
                    self.emit_line(format!(
                        "{} = insertvalue {} {}, {} {}, {}",
                        temp,
                        expected.ir(),
                        aggregate.value,
                        element.llvm_type.ir(),
                        element.value,
                        index
                    ));
                    aggregate = ExprValue {
                        llvm_type: expected.clone(),
                        value: temp,
                    };
                }
                Ok(aggregate)
            }
            _ => {
                let element = self.compile_expr_with_expected(fill_node, Some(expected))?;
                self.coerce_expr(element, expected, "array fill")
            }
        }
    }

    fn compile_slice_literal(
        &mut self,
        elements: &[Node],
        expected: &LlvmType,
    ) -> Result<ExprValue, String> {
        let elem_type = match expected {
            LlvmType::Slice { elem_type } => elem_type.as_ref().clone(),
            other => {
                return Err(format!(
                    "LLVM backend cannot use a slice literal to initialize `{}`",
                    other.ir()
                ))
            }
        };

        let backing_type = LlvmType::Array {
            elem_type: Box::new(elem_type),
            len: elements.len(),
        };
        let aggregate = self.compile_array_literal(elements, &backing_type)?;
        let backing_ptr = self.emit_heap_alloc(backing_type.clone(), "slice_lit");
        self.emit_store(&backing_ptr, &aggregate);
        self.build_slice_from_array_ptr(&backing_ptr, &backing_type, None, None)
    }

    fn compile_slice_from_ptr(
        &mut self,
        ptr: &str,
        current_type: &LlvmType,
        start: Option<&Node>,
        end: Option<&Node>,
    ) -> Result<ExprValue, String> {
        match current_type {
            LlvmType::Array { .. } => {
                self.build_slice_from_array_ptr(ptr, current_type, start, end)
            }
            LlvmType::Slice { elem_type } => {
                let slice_value = self.load_from_ptr(ptr, current_type)?;
                let data_ptr = self.extract_slice_data(&slice_value)?;
                let len_value = self.extract_slice_len(&slice_value)?;
                self.build_slice_header_from_data_ptr(
                    &data_ptr,
                    elem_type.as_ref(),
                    &len_value,
                    start,
                    end,
                )
            }
            other => Err(format!(
                "cannot take a slice of `{}` in LLVM backend",
                other.ir()
            )),
        }
    }

    fn build_slice_from_array_ptr(
        &mut self,
        array_ptr: &str,
        array_type: &LlvmType,
        start: Option<&Node>,
        end: Option<&Node>,
    ) -> Result<ExprValue, String> {
        let (elem_type, len) = match array_type {
            LlvmType::Array { elem_type, len } => (elem_type.as_ref(), *len),
            other => {
                return Err(format!(
                    "expected array storage for slice construction, found `{}`",
                    other.ir()
                ))
            }
        };
        let len_value = ExprValue {
            llvm_type: LlvmType::I32,
            value: len.to_string(),
        };
        let data_ptr = if len == 0 {
            "null".to_string()
        } else {
            let data_ptr = self.next_temp();
            self.emit_line(format!(
                "{} = getelementptr inbounds {}, ptr {}, i64 0, i64 0",
                data_ptr,
                array_type.ir(),
                array_ptr
            ));
            data_ptr
        };
        self.build_slice_header_from_data_ptr(&data_ptr, elem_type, &len_value, start, end)
    }

    fn build_slice_header_from_data_ptr(
        &mut self,
        data_ptr: &str,
        elem_type: &LlvmType,
        base_len: &ExprValue,
        start: Option<&Node>,
        end: Option<&Node>,
    ) -> Result<ExprValue, String> {
        let start_value = match start {
            Some(node) => {
                let value = self.compile_expr_with_expected(node, Some(&LlvmType::I32))?;
                self.coerce_expr(value, &LlvmType::I32, "slice start")?
            }
            None => ExprValue {
                llvm_type: LlvmType::I32,
                value: "0".to_string(),
            },
        };
        let end_value = match end {
            Some(node) => {
                let value = self.compile_expr_with_expected(node, Some(&LlvmType::I32))?;
                self.coerce_expr(value, &LlvmType::I32, "slice end")?
            }
            None => base_len.clone(),
        };

        let start_i64 = self.coerce_expr(start_value.clone(), &LlvmType::I64, "slice start")?;
        let offset_ptr = if data_ptr == "null" {
            "null".to_string()
        } else {
            let offset_ptr = self.next_temp();
            self.emit_line(format!(
                "{} = getelementptr inbounds {}, ptr {}, i64 {}",
                offset_ptr,
                elem_type.ir(),
                data_ptr,
                start_i64.value
            ));
            offset_ptr
        };

        let slice_len = self.next_temp();
        self.emit_line(format!(
            "{} = sub i32 {}, {}",
            slice_len, end_value.value, start_value.value
        ));

        let slice_type = LlvmType::Slice {
            elem_type: Box::new(elem_type.clone()),
        };
        let with_ptr = self.next_temp();
        self.emit_line(format!(
            "{} = insertvalue {} zeroinitializer, ptr {}, 0",
            with_ptr,
            slice_type.ir(),
            offset_ptr
        ));
        let full = self.next_temp();
        self.emit_line(format!(
            "{} = insertvalue {} {}, i32 {}, 1",
            full,
            slice_type.ir(),
            with_ptr,
            slice_len
        ));
        Ok(ExprValue {
            llvm_type: slice_type,
            value: full,
        })
    }

    fn load_from_ptr(&mut self, ptr: &str, llvm_type: &LlvmType) -> Result<ExprValue, String> {
        let temp = self.next_temp();
        self.emit_line(format!(
            "{} = load {}, ptr {}, align {}",
            temp,
            llvm_type.ir(),
            ptr,
            self.align_of(llvm_type)
        ));
        Ok(ExprValue {
            llvm_type: llvm_type.clone(),
            value: temp,
        })
    }

    fn extract_slice_data(&mut self, slice: &ExprValue) -> Result<String, String> {
        match &slice.llvm_type {
            LlvmType::Slice { .. } => {
                let data_ptr = self.next_temp();
                self.emit_line(format!(
                    "{} = extractvalue {} {}, 0",
                    data_ptr,
                    slice.llvm_type.ir(),
                    slice.value
                ));
                Ok(data_ptr)
            }
            other => Err(format!("expected slice value, found `{}`", other.ir())),
        }
    }

    fn extract_slice_len(&mut self, slice: &ExprValue) -> Result<ExprValue, String> {
        match &slice.llvm_type {
            LlvmType::Slice { .. } => {
                let len = self.next_temp();
                self.emit_line(format!(
                    "{} = extractvalue {} {}, 1",
                    len,
                    slice.llvm_type.ir(),
                    slice.value
                ));
                Ok(ExprValue {
                    llvm_type: LlvmType::I32,
                    value: len,
                })
            }
            other => Err(format!("expected slice value, found `{}`", other.ir())),
        }
    }

    /// Lowers a typed struct literal into an LLVM aggregate value.
    ///
    /// Field expressions are compiled one by one and inserted into a
    /// zero-initialized aggregate using the resolved struct layout.
    fn compile_struct_literal(
        &mut self,
        struct_name: &str,
        fields: &[(String, Node)],
        expected: &LlvmType,
    ) -> Result<ExprValue, String> {
        let expected_name = match expected {
            LlvmType::Struct(name) => name,
            other => {
                return Err(format!(
                    "LLVM backend cannot use struct literal `{}` to initialize `{}`",
                    struct_name,
                    other.ir()
                ))
            }
        };
        if expected_name != struct_name {
            return Err(format!(
                "struct literal `{}` cannot initialize `{}`",
                struct_name, expected_name
            ));
        }

        let layout = self
            .structs
            .get(struct_name)
            .ok_or_else(|| format!("unknown struct `{}` in LLVM backend", struct_name))?;

        let mut aggregate = ExprValue {
            llvm_type: expected.clone(),
            value: "zeroinitializer".to_string(),
        };

        for (field_name, field_node) in fields {
            let (index, field_type) =
                self.struct_field_info(struct_name, field_name)
                    .ok_or_else(|| {
                        format!("unknown field `{}` on struct `{}`", field_name, struct_name)
                    })?;
            let field_value = self.compile_expr_with_expected(field_node, Some(&field_type))?;
            let field_value = self.coerce_expr(field_value, &field_type, "struct field")?;
            let temp = self.next_temp();
            self.emit_line(format!(
                "{} = insertvalue {} {}, {} {}, {}",
                temp,
                expected.ir(),
                aggregate.value,
                field_value.llvm_type.ir(),
                field_value.value,
                index
            ));
            aggregate = ExprValue {
                llvm_type: expected.clone(),
                value: temp,
            };
        }

        if fields.len() > layout.fields.len() {
            return Err(format!(
                "too many fields provided when initializing `{}`",
                struct_name
            ));
        }

        Ok(aggregate)
    }

    fn compile_access_expr(&mut self, nodes: &[Node]) -> Result<ExprValue, String> {
        if let Some(expr) = self.try_compile_member_call(nodes)? {
            return Ok(expr);
        }
        if let Some(expr) = self.compile_length_member(nodes)? {
            return Ok(expr);
        }
        let (ptr, llvm_type) = self.resolve_access_ptr(nodes)?;
        self.load_from_ptr(&ptr, &llvm_type)
    }

    fn try_compile_member_call(&mut self, nodes: &[Node]) -> Result<Option<ExprValue>, String> {
        let (receiver_nodes, method_name, arguments) = match nodes.last() {
            Some(Node::MemberAccess { member, .. }) => match member.as_ref() {
                Node::FunctionCall {
                    name, arguments, ..
                } => (&nodes[..nodes.len() - 1], name.as_str(), arguments),
                _ => return Ok(None),
            },
            _ => return Ok(None),
        };

        let receiver_type = if let Ok((_, llvm_type)) = self.resolve_access_ptr(receiver_nodes) {
            llvm_type
        } else {
            self.compile_expr(&Node::Access {
                nodes: receiver_nodes.to_vec(),
            })?
            .llvm_type
        };

        if let LlvmType::TraitObject(trait_name) = receiver_type {
            let trait_layout = self
                .traits
                .get(&trait_name)
                .ok_or_else(|| format!("unknown trait `{}` in LLVM backend", trait_name))?;
            let (method_index, method) = trait_layout
                .methods
                .iter()
                .enumerate()
                .find(|(_, method)| method.name == method_name)
                .ok_or_else(|| {
                    format!(
                        "unknown method `{}` on trait object `{}`",
                        method_name, trait_name
                    )
                })?;

            let first_args = arguments
                .first()
                .ok_or_else(|| "method call is missing its first argument group".to_string())?;
            if first_args.len() != method.parameters.len() {
                return Err(format!(
                    "method `{}` expects {} arguments, got {}",
                    method_name,
                    method.parameters.len(),
                    first_args.len()
                ));
            }

            let receiver_value = self.compile_expr(&Node::Access {
                nodes: receiver_nodes.to_vec(),
            })?;
            let data_ptr = self.next_temp();
            self.emit_line(format!(
                "{} = extractvalue {} {}, 0",
                data_ptr,
                receiver_value.llvm_type.ir(),
                receiver_value.value
            ));
            let vtable_ptr = self.next_temp();
            self.emit_line(format!(
                "{} = extractvalue {} {}, 1",
                vtable_ptr,
                receiver_value.llvm_type.ir(),
                receiver_value.value
            ));
            let slot_ptr = self.next_temp();
            self.emit_line(format!(
                "{} = getelementptr inbounds %vtable.{}, ptr {}, i32 0, i32 {}",
                slot_ptr,
                sanitize_name(&trait_name),
                vtable_ptr,
                method_index
            ));
            let fn_ptr = self.next_temp();
            self.emit_line(format!("{} = load ptr, ptr {}, align 8", fn_ptr, slot_ptr));

            let mut arg_parts = vec![format!("ptr {}", data_ptr)];
            for (arg_node, expected_type) in first_args.iter().zip(method.parameters.iter()) {
                let arg = self.compile_expr_with_expected(arg_node, Some(expected_type))?;
                let arg = self.coerce_expr(arg, expected_type, "trait method argument")?;
                arg_parts.push(format!("{} {}", arg.llvm_type.ir(), arg.value));
            }

            let mut current = if method.return_type == LlvmType::Void {
                self.emit_line(format!("call void {}({})", fn_ptr, arg_parts.join(", ")));
                ExprValue {
                    llvm_type: LlvmType::Void,
                    value: "void".to_string(),
                }
            } else {
                let temp = self.next_temp();
                self.emit_line(format!(
                    "{} = call {} {}({})",
                    temp,
                    method.return_type.ir(),
                    fn_ptr,
                    arg_parts.join(", ")
                ));
                ExprValue {
                    llvm_type: method.return_type.clone(),
                    value: temp,
                }
            };

            for arg_group in arguments.iter().skip(1) {
                current = self.compile_closure_call(current, arg_group, "trait method call")?;
            }

            return Ok(Some(current));
        }

        if receiver_type == LlvmType::Allocator {
            if arguments.len() != 1 {
                return Err(format!(
                    "Allocator method `{}` expects exactly one argument list",
                    method_name
                ));
            }
            let method_args = &arguments[0];
            let allocator_value = self.compile_expr(&Node::Access {
                nodes: receiver_nodes.to_vec(),
            })?;
            return match method_name {
                "destroy" => {
                    if method_args.len() != 1 {
                        return Err(
                            "Allocator.destroy expects exactly one pointer argument".to_string()
                        );
                    }
                    let ptr_value = self.compile_expr(&method_args[0])?;
                    match ptr_value.llvm_type {
                        LlvmType::Pointer { .. } => {
                            self.emit_line(format!(
                                "call void @skunk_alloc_destroy(ptr {}, ptr {})",
                                allocator_value.value, ptr_value.value
                            ));
                            Ok(Some(ExprValue {
                                llvm_type: LlvmType::Void,
                                value: "void".to_string(),
                            }))
                        }
                        other => Err(format!(
                            "Allocator.destroy expects a pointer argument, found `{}`",
                            other.ir()
                        )),
                    }
                }
                "free" => {
                    if method_args.len() != 1 {
                        return Err("Allocator.free expects exactly one slice argument".to_string());
                    }
                    let slice_value = self.compile_expr(&method_args[0])?;
                    match slice_value.llvm_type {
                        LlvmType::Slice { .. } => {
                            let data_ptr = self.extract_slice_data(&slice_value)?;
                            self.emit_line(format!(
                                "call void @skunk_alloc_free(ptr {}, ptr {})",
                                allocator_value.value, data_ptr
                            ));
                            Ok(Some(ExprValue {
                                llvm_type: LlvmType::Void,
                                value: "void".to_string(),
                            }))
                        }
                        other => Err(format!(
                            "Allocator.free expects a slice argument, found `{}`",
                            other.ir()
                        )),
                    }
                }
                _ => Err(format!("unknown Allocator method `{}`", method_name)),
            };
        }

        if receiver_type == LlvmType::Arena {
            if arguments.len() != 1 || !arguments[0].is_empty() {
                return Err(format!(
                    "Arena method `{}` expects no arguments",
                    method_name
                ));
            }
            let arena_value = self.compile_expr(&Node::Access {
                nodes: receiver_nodes.to_vec(),
            })?;
            return match method_name {
                "allocator" => {
                    let temp = self.next_temp();
                    self.emit_line(format!(
                        "{} = call ptr @skunk_arena_allocator(ptr {})",
                        temp, arena_value.value
                    ));
                    Ok(Some(ExprValue {
                        llvm_type: LlvmType::Allocator,
                        value: temp,
                    }))
                }
                "reset" => {
                    self.emit_line(format!(
                        "call void @skunk_arena_reset(ptr {})",
                        arena_value.value
                    ));
                    Ok(Some(ExprValue {
                        llvm_type: LlvmType::Void,
                        value: "void".to_string(),
                    }))
                }
                "deinit" => {
                    self.emit_line(format!(
                        "call void @skunk_arena_deinit(ptr {})",
                        arena_value.value
                    ));
                    Ok(Some(ExprValue {
                        llvm_type: LlvmType::Void,
                        value: "void".to_string(),
                    }))
                }
                _ => Err(format!("unknown Arena method `{}`", method_name)),
            };
        }

        if receiver_type == LlvmType::Window {
            if arguments.len() != 1 {
                return Err(format!(
                    "Window method `{}` expects exactly one argument list",
                    method_name
                ));
            }
            let method_args = &arguments[0];
            let window_value = self.compile_expr(&Node::Access {
                nodes: receiver_nodes.to_vec(),
            })?;
            return match method_name {
                "is_open" => {
                    if !method_args.is_empty() {
                        return Err("Window.is_open expects no arguments".to_string());
                    }
                    let temp = self.next_temp();
                    self.emit_line(format!(
                        "{} = call i1 @skunk_window_is_open(ptr {})",
                        temp, window_value.value
                    ));
                    Ok(Some(ExprValue {
                        llvm_type: LlvmType::I1,
                        value: temp,
                    }))
                }
                "poll" => {
                    if !method_args.is_empty() {
                        return Err("Window.poll expects no arguments".to_string());
                    }
                    self.emit_line(format!(
                        "call void @skunk_window_poll(ptr {})",
                        window_value.value
                    ));
                    Ok(Some(ExprValue {
                        llvm_type: LlvmType::Void,
                        value: "void".to_string(),
                    }))
                }
                "clear" => {
                    if method_args.len() != 1 {
                        return Err("Window.clear expects one color argument".to_string());
                    }
                    let color =
                        self.compile_expr_with_expected(&method_args[0], Some(&LlvmType::I32))?;
                    let color = self.coerce_expr(color, &LlvmType::I32, "Window.clear color")?;
                    self.emit_line(format!(
                        "call void @skunk_window_clear(ptr {}, i32 {})",
                        window_value.value, color.value
                    ));
                    Ok(Some(ExprValue {
                        llvm_type: LlvmType::Void,
                        value: "void".to_string(),
                    }))
                }
                "draw_rect" => {
                    if method_args.len() != 5 {
                        return Err(
                            "Window.draw_rect expects x, y, width, height, and color".to_string()
                        );
                    }
                    let x =
                        self.compile_expr_with_expected(&method_args[0], Some(&LlvmType::F64))?;
                    let x = self.coerce_expr(x, &LlvmType::F64, "Window.draw_rect x")?;
                    let y =
                        self.compile_expr_with_expected(&method_args[1], Some(&LlvmType::F64))?;
                    let y = self.coerce_expr(y, &LlvmType::F64, "Window.draw_rect y")?;
                    let width =
                        self.compile_expr_with_expected(&method_args[2], Some(&LlvmType::F64))?;
                    let width =
                        self.coerce_expr(width, &LlvmType::F64, "Window.draw_rect width")?;
                    let height =
                        self.compile_expr_with_expected(&method_args[3], Some(&LlvmType::F64))?;
                    let height =
                        self.coerce_expr(height, &LlvmType::F64, "Window.draw_rect height")?;
                    let color =
                        self.compile_expr_with_expected(&method_args[4], Some(&LlvmType::I32))?;
                    let color =
                        self.coerce_expr(color, &LlvmType::I32, "Window.draw_rect color")?;
                    self.emit_line(format!(
                        "call void @skunk_window_draw_rect(ptr {}, double {}, double {}, double {}, double {}, i32 {})",
                        window_value.value, x.value, y.value, width.value, height.value, color.value
                    ));
                    Ok(Some(ExprValue {
                        llvm_type: LlvmType::Void,
                        value: "void".to_string(),
                    }))
                }
                "present" => {
                    if !method_args.is_empty() {
                        return Err("Window.present expects no arguments".to_string());
                    }
                    self.emit_line(format!(
                        "call void @skunk_window_present(ptr {})",
                        window_value.value
                    ));
                    Ok(Some(ExprValue {
                        llvm_type: LlvmType::Void,
                        value: "void".to_string(),
                    }))
                }
                "delta_time" => {
                    if !method_args.is_empty() {
                        return Err("Window.delta_time expects no arguments".to_string());
                    }
                    let temp = self.next_temp();
                    self.emit_line(format!(
                        "{} = call double @skunk_window_delta_time(ptr {})",
                        temp, window_value.value
                    ));
                    Ok(Some(ExprValue {
                        llvm_type: LlvmType::F64,
                        value: temp,
                    }))
                }
                "close" => {
                    if !method_args.is_empty() {
                        return Err("Window.close expects no arguments".to_string());
                    }
                    self.emit_line(format!(
                        "call void @skunk_window_close(ptr {})",
                        window_value.value
                    ));
                    Ok(Some(ExprValue {
                        llvm_type: LlvmType::Void,
                        value: "void".to_string(),
                    }))
                }
                "deinit" => {
                    if !method_args.is_empty() {
                        return Err("Window.deinit expects no arguments".to_string());
                    }
                    self.emit_line(format!(
                        "call void @skunk_window_deinit(ptr {})",
                        window_value.value
                    ));
                    Ok(Some(ExprValue {
                        llvm_type: LlvmType::Void,
                        value: "void".to_string(),
                    }))
                }
                _ => Err(format!("unknown Window method `{}`", method_name)),
            };
        }

        let struct_name = match receiver_type {
            LlvmType::Struct(name) => name,
            LlvmType::Reference { target_type, .. } => match *target_type {
                LlvmType::Struct(name) => name,
                other => {
                    return Err(format!(
                        "method `{}` requires a struct receiver, found `{}`",
                        method_name,
                        other.ir()
                    ))
                }
            },
            LlvmType::Pointer { target_type } => match *target_type {
                LlvmType::Struct(name) => name,
                other => {
                    return Err(format!(
                        "method `{}` requires a struct receiver, found `{}`",
                        method_name,
                        other.ir()
                    ))
                }
            },
            other => {
                return Err(format!(
                    "method `{}` requires a struct receiver, found `{}`",
                    method_name,
                    other.ir()
                ))
            }
        };

        let signature_key = format!("{}::{}", struct_name, method_name);
        let signature = self
            .signatures
            .get(&signature_key)
            .cloned()
            .ok_or_else(|| format!("unknown method `{}` on `{}`", method_name, struct_name))?;

        let receiver_ptr = match self.resolve_access_ptr(receiver_nodes) {
            Ok((ptr, llvm_type)) => {
                if is_pointer_like_llvm_type(&llvm_type) {
                    self.load_from_ptr(&ptr, &llvm_type)?.value
                } else {
                    ptr
                }
            }
            Err(_) => {
                let temp_var =
                    self.emit_heap_alloc(LlvmType::Struct(struct_name.clone()), "receiver_tmp");
                let receiver_expr = self.compile_expr(&Node::Access {
                    nodes: receiver_nodes.to_vec(),
                })?;
                self.emit_store(&temp_var, &receiver_expr);
                temp_var
            }
        };

        let first_args = arguments
            .first()
            .ok_or_else(|| "method call is missing its first argument group".to_string())?;
        if first_args.len() != signature.parameters.len() {
            return Err(format!(
                "method `{}` expects {} arguments, got {}",
                method_name,
                signature.parameters.len(),
                first_args.len()
            ));
        }
        let mut arg_parts = vec![format!("ptr {}", receiver_ptr)];
        for (arg_node, expected_type) in first_args.iter().zip(signature.parameters.iter()) {
            let arg = self.compile_expr_with_expected(arg_node, Some(expected_type))?;
            let arg = self.coerce_expr(arg, expected_type, "method argument")?;
            arg_parts.push(format!("{} {}", arg.llvm_type.ir(), arg.value));
        }

        let mut current = if signature.return_type == LlvmType::Void {
            self.emit_line(format!(
                "call void @{}({})",
                signature.symbol_name,
                arg_parts.join(", ")
            ));
            ExprValue {
                llvm_type: LlvmType::Void,
                value: "void".to_string(),
            }
        } else {
            let temp = self.next_temp();
            self.emit_line(format!(
                "{} = call {} @{}({})",
                temp,
                signature.return_type.ir(),
                signature.symbol_name,
                arg_parts.join(", ")
            ));
            ExprValue {
                llvm_type: signature.return_type,
                value: temp,
            }
        };

        for arg_group in arguments.iter().skip(1) {
            current = self.compile_closure_call(current, arg_group, "method call")?;
        }

        Ok(Some(current))
    }

    fn compile_length_member(&mut self, nodes: &[Node]) -> Result<Option<ExprValue>, String> {
        if let Some(Node::MemberAccess { member, .. }) = nodes.last() {
            if let Node::Identifier(name) = member.as_ref() {
                if name == "len" {
                    let mut current = self.access_base_type(nodes)?;
                    for node in &nodes[1..nodes.len() - 1] {
                        current = self.apply_access_type_step(current, node)?;
                    }
                    return match current {
                        LlvmType::Array { len, .. } => Ok(Some(ExprValue {
                            llvm_type: LlvmType::I32,
                            value: len.to_string(),
                        })),
                        LlvmType::Slice { .. } => {
                            let (ptr, llvm_type) = self.resolve_access_ptr(&nodes[..nodes.len() - 1])?;
                            let slice_value = self.load_from_ptr(&ptr, &llvm_type)?;
                            Ok(Some(self.extract_slice_len(&slice_value)?))
                        }
                        other => Err(format!(
                            "member `len` is only available on arrays and slices in LLVM backend, found `{}`",
                            other.ir()
                        )),
                    };
                }
            }
        }
        Ok(None)
    }

    fn access_base_type(&self, nodes: &[Node]) -> Result<LlvmType, String> {
        match nodes.first() {
            Some(Node::Identifier(name)) => self
                .lookup_local(name)
                .map(|local| local.llvm_type.clone())
                .ok_or_else(|| format!("unknown variable `{}` in LLVM backend", name)),
            _ => Err("LLVM backend currently supports only identifier-rooted access".to_string()),
        }
    }

    fn apply_access_type_step(&self, current: LlvmType, node: &Node) -> Result<LlvmType, String> {
        let current = match (&current, node) {
            (LlvmType::Reference { target_type, .. }, Node::MemberAccess { .. })
            | (LlvmType::Reference { target_type, .. }, Node::ArrayAccess { .. })
            | (LlvmType::Reference { target_type, .. }, Node::SliceAccess { .. }) => {
                target_type.as_ref().clone()
            }
            (LlvmType::Pointer { target_type }, Node::MemberAccess { .. })
            | (LlvmType::Pointer { target_type }, Node::ArrayAccess { .. })
            | (LlvmType::Pointer { target_type }, Node::SliceAccess { .. }) => {
                target_type.as_ref().clone()
            }
            _ => current,
        };
        match node {
            Node::Dereference { .. } => match current {
                LlvmType::Reference { target_type, .. } => Ok(*target_type),
                LlvmType::Pointer { target_type } => Ok(*target_type),
                other => Err(format!(
                    "cannot dereference non-pointer type `{}` in LLVM backend",
                    other.ir()
                )),
            },
            Node::ArrayAccess { coordinates } => {
                let mut current = current;
                for _ in coordinates {
                    current = match current {
                        LlvmType::Array { elem_type, .. } => *elem_type,
                        LlvmType::Slice { elem_type } => *elem_type,
                        other => {
                            return Err(format!(
                                "cannot index non-array type `{}` in LLVM backend",
                                other.ir()
                            ))
                        }
                    };
                }
                Ok(current)
            }
            Node::SliceAccess { .. } => match current {
                LlvmType::Array { elem_type, .. } => Ok(LlvmType::Slice { elem_type }),
                LlvmType::Slice { elem_type } => Ok(LlvmType::Slice { elem_type }),
                other => Err(format!(
                    "cannot take a slice of `{}` in LLVM backend",
                    other.ir()
                )),
            },
            Node::MemberAccess { member, .. } => match member.as_ref() {
                Node::Identifier(name) if name == "len" => Ok(current),
                Node::Identifier(name) => match current {
                    LlvmType::Struct(struct_name) => self
                        .struct_field_info(&struct_name, name)
                        .map(|(_, field_type)| field_type)
                        .ok_or_else(|| {
                            format!(
                                "unknown field `{}` on struct `{}` in LLVM backend",
                                name, struct_name
                            )
                        }),
                    other => Err(format!(
                        "member `{}` is not available on `{}` in LLVM backend",
                        name,
                        other.ir()
                    )),
                },
                Node::FunctionCall { .. } => Ok(current),
                other => Err(format!(
                    "LLVM backend does not support member access `{:?}` yet",
                    other
                )),
            },
            other => Err(format!(
                "LLVM backend does not support access step `{:?}` yet",
                other
            )),
        }
    }

    fn resolve_access_ptr(&mut self, nodes: &[Node]) -> Result<(String, LlvmType), String> {
        let (mut ptr, mut current_type) = match nodes.first() {
            Some(Node::Identifier(name)) => {
                let local = self
                    .lookup_local(name)
                    .cloned()
                    .ok_or_else(|| format!("unknown variable `{}` in LLVM backend", name))?;
                (local.ptr, local.llvm_type)
            }
            _ => {
                return Err(
                    "LLVM backend currently supports only identifier-rooted access".to_string(),
                )
            }
        };

        for node in &nodes[1..] {
            if matches!(
                (&current_type, node),
                (LlvmType::Reference { .. }, Node::MemberAccess { .. })
                    | (LlvmType::Reference { .. }, Node::ArrayAccess { .. })
                    | (LlvmType::Reference { .. }, Node::SliceAccess { .. })
                    | (LlvmType::Pointer { .. }, Node::MemberAccess { .. })
                    | (LlvmType::Pointer { .. }, Node::ArrayAccess { .. })
                    | (LlvmType::Pointer { .. }, Node::SliceAccess { .. })
            ) {
                let loaded = self.load_from_ptr(&ptr, &current_type)?;
                match current_type.clone() {
                    LlvmType::Reference { target_type, .. } => {
                        ptr = loaded.value;
                        current_type = *target_type;
                    }
                    LlvmType::Pointer { target_type } => {
                        ptr = loaded.value;
                        current_type = *target_type;
                    }
                    _ => unreachable!(),
                }
            }
            match node {
                Node::Dereference { .. } => {
                    match current_type.clone() {
                        LlvmType::Reference { target_type, .. } => {
                            let loaded = self.load_from_ptr(&ptr, &current_type)?;
                            ptr = loaded.value;
                            current_type = *target_type;
                        }
                        LlvmType::Pointer { target_type } => {
                            if !self.unsafe_allowed() {
                                return Err("pointer dereference requires an unsafe block".to_string());
                            }
                            let loaded = self.load_from_ptr(&ptr, &current_type)?;
                            ptr = loaded.value;
                            current_type = *target_type;
                        }
                        other => {
                            return Err(format!(
                                "cannot dereference non-pointer type `{}` in LLVM backend",
                                other.ir()
                            ))
                        }
                    }
                }
                Node::ArrayAccess { coordinates } => {
                    for coordinate in coordinates {
                        let index = self.compile_expr(coordinate)?;
                        let index = self.coerce_expr(index, &LlvmType::I64, "array index")?;
                        match current_type.clone() {
                            LlvmType::Array { elem_type, .. } => {
                                let temp = self.next_temp();
                                self.emit_line(format!(
                                    "{} = getelementptr inbounds {}, ptr {}, i64 0, i64 {}",
                                    temp,
                                    current_type.ir(),
                                    ptr,
                                    index.value
                                ));
                                ptr = temp;
                                current_type = *elem_type;
                            }
                            LlvmType::Slice { elem_type } => {
                                let slice_value = self.load_from_ptr(&ptr, &current_type)?;
                                let data_ptr = self.extract_slice_data(&slice_value)?;
                                let temp = self.next_temp();
                                self.emit_line(format!(
                                    "{} = getelementptr inbounds {}, ptr {}, i64 {}",
                                    temp,
                                    elem_type.ir(),
                                    data_ptr,
                                    index.value
                                ));
                                ptr = temp;
                                current_type = *elem_type;
                            }
                            other => {
                                return Err(format!(
                                    "cannot index non-array type `{}` in LLVM backend",
                                    other.ir()
                                ))
                            }
                        }
                    }
                }
                Node::SliceAccess { start, end } => {
                    let slice_value = self.compile_slice_from_ptr(
                        &ptr,
                        &current_type,
                        start.as_deref(),
                        end.as_deref(),
                    )?;
                    let slice_ptr = self.emit_alloca(slice_value.llvm_type.clone(), "slice_tmp");
                    self.emit_store(&slice_ptr, &slice_value);
                    ptr = slice_ptr;
                    current_type = slice_value.llvm_type;
                }
                Node::MemberAccess { member, .. } => match member.as_ref() {
                    Node::Identifier(name) if name == "len" => break,
                    Node::Identifier(name) => match current_type.clone() {
                        LlvmType::Pointer { .. } => unreachable!(),
                        LlvmType::Struct(struct_name) => {
                            let (field_index, field_type) =
                                self.struct_field_info(&struct_name, name).ok_or_else(|| {
                                    format!(
                                        "unknown field `{}` on struct `{}` in LLVM backend",
                                        name, struct_name
                                    )
                                })?;
                            let temp = self.next_temp();
                            self.emit_line(format!(
                                "{} = getelementptr inbounds {}, ptr {}, i32 0, i32 {}",
                                temp,
                                current_type.ir(),
                                ptr,
                                field_index
                            ));
                            ptr = temp;
                            current_type = field_type;
                        }
                        other @ LlvmType::Allocator | other @ LlvmType::Arena => {
                            return Err(format!(
                                "member `{}` is not a field on `{}` in LLVM backend",
                                name,
                                other.ir()
                            ))
                        }
                        other => {
                            return Err(format!(
                                "member `{}` is not available on `{}` in LLVM backend",
                                name,
                                other.ir()
                            ))
                        }
                    },
                    Node::FunctionCall { .. } => break,
                    other => {
                        return Err(format!(
                            "LLVM backend does not support member access `{:?}` yet",
                            other
                        ))
                    }
                },
                other => {
                    return Err(format!(
                        "LLVM backend does not support access step `{:?}` yet",
                        other
                    ))
                }
            }
        }

        Ok((ptr, current_type))
    }

    fn struct_field_info(&self, struct_name: &str, field_name: &str) -> Option<(usize, LlvmType)> {
        self.structs.get(struct_name).and_then(|layout| {
            layout
                .fields
                .iter()
                .enumerate()
                .find(|(_, (name, _))| name == field_name)
                .map(|(index, (_, llvm_type))| (index, llvm_type.clone()))
        })
    }

    fn visible_locals(&self) -> Vec<(String, LocalVar)> {
        let mut locals = BTreeMap::<String, LocalVar>::new();
        for scope in &self.scopes {
            for (name, local) in scope {
                locals.insert(name.clone(), local.clone());
            }
        }
        locals.into_iter().collect()
    }

    fn compile_lambda_expr(
        &mut self,
        parameters: &[(String, Type)],
        return_type: &Type,
        body: &[Node],
        expected: Option<&LlvmType>,
    ) -> Result<ExprValue, String> {
        let function_type = match expected {
            Some(LlvmType::Function {
                parameters,
                return_type,
            }) => LlvmType::Function {
                parameters: parameters.clone(),
                return_type: return_type.clone(),
            },
            Some(other) => {
                return Err(format!(
                    "lambda expression cannot initialize `{}` in LLVM backend",
                    other.ir()
                ))
            }
            None => LlvmType::Function {
                parameters: parameters
                    .iter()
                    .map(|(_, sk_type)| llvm_type(sk_type, self.structs, self.enums, self.traits))
                    .collect::<Result<Vec<_>, _>>()?,
                return_type: Box::new(llvm_type(return_type, self.structs, self.enums, self.traits)?),
            },
        };

        let visible_locals = self.visible_locals();
        let captures = visible_locals
            .iter()
            .map(|(name, local)| (name.clone(), local.llvm_type.clone()))
            .collect::<Vec<_>>();

        let lambda_id = *self.lambda_counter;
        *self.lambda_counter += 1;
        let symbol_name = format!(
            "skunk_lambda_{}_{}",
            sanitize_name(self.function_name),
            lambda_id
        );
        let env_type_name = format!("{}_env", symbol_name);
        let env = ClosureEnv {
            type_name: env_type_name.clone(),
            captures: captures.clone(),
        };

        let env_fields = if captures.is_empty() {
            String::new()
        } else {
            vec!["ptr"; captures.len()].join(", ")
        };
        self.extra_type_decls.push(format!(
            "%env.{} = type {{ {} }}",
            sanitize_name(&env.type_name),
            env_fields
        ));

        let lambda_return_type = match &function_type {
            LlvmType::Function { return_type, .. } => return_type.as_ref().clone(),
            _ => unreachable!(),
        };

        let nested_compiler = FunctionCompiler::new(
            &symbol_name,
            lambda_return_type.clone(),
            self.signatures,
            self.structs,
            self.enums,
            self.traits,
            self.trait_vtables,
            self.globals,
            self.extra_type_decls,
            self.extra_function_irs,
            self.lambda_counter,
            Some(env.clone()),
        );
        let body_lines = nested_compiler.compile(parameters, body)?;
        let param_defs = parameters
            .iter()
            .enumerate()
            .map(|(index, (_, ty))| {
                Ok(format!(
                    "{} %arg{}",
                    llvm_type(ty, self.structs, self.enums, self.traits)?.ir(),
                    index + 1
                ))
            })
            .collect::<Result<Vec<_>, String>>()?;
        let mut function_ir = String::new();
        let _ = writeln!(
            function_ir,
            "define {} @{}({}) {{",
            lambda_return_type.ir(),
            symbol_name,
            std::iter::once("ptr %env".to_string())
                .chain(param_defs.into_iter())
                .collect::<Vec<_>>()
                .join(", ")
        );
        let _ = writeln!(function_ir, "entry:");
        for line in body_lines {
            let _ = writeln!(function_ir, "{}", line);
        }
        let _ = writeln!(function_ir, "}}");
        self.extra_function_irs.push(function_ir);

        let env_ptr = if visible_locals.is_empty() {
            "null".to_string()
        } else {
            let env_size = captures.len() * 8;
            let env_ptr = self.next_temp();
            self.emit_line(format!("{} = call ptr @malloc(i64 {})", env_ptr, env_size));
            for (index, (_, local)) in visible_locals.iter().enumerate() {
                let field_ptr = self.next_temp();
                self.emit_line(format!(
                    "{} = getelementptr inbounds %env.{}, ptr {}, i32 0, i32 {}",
                    field_ptr,
                    sanitize_name(&env.type_name),
                    env_ptr,
                    index
                ));
                self.emit_line(format!(
                    "store ptr {}, ptr {}, align 8",
                    local.ptr, field_ptr
                ));
            }
            env_ptr
        };

        let with_fn = self.next_temp();
        self.emit_line(format!(
            "{} = insertvalue {} zeroinitializer, ptr @{}, 0",
            with_fn,
            function_type.ir(),
            symbol_name
        ));
        let full = self.next_temp();
        self.emit_line(format!(
            "{} = insertvalue {} {}, ptr {}, 1",
            full,
            function_type.ir(),
            with_fn,
            env_ptr
        ));
        Ok(ExprValue {
            llvm_type: function_type,
            value: full,
        })
    }

    fn compile_direct_call(
        &mut self,
        display_name: &str,
        signature: &FunctionSignature,
        provided_args: &[Node],
        context: &str,
    ) -> Result<ExprValue, String> {
        if provided_args.len() != signature.parameters.len() {
            return Err(format!(
                "{} `{}` expects {} arguments, got {}",
                context,
                display_name,
                signature.parameters.len(),
                provided_args.len()
            ));
        }

        let mut arg_parts = Vec::with_capacity(provided_args.len());
        for (arg_node, expected_type) in provided_args.iter().zip(signature.parameters.iter()) {
            let arg = self.compile_expr_with_expected(arg_node, Some(expected_type))?;
            let arg = self.coerce_expr(arg, expected_type, &format!("{} argument", context))?;
            arg_parts.push(format!("{} {}", arg.llvm_type.ir(), arg.value));
        }

        if signature.return_type == LlvmType::Void {
            self.emit_line(format!(
                "call void @{}({})",
                signature.symbol_name,
                arg_parts.join(", ")
            ));
            Ok(ExprValue {
                llvm_type: LlvmType::Void,
                value: "void".to_string(),
            })
        } else {
            let temp = self.next_temp();
            self.emit_line(format!(
                "{} = call {} @{}({})",
                temp,
                signature.return_type.ir(),
                signature.symbol_name,
                arg_parts.join(", ")
            ));
            Ok(ExprValue {
                llvm_type: signature.return_type.clone(),
                value: temp,
            })
        }
    }

    fn compile_closure_call(
        &mut self,
        callee: ExprValue,
        provided_args: &[Node],
        context: &str,
    ) -> Result<ExprValue, String> {
        let (parameters, return_type) = match &callee.llvm_type {
            LlvmType::Function {
                parameters,
                return_type,
            } => (parameters.clone(), return_type.as_ref().clone()),
            other => {
                return Err(format!(
                    "{} requires a function value, found `{}`",
                    context,
                    other.ir()
                ))
            }
        };

        if provided_args.len() != parameters.len() {
            return Err(format!(
                "{} expects {} arguments, got {}",
                context,
                parameters.len(),
                provided_args.len()
            ));
        }

        let fn_ptr = self.next_temp();
        self.emit_line(format!(
            "{} = extractvalue {} {}, 0",
            fn_ptr,
            callee.llvm_type.ir(),
            callee.value
        ));
        let env_ptr = self.next_temp();
        self.emit_line(format!(
            "{} = extractvalue {} {}, 1",
            env_ptr,
            callee.llvm_type.ir(),
            callee.value
        ));

        let mut arg_parts = vec![format!("ptr {}", env_ptr)];
        for (arg_node, expected_type) in provided_args.iter().zip(parameters.iter()) {
            let arg = self.compile_expr_with_expected(arg_node, Some(expected_type))?;
            let arg = self.coerce_expr(arg, expected_type, "function argument")?;
            arg_parts.push(format!("{} {}", arg.llvm_type.ir(), arg.value));
        }

        if return_type == LlvmType::Void {
            self.emit_line(format!("call void {}({})", fn_ptr, arg_parts.join(", ")));
            Ok(ExprValue {
                llvm_type: LlvmType::Void,
                value: "void".to_string(),
            })
        } else {
            let temp = self.next_temp();
            self.emit_line(format!(
                "{} = call {} {}({})",
                temp,
                return_type.ir(),
                fn_ptr,
                arg_parts.join(", ")
            ));
            Ok(ExprValue {
                llvm_type: return_type,
                value: temp,
            })
        }
    }

    fn compile_function_call(
        &mut self,
        name: &str,
        arguments: &[Vec<Node>],
    ) -> Result<ExprValue, String> {
        if let Some(local) = self.lookup_local(name).cloned() {
            return match local.llvm_type {
                LlvmType::Function { .. } => {
                    let mut current = self.load_from_ptr(&local.ptr, &local.llvm_type)?;
                    for arg_group in arguments {
                        current = self.compile_closure_call(current, arg_group, "function call")?;
                    }
                    Ok(current)
                }
                other => Err(format!(
                    "`{}` is not callable in LLVM backend; found `{}`",
                    name,
                    other.ir()
                )),
            };
        }

        let signature = self
            .signatures
            .get(name)
            .cloned()
            .ok_or_else(|| format!("unknown function `{}` in LLVM backend", name))?;

        let mut current = self.compile_direct_call(name, &signature, &arguments[0], "function")?;
        for arg_group in arguments.iter().skip(1) {
            current = self.compile_closure_call(current, arg_group, "function call")?;
        }
        Ok(current)
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
        self.load_from_ptr(&local.ptr, &local.llvm_type)
    }

    fn resolve_local_from_access(&mut self, node: &Node) -> Result<LocalVar, String> {
        match node {
            Node::Access { nodes } => {
                if let Some(Node::MemberAccess { member, .. }) = nodes.last() {
                    match member.as_ref() {
                        Node::Identifier(name) if name == "len" => {
                            return Err("cannot assign to array length".to_string())
                        }
                        Node::FunctionCall { .. } => {
                            return Err("cannot assign to a method call result".to_string())
                        }
                        _ => {}
                    }
                }
                if matches!(nodes.last(), Some(Node::SliceAccess { .. })) {
                    return Err("cannot assign to a slice expression".to_string());
                }
                let (ptr, llvm_type) = self.resolve_access_ptr(nodes)?;
                Ok(LocalVar { ptr, llvm_type })
            }
            _ => Err("LLVM backend currently supports only access assignments".to_string()),
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
            LlvmType::Allocator => ExprValue {
                llvm_type: LlvmType::Allocator,
                value: "null".to_string(),
            },
            LlvmType::Arena => ExprValue {
                llvm_type: LlvmType::Arena,
                value: "null".to_string(),
            },
            LlvmType::Window => ExprValue {
                llvm_type: LlvmType::Window,
                value: "null".to_string(),
            },
            LlvmType::TraitObject(_) => ExprValue {
                llvm_type: llvm_type.clone(),
                value: "zeroinitializer".to_string(),
            },
            LlvmType::Reference { .. } => ExprValue {
                llvm_type: llvm_type.clone(),
                value: "null".to_string(),
            },
            LlvmType::Pointer { .. } => ExprValue {
                llvm_type: llvm_type.clone(),
                value: "null".to_string(),
            },
            LlvmType::Struct(_) => ExprValue {
                llvm_type: llvm_type.clone(),
                value: "zeroinitializer".to_string(),
            },
            LlvmType::Enum(_) => ExprValue {
                llvm_type: llvm_type.clone(),
                value: "zeroinitializer".to_string(),
            },
            LlvmType::Function { .. } => ExprValue {
                llvm_type: llvm_type.clone(),
                value: "zeroinitializer".to_string(),
            },
            LlvmType::Slice { .. } => ExprValue {
                llvm_type: llvm_type.clone(),
                value: "zeroinitializer".to_string(),
            },
            LlvmType::Array { .. } => ExprValue {
                llvm_type: llvm_type.clone(),
                value: "zeroinitializer".to_string(),
            },
            LlvmType::Void => ExprValue {
                llvm_type: LlvmType::Void,
                value: "void".to_string(),
            },
        }
    }

    fn extract_enum_payloads(
        &mut self,
        value: &ExprValue,
        variant: &EnumVariantLayout,
    ) -> Result<Vec<ExprValue>, String> {
        let mut payloads = Vec::new();
        for (payload_type, field_index) in variant
            .payload_types
            .iter()
            .cloned()
            .zip(variant.field_indices.iter().copied())
        {
            let temp = self.next_temp();
            self.emit_line(format!(
                "{} = extractvalue {} {}, {}",
                temp,
                value.llvm_type.ir(),
                value.value,
                field_index
            ));
            payloads.push(ExprValue {
                llvm_type: payload_type,
                value: temp,
            });
        }
        Ok(payloads)
    }

    fn extract_struct_field_value(
        &mut self,
        value: &ExprValue,
        field_index: usize,
        field_type: LlvmType,
    ) -> Result<ExprValue, String> {
        let temp = self.next_temp();
        self.emit_line(format!(
            "{} = extractvalue {} {}, {}",
            temp,
            value.llvm_type.ir(),
            value.value,
            field_index
        ));
        Ok(ExprValue {
            llvm_type: field_type,
            value: temp,
        })
    }

    fn bind_struct_pattern_fields(
        &mut self,
        struct_name: &str,
        value: &ExprValue,
        fields: &[ast::StructPatternField],
    ) -> Result<(), String> {
        for field in fields {
            let (field_index, field_type) = self
                .struct_field_info(struct_name, &field.field_name)
                .ok_or_else(|| {
                    format!(
                        "unknown field `{}` on struct `{}`",
                        field.field_name, struct_name
                    )
                })?;
            let field_value =
                self.extract_struct_field_value(value, field_index, field_type.clone())?;
            let field_ptr = self.emit_heap_alloc(field_type.clone(), &field.binding);
            self.emit_store(&field_ptr, &field_value);
            self.declare_local(field.binding.clone(), field_ptr, field_type);
        }
        Ok(())
    }

    /// Converts a compiled expression into an expected LLVM type when the
    /// language permits an implicit coercion.
    ///
    /// Numeric widening and trait-object coercions both flow through here.
    fn coerce_expr(
        &mut self,
        value: ExprValue,
        expected: &LlvmType,
        context: &str,
    ) -> Result<ExprValue, String> {
        if &value.llvm_type == expected {
            return Ok(value);
        }

        if let LlvmType::TraitObject(trait_name) = expected {
            return match value.llvm_type.clone() {
                LlvmType::Struct(concrete_name) => {
                    self.box_trait_object(trait_name, &concrete_name, value, context)
                }
                other => Err(format!(
                    "type mismatch in {}: expected `{}`, got `{}`",
                    context,
                    expected.ir(),
                    other.ir()
                )),
            };
        }

        let temp = self.next_temp();
        let line = match (&value.llvm_type, expected) {
            (LlvmType::Struct(_), LlvmType::Struct(_)) => {
                return Err(format!(
                    "type mismatch in {}: expected `{}`, got `{}`",
                    context,
                    expected.ir(),
                    value.llvm_type.ir()
                ))
            }
            (LlvmType::Enum(_), LlvmType::Enum(_)) => {
                return Err(format!(
                    "type mismatch in {}: expected `{}`, got `{}`",
                    context,
                    expected.ir(),
                    value.llvm_type.ir()
                ))
            }
            (LlvmType::Array { .. }, LlvmType::Array { .. }) => {
                return Err(format!(
                    "type mismatch in {}: expected `{}`, got `{}`",
                    context,
                    expected.ir(),
                    value.llvm_type.ir()
                ))
            }
            (
                LlvmType::Reference {
                    target_type: actual_target,
                    mutable: true,
                },
                LlvmType::Reference {
                    target_type: expected_target,
                    mutable: false,
                },
            ) if actual_target == expected_target => {
                return Ok(ExprValue {
                    llvm_type: expected.clone(),
                    value: value.value,
                });
            }
            (LlvmType::Reference { .. }, LlvmType::Reference { .. })
            | 
            (LlvmType::Pointer { .. }, LlvmType::Pointer { .. })
            | (LlvmType::Allocator, LlvmType::Allocator)
            | (LlvmType::Arena, LlvmType::Arena) => {
                return Err(format!(
                    "type mismatch in {}: expected `{}`, got `{}`",
                    context,
                    expected.ir(),
                    value.llvm_type.ir()
                ))
            }
            (LlvmType::Function { .. }, LlvmType::Function { .. }) => {
                return Err(format!(
                    "type mismatch in {}: expected `{}`, got `{}`",
                    context,
                    expected.ir(),
                    value.llvm_type.ir()
                ))
            }
            (LlvmType::Slice { .. }, LlvmType::Slice { .. }) => {
                return Err(format!(
                    "type mismatch in {}: expected `{}`, got `{}`",
                    context,
                    expected.ir(),
                    value.llvm_type.ir()
                ))
            }
            (LlvmType::I8, LlvmType::I16) => format!("{} = sext i8 {} to i16", temp, value.value),
            (LlvmType::I8, LlvmType::I32) => format!("{} = sext i8 {} to i32", temp, value.value),
            (LlvmType::I8, LlvmType::I64) => format!("{} = sext i8 {} to i64", temp, value.value),
            (LlvmType::I16, LlvmType::I32) => format!("{} = sext i16 {} to i32", temp, value.value),
            (LlvmType::I16, LlvmType::I64) => format!("{} = sext i16 {} to i64", temp, value.value),
            (LlvmType::I32, LlvmType::I64) => format!("{} = sext i32 {} to i64", temp, value.value),
            (LlvmType::Char16, LlvmType::I32) => {
                format!("{} = zext i16 {} to i32", temp, value.value)
            }
            (LlvmType::I32, LlvmType::I16) => {
                format!("{} = trunc i32 {} to i16", temp, value.value)
            }
            (LlvmType::I32, LlvmType::I8) => format!("{} = trunc i32 {} to i8", temp, value.value),
            (LlvmType::I64, LlvmType::I32) => {
                format!("{} = trunc i64 {} to i32", temp, value.value)
            }
            (LlvmType::I64, LlvmType::I16) => {
                format!("{} = trunc i64 {} to i16", temp, value.value)
            }
            (LlvmType::I64, LlvmType::I8) => format!("{} = trunc i64 {} to i8", temp, value.value),
            (LlvmType::I8, LlvmType::F32) => {
                format!("{} = sitofp i8 {} to float", temp, value.value)
            }
            (LlvmType::I8, LlvmType::F64) => {
                format!("{} = sitofp i8 {} to double", temp, value.value)
            }
            (LlvmType::I16, LlvmType::F32) => {
                format!("{} = sitofp i16 {} to float", temp, value.value)
            }
            (LlvmType::I16, LlvmType::F64) => {
                format!("{} = sitofp i16 {} to double", temp, value.value)
            }
            (LlvmType::I32, LlvmType::F32) => {
                format!("{} = sitofp i32 {} to float", temp, value.value)
            }
            (LlvmType::I32, LlvmType::F64) => {
                format!("{} = sitofp i32 {} to double", temp, value.value)
            }
            (LlvmType::I64, LlvmType::F32) => {
                format!("{} = sitofp i64 {} to float", temp, value.value)
            }
            (LlvmType::I64, LlvmType::F64) => {
                format!("{} = sitofp i64 {} to double", temp, value.value)
            }
            (LlvmType::F32, LlvmType::F64) => {
                format!("{} = fpext float {} to double", temp, value.value)
            }
            (LlvmType::F64, LlvmType::F32) => {
                format!("{} = fptrunc double {} to float", temp, value.value)
            }
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

    fn box_trait_object(
        &mut self,
        trait_name: &str,
        concrete_name: &str,
        value: ExprValue,
        context: &str,
    ) -> Result<ExprValue, String> {
        let vtable_symbol = self.trait_vtable_symbol(trait_name, concrete_name, context)?;

        let allocator = self.next_temp();
        self.emit_line(format!("{} = call ptr @skunk_system_allocator()", allocator));
        let boxed_ptr = self.next_temp();
        self.emit_line(format!(
            "{} = call ptr @skunk_alloc_create(ptr {}, i64 {})",
            boxed_ptr,
            allocator,
            self.size_of(&value.llvm_type)
        ));
        self.emit_store(&boxed_ptr, &value);

        self.trait_object_from_data_ptr(trait_name, boxed_ptr, &vtable_symbol)
    }

    fn trait_vtable_symbol(
        &self,
        trait_name: &str,
        concrete_name: &str,
        context: &str,
    ) -> Result<String, String> {
        self.trait_vtables
            .get(&format!("{}=>{}", trait_name, concrete_name))
            .cloned()
            .ok_or_else(|| {
                format!(
                    "type mismatch in {}: `{}` does not implement trait `{}`",
                    context, concrete_name, trait_name
                )
            })
    }

    fn trait_object_from_ptr(
        &mut self,
        trait_name: &str,
        concrete_name: &str,
        data_ptr: String,
        context: &str,
    ) -> Result<ExprValue, String> {
        let vtable_symbol = self.trait_vtable_symbol(trait_name, concrete_name, context)?;
        self.trait_object_from_data_ptr(trait_name, data_ptr, &vtable_symbol)
    }

    fn trait_object_from_data_ptr(
        &mut self,
        trait_name: &str,
        data_ptr: String,
        vtable_symbol: &str,
    ) -> Result<ExprValue, String> {
        let trait_type = LlvmType::TraitObject(trait_name.to_string());
        let with_data = self.next_temp();
        self.emit_line(format!(
            "{} = insertvalue {} zeroinitializer, ptr {}, 0",
            with_data,
            trait_type.ir(),
            data_ptr
        ));
        let boxed_value = self.next_temp();
        self.emit_line(format!(
            "{} = insertvalue {} {}, ptr @{}, 1",
            boxed_value,
            trait_type.ir(),
            with_data,
            vtable_symbol
        ));

        Ok(ExprValue {
            llvm_type: trait_type,
            value: boxed_value,
        })
    }

    fn global_c_string(&mut self, prefix: &str, value: &str) -> String {
        let mut bytes = value.as_bytes().to_vec();
        bytes.push(0);
        if let Some(existing) = self.globals.iter().find(|g| g.bytes == bytes) {
            return existing.name.clone();
        }
        let name = format!("{}.{}", prefix, self.globals.len());
        self.globals.push(GlobalString {
            name: name.clone(),
            bytes,
        });
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
            self.align_of(&expr.llvm_type)
        ));
    }

    fn emit_alloca(&mut self, llvm_type: LlvmType, hint: &str) -> String {
        let ptr = format!("%{}_{}", sanitize_name(hint), self.temp_counter);
        self.temp_counter += 1;
        self.emit_line(format!(
            "{} = alloca {}, align {}",
            ptr,
            llvm_type.ir(),
            self.align_of(&llvm_type)
        ));
        ptr
    }

    fn emit_heap_alloc(&mut self, llvm_type: LlvmType, hint: &str) -> String {
        let ptr = format!("%{}_{}", sanitize_name(hint), self.temp_counter);
        self.temp_counter += 1;
        self.emit_line(format!(
            "{} = call ptr @malloc(i64 {})",
            ptr,
            self.size_of(&llvm_type)
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
        let label = format!(
            "{}_{}_{}",
            sanitize_name(self.function_name),
            prefix,
            self.label_counter
        );
        self.label_counter += 1;
        label
    }

    fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    fn enter_unsafe(&mut self) {
        self.unsafe_depth += 1;
    }

    fn exit_unsafe(&mut self) {
        if self.unsafe_depth > 0 {
            self.unsafe_depth -= 1;
        }
    }

    fn unsafe_allowed(&self) -> bool {
        self.unsafe_depth > 0
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

    fn align_of(&self, llvm_type: &LlvmType) -> usize {
        match llvm_type {
            LlvmType::I8 | LlvmType::I1 => 1,
            LlvmType::I16 | LlvmType::Char16 => 2,
            LlvmType::I32 | LlvmType::F32 => 4,
            LlvmType::I64
            | LlvmType::F64
            | LlvmType::PtrI8
            | LlvmType::Allocator
            | LlvmType::Arena
            | LlvmType::Window
            | LlvmType::TraitObject(_)
            | LlvmType::Reference { .. }
            | LlvmType::Pointer { .. } => 8,
            LlvmType::Function { .. } => 8,
            LlvmType::Struct(name) => self
                .structs
                .get(name)
                .map(|layout| {
                    layout
                        .fields
                        .iter()
                        .map(|(_, field_type)| self.align_of(field_type))
                        .max()
                        .unwrap_or(1)
                })
                .unwrap_or(8),
            LlvmType::Enum(name) => self
                .enums
                .get(name)
                .map(|layout| {
                    layout
                        .variants
                        .iter()
                        .flat_map(|variant| variant.payload_types.iter())
                        .map(|payload_type| self.align_of(payload_type))
                        .max()
                        .unwrap_or(4)
                        .max(4)
                })
                .unwrap_or(8),
            LlvmType::Slice { .. } => 8,
            LlvmType::Array { elem_type, .. } => self.align_of(elem_type),
            LlvmType::Void => 1,
        }
    }

    fn size_of(&self, llvm_type: &LlvmType) -> usize {
        match llvm_type {
            LlvmType::I8 | LlvmType::I1 => 1,
            LlvmType::I16 | LlvmType::Char16 => 2,
            LlvmType::I32 | LlvmType::F32 => 4,
            LlvmType::I64
            | LlvmType::F64
            | LlvmType::PtrI8
            | LlvmType::Allocator
            | LlvmType::Arena
            | LlvmType::Window
            | LlvmType::Reference { .. }
            | LlvmType::Pointer { .. } => 8,
            LlvmType::TraitObject(_) => 16,
            LlvmType::Function { .. } => 16,
            LlvmType::Slice { .. } => 16,
            LlvmType::Array { elem_type, len } => self.size_of(elem_type) * len,
            LlvmType::Struct(name) => {
                let Some(layout) = self.structs.get(name) else {
                    return 8;
                };
                let mut offset = 0usize;
                let mut max_align = 1usize;
                for (_, field_type) in &layout.fields {
                    let align = self.align_of(field_type);
                    max_align = max_align.max(align);
                    offset = align_up(offset, align);
                    offset += self.size_of(field_type);
                }
                align_up(offset, max_align)
            }
            LlvmType::Enum(name) => {
                let Some(layout) = self.enums.get(name) else {
                    return 8;
                };
                let mut offset = 4usize;
                let mut max_align = 4usize;
                for variant in &layout.variants {
                    for payload_type in &variant.payload_types {
                        let align = self.align_of(payload_type);
                        max_align = max_align.max(align);
                        offset = align_up(offset, align);
                        offset += self.size_of(payload_type);
                    }
                }
                align_up(offset, max_align)
            }
            LlvmType::Void => 1,
        }
    }
}

fn align_up(value: usize, align: usize) -> usize {
    if align <= 1 {
        value
    } else {
        let rem = value % align;
        if rem == 0 {
            value
        } else {
            value + (align - rem)
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

/// Compiles a checked Skunk program into LLVM IR and a native executable.
pub fn compile_to_executable(
    program: &Node,
    source_path: &Path,
    output_path: &Path,
) -> Result<CompiledArtifact, String> {
    let llvm_ir = compile_to_llvm_ir(program)?;
    let llvm_ir_path = output_path.with_extension("ll");
    fs::write(&llvm_ir_path, llvm_ir).map_err(|err| {
        format!(
            "failed to write LLVM IR to {}: {}",
            llvm_ir_path.display(),
            err
        )
    })?;

    let runtime_c_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("runtime/skunk_runtime.c");
    let runtime_window_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("runtime/skunk_window_runtime.m");
    let mut command = Command::new("clang");
    command.arg(&llvm_ir_path).arg(&runtime_c_path);
    if cfg!(target_os = "macos") {
        command
            .arg(&runtime_window_path)
            .arg("-framework")
            .arg("Cocoa");
    }
    let status = command
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

/// Lowers a checked Skunk program into textual LLVM IR without invoking the
/// system linker.
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

    let structs = collect_struct_layouts(statements)?;
    let enums = collect_enum_layouts(statements, &structs)?;
    let traits = collect_trait_layouts(statements, &structs, &enums)?;
    let mut signatures = HashMap::<String, FunctionSignature>::new();
    let mut functions = Vec::<FunctionPlan>::new();
    let mut trait_vtables = HashMap::<String, String>::new();
    let mut trait_vtable_globals = Vec::<String>::new();

    for statement in statements {
        match statement {
            Node::FunctionDeclaration {
                name,
                parameters,
                return_type,
                body,
                ..
            } => {
                let params = parameters
                    .iter()
                    .map(|(_, ty)| llvm_type(ty, &structs, &enums, &traits))
                    .collect::<Result<Vec<_>, _>>()?;
                let llvm_return_type = llvm_type(return_type, &structs, &enums, &traits)?;
                signatures.insert(
                    name.clone(),
                    FunctionSignature {
                        symbol_name: format!("skunk_{}", name),
                        return_type: llvm_return_type,
                        parameters: params,
                    },
                );
                functions.push(FunctionPlan {
                    symbol_name: format!("skunk_{}", name),
                    parameters: parameters.clone(),
                    return_type: return_type.clone(),
                    body: body.clone(),
                    is_method: false,
                });
            }
            Node::EOI => {}
            Node::EnumDeclaration { .. } => {}
            Node::TraitDeclaration { .. }
            | Node::ShapeDeclaration { .. }
            | Node::ImplDeclaration { .. } => {}
            Node::StructDeclaration {
                name,
                functions: struct_functions,
                ..
            } => {
                for function in struct_functions {
                    if let Node::FunctionDeclaration {
                        name: method_name,
                        parameters,
                        return_type,
                        body,
                        ..
                    } = function
                    {
                        let mut parameter_types = Vec::new();
                        let has_receiver = parameters
                            .first()
                            .is_some_and(|(_, param_type)| ast::is_self_type(param_type));
                        let mut compile_params = Vec::new();
                        for (index, (param_name, param_type)) in parameters.iter().enumerate() {
                            if has_receiver && index == 0 {
                                continue;
                            }
                            parameter_types.push(llvm_type(param_type, &structs, &enums, &traits)?);
                            compile_params.push((param_name.clone(), param_type.clone()));
                        }
                        let llvm_return_type = llvm_type(return_type, &structs, &enums, &traits)?;
                        let key = format!("{}::{}", name, method_name);
                        let symbol_name =
                            format!("skunk_{}_{}", sanitize_name(name), sanitize_name(method_name));
                        signatures.insert(
                            key,
                            FunctionSignature {
                                symbol_name: symbol_name.clone(),
                                return_type: llvm_return_type.clone(),
                                parameters: parameter_types,
                            },
                        );
                        functions.push(FunctionPlan {
                            symbol_name,
                            parameters: if has_receiver {
                                let mut method_params =
                                    vec![("self".to_string(), Type::Custom(name.clone()))];
                                method_params.extend(compile_params);
                                method_params
                            } else {
                                compile_params
                            },
                            return_type: return_type.clone(),
                            body: body.clone(),
                            is_method: has_receiver,
                        });
                    }
                }
            }
            other => {
                return Err(format!(
                    "LLVM backend currently expects top-level function, struct, and enum declarations only, found `{:?}`",
                    other
                ))
            }
        }
    }

    if !signatures.contains_key("main") {
        return Err("LLVM backend currently requires `function main(): ... {}`".to_string());
    }

    for statement in statements {
        let Node::ImplDeclaration {
            generic_params,
            trait_names,
            target_type,
            ..
        } = statement
        else {
            continue;
        };
        if !generic_params.is_empty() {
            continue;
        }
        let target_name = match target_type {
            Type::Custom(name) => name.clone(),
            other => {
                return Err(format!(
                    "runtime trait values currently require concrete nominal impl targets, found `{}`",
                    ast::type_to_string(other)
                ))
            }
        };
        for trait_name in trait_names {
            let trait_layout = traits
                .get(trait_name)
                .ok_or_else(|| format!("unknown trait `{}` in LLVM backend", trait_name))?;
            let vtable_symbol = format!(
                "skunk_vtable_{}_{}",
                sanitize_name(trait_name),
                sanitize_name(&target_name)
            );
            trait_vtables.insert(
                format!("{}=>{}", trait_name, target_name),
                vtable_symbol.clone(),
            );
            let entries = trait_layout
                .methods
                .iter()
                .map(|method| {
                    let signature_key = format!("{}::{}", target_name, method.name);
                    let signature = signatures.get(&signature_key).ok_or_else(|| {
                        format!(
                            "missing concrete method `{}` for trait `{}` implementation on `{}`",
                            method.name, trait_name, target_name
                        )
                    })?;
                    Ok(format!("ptr @{}", signature.symbol_name))
                })
                .collect::<Result<Vec<_>, String>>()?;
            trait_vtable_globals.push(format!(
                "@{} = private unnamed_addr constant %vtable.{} {{ {} }}",
                vtable_symbol,
                sanitize_name(trait_name),
                entries.join(", ")
            ));
        }
    }

    let mut globals = Vec::<GlobalString>::new();
    let mut extra_type_decls = Vec::<String>::new();
    let mut function_irs = Vec::<String>::new();
    let mut extra_function_irs = Vec::<String>::new();
    let mut lambda_counter = 0usize;

    for function in &functions {
        let llvm_return_type = llvm_type(&function.return_type, &structs, &enums, &traits)?;
        let param_defs = if function.is_method {
            let mut defs = vec![format!("ptr %arg0")];
            for (index, (_, ty)) in function.parameters.iter().skip(1).enumerate() {
                defs.push(format!(
                    "{} %arg{}",
                    llvm_type(ty, &structs, &enums, &traits)?.ir(),
                    index + 1
                ));
            }
            defs
        } else {
            function
                .parameters
                .iter()
                .enumerate()
                .map(|(index, (_, ty))| {
                    Ok(format!(
                        "{} %arg{}",
                        llvm_type(ty, &structs, &enums, &traits)?.ir(),
                        index
                    ))
                })
                .collect::<Result<Vec<_>, String>>()?
        };
        let compiler = FunctionCompiler::new(
            &function.symbol_name,
            llvm_return_type.clone(),
            &signatures,
            &structs,
            &enums,
            &traits,
            &trait_vtables,
            &mut globals,
            &mut extra_type_decls,
            &mut extra_function_irs,
            &mut lambda_counter,
            None,
        );
        let body_lines = compiler.compile(&function.parameters, &function.body)?;
        let mut function_ir = String::new();
        let _ = writeln!(
            function_ir,
            "define {} @{}({}) {{",
            llvm_return_type.ir(),
            function.symbol_name,
            param_defs.join(", ")
        );
        let _ = writeln!(function_ir, "entry:");
        for line in body_lines {
            let _ = writeln!(function_ir, "{}", line);
        }
        let _ = writeln!(function_ir, "}}");
        function_irs.push(function_ir);
    }

    let main_signature = signatures.get("main").expect("validated above");
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
        LlvmType::Allocator
        | LlvmType::Arena
        | LlvmType::Window
        | LlvmType::TraitObject(_)
        | LlvmType::Reference { .. }
        | LlvmType::Pointer { .. } => {
            return Err("LLVM backend does not support `main` returning pointer-like values yet".to_string())
        }
        LlvmType::Function { .. } => {
            return Err("LLVM backend does not support `main` returning functions yet".to_string())
        }
        LlvmType::Slice { .. } => {
            return Err("LLVM backend does not support `main` returning slices yet".to_string())
        }
        LlvmType::Array { .. } => {
            return Err("LLVM backend does not support `main` returning arrays yet".to_string())
        }
        LlvmType::Struct(_) | LlvmType::Enum(_) => {
            return Err(
                "LLVM backend does not support `main` returning structs or enums yet".to_string()
            )
        }
    };

    let mut ir = String::new();
    let _ = writeln!(ir, "declare i32 @printf(ptr, ...)");
    let _ = writeln!(ir, "declare ptr @malloc(i64)");
    let _ = writeln!(ir, "declare ptr @memcpy(ptr, ptr, i64)");
    let _ = writeln!(ir, "declare ptr @memset(ptr, i32, i64)");
    let _ = writeln!(ir, "declare ptr @skunk_system_allocator()");
    let _ = writeln!(ir, "declare ptr @skunk_arena_init(ptr)");
    let _ = writeln!(ir, "declare ptr @skunk_arena_allocator(ptr)");
    let _ = writeln!(ir, "declare void @skunk_arena_reset(ptr)");
    let _ = writeln!(ir, "declare void @skunk_arena_deinit(ptr)");
    let _ = writeln!(ir, "declare ptr @skunk_alloc_create(ptr, i64)");
    let _ = writeln!(ir, "declare ptr @skunk_alloc_buffer(ptr, i64, i32)");
    let _ = writeln!(ir, "declare void @skunk_alloc_destroy(ptr, ptr)");
    let _ = writeln!(ir, "declare void @skunk_alloc_free(ptr, ptr)");
    let _ = writeln!(ir, "declare ptr @skunk_window_create(i32, i32, ptr)");
    let _ = writeln!(ir, "declare i1 @skunk_window_is_open(ptr)");
    let _ = writeln!(ir, "declare void @skunk_window_poll(ptr)");
    let _ = writeln!(ir, "declare void @skunk_window_clear(ptr, i32)");
    let _ = writeln!(ir, "declare void @skunk_window_draw_rect(ptr, double, double, double, double, i32)");
    let _ = writeln!(ir, "declare void @skunk_window_present(ptr)");
    let _ = writeln!(ir, "declare double @skunk_window_delta_time(ptr)");
    let _ = writeln!(ir, "declare void @skunk_window_close(ptr)");
    let _ = writeln!(ir, "declare void @skunk_window_deinit(ptr)");
    let _ = writeln!(ir, "declare i1 @skunk_keyboard_is_down(ptr, i16)");
    let _ = writeln!(ir);
    for layout in traits.values() {
        let _ = writeln!(
            ir,
            "%trait.{} = type {{ ptr, ptr }}",
            sanitize_name(&layout.name)
        );
        let vtable_fields = if layout.methods.is_empty() {
            String::new()
        } else {
            std::iter::repeat("ptr")
                .take(layout.methods.len())
                .collect::<Vec<_>>()
                .join(", ")
        };
        let _ = writeln!(
            ir,
            "%vtable.{} = type {{ {} }}",
            sanitize_name(&layout.name),
            vtable_fields
        );
    }
    if !traits.is_empty() {
        let _ = writeln!(ir);
    }
    for layout in structs.values() {
        let field_types = layout
            .fields
            .iter()
            .map(|(_, field_type)| field_type.ir())
            .collect::<Vec<_>>()
            .join(", ");
        let _ = writeln!(
            ir,
            "%struct.{} = type {{ {} }}",
            sanitize_name(&layout.name),
            field_types
        );
    }
    if !structs.is_empty() {
        let _ = writeln!(ir);
    }
    for layout in enums.values() {
        let mut field_types = vec!["i32".to_string()];
        for variant in &layout.variants {
            for payload_type in &variant.payload_types {
                field_types.push(payload_type.ir());
            }
        }
        let _ = writeln!(
            ir,
            "%enum.{} = type {{ {} }}",
            sanitize_name(&layout.name),
            field_types.join(", ")
        );
    }
    if !enums.is_empty() {
        let _ = writeln!(ir);
    }
    for global in &trait_vtable_globals {
        let _ = writeln!(ir, "{}", global);
    }
    if !trait_vtable_globals.is_empty() {
        let _ = writeln!(ir);
    }
    for decl in &extra_type_decls {
        let _ = writeln!(ir, "{}", decl);
    }
    if !extra_type_decls.is_empty() {
        let _ = writeln!(ir);
    }
    for global in &globals {
        let _ = writeln!(ir, "{}", global.ir_decl());
    }
    if !globals.is_empty() {
        let _ = writeln!(ir);
    }
    for function_ir in function_irs {
        let _ = writeln!(ir, "{}", function_ir);
    }
    for function_ir in extra_function_irs {
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
    use crate::monomorphize;
    use crate::source;
    use crate::type_checker;
    use std::env;
    use std::path::Path;
    use std::process::Command;
    use uuid::Uuid;

    fn compile_and_run(source: &str) -> Result<String, String> {
        let program = ast::parse(source);
        let program = monomorphize::prepare_program(&program)?;
        type_checker::check(&program)?;

        let id = Uuid::new_v4().to_string();
        let source_path = env::temp_dir().join(format!("skunk_compiler_test_{}.skunk", id));
        let output_path = env::temp_dir().join(format!("skunk_compiler_test_{}", id));
        fs::write(&source_path, source)
            .map_err(|err| format!("failed to write test source: {}", err))?;

        let artifact = compile_to_executable(&program, &source_path, &output_path)?;
        let output = Command::new(&artifact.binary_path)
            .output()
            .map_err(|err| format!("failed to run compiled test binary: {}", err))?;

        let _ = fs::remove_file(&source_path);
        let _ = fs::remove_file(&artifact.llvm_ir_path);
        let _ = fs::remove_file(&artifact.binary_path);

        if !output.status.success() {
            return Err(format!(
                "compiled program exited with status {}",
                output.status
            ));
        }

        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }

    fn compile_and_run_with_env(source: &str, env_vars: &[(&str, &str)]) -> Result<String, String> {
        let program = ast::parse(source);
        let program = monomorphize::prepare_program(&program)?;
        type_checker::check(&program)?;

        let id = Uuid::new_v4().to_string();
        let source_path = env::temp_dir().join(format!("skunk_compiler_test_{}.skunk", id));
        let output_path = env::temp_dir().join(format!("skunk_compiler_test_{}", id));
        fs::write(&source_path, source)
            .map_err(|err| format!("failed to write test source: {}", err))?;

        let artifact = compile_to_executable(&program, &source_path, &output_path)?;
        let mut command = Command::new(&artifact.binary_path);
        for (key, value) in env_vars {
            command.env(key, value);
        }
        let output = command
            .output()
            .map_err(|err| format!("failed to run compiled test binary: {}", err))?;

        let _ = fs::remove_file(&source_path);
        let _ = fs::remove_file(&artifact.llvm_ir_path);
        let _ = fs::remove_file(&artifact.binary_path);

        if !output.status.success() {
            return Err(format!(
                "compiled program exited with status {}",
                output.status
            ));
        }

        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }

    fn compile_project_and_run(files: &[(&str, &str)], entry: &str) -> Result<String, String> {
        let root = env::temp_dir().join(format!("skunk_compiler_project_{}", Uuid::new_v4()));
        fs::create_dir_all(&root)
            .map_err(|err| format!("failed to create test project root: {}", err))?;

        for (relative_path, contents) in files {
            let path = root.join(relative_path);
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)
                    .map_err(|err| format!("failed to create `{}`: {}", parent.display(), err))?;
            }
            fs::write(&path, contents)
                .map_err(|err| format!("failed to write `{}`: {}", path.display(), err))?;
        }

        let entry_path = root.join(entry);
        let program = source::load_program(&entry_path)?;
        let program = monomorphize::prepare_program(&program)?;
        type_checker::check(&program)?;

        let output_path = root.join("app_out");
        let artifact = compile_to_executable(&program, Path::new(&entry_path), &output_path)?;
        let output = Command::new(&artifact.binary_path)
            .output()
            .map_err(|err| format!("failed to run compiled test binary: {}", err))?;

        let _ = fs::remove_dir_all(&root);

        if !output.status.success() {
            return Err(format!(
                "compiled program exited with status {}",
                output.status
            ));
        }

        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }

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
    fn compiles_structs_to_ir() {
        let program = ast::parse(
            r#"
            struct Point {
                x: int;
                y: int;
            }

            attach Point {
                function sum(self): int {
                    return self.x + self.y;
                }
            }

            function main(): void {
                p: Point = Point { x: 2, y: 3 };
                print(p.sum());
            }
            "#,
        );

        let ir = compile_to_llvm_ir(&program).unwrap();
        assert!(ir.contains("%struct.Point = type { i32, i32 }"));
        assert!(ir.contains("define i32 @skunk_Point_sum(ptr %arg0)"));
        assert!(ir.contains("insertvalue %struct.Point"));
        assert!(ir.contains("getelementptr inbounds %struct.Point"));
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

    #[test]
    fn compiles_fixed_arrays_to_ir() {
        let program = ast::parse(
            r#"
            function main(): void {
                values: [3]int = [1, 2, 3];
                values[1] = values[0] + 9;
                print(values[1]);
                print(values.len);
            }
            "#,
        );

        let ir = compile_to_llvm_ir(&program).unwrap();
        assert!(ir.contains("[3 x i32]"));
        assert!(ir.contains("getelementptr inbounds [3 x i32]"));
        assert!(ir.contains("insertvalue [3 x i32]"));
    }

    #[test]
    fn runs_compiled_fixed_array_program() {
        let stdout = compile_and_run(
            r#"
            function main(): void {
                zeros: [3]int;
                filled: [3]int = [3]int::fill(7);
                filled[1] = filled[1] + 1;
                print(zeros[0]);
                print(filled[1]);
                print(filled.len);
            }
            "#,
        )
        .unwrap();

        assert_eq!(stdout, "0\n8\n3\n");
    }

    #[test]
    fn runs_compiled_nested_array_program() {
        let stdout = compile_and_run(
            r#"
            function main(): void {
                matrix: [2][2]int = [
                    [1, 2],
                    [3, 4]
                ];
                matrix[1][0] = matrix[0][1] + 5;
                print(matrix[1][0]);
                print(matrix[0].len);
            }
            "#,
        )
        .unwrap();

        assert_eq!(stdout, "7\n2\n");
    }

    #[test]
    fn runs_compiled_array_return_and_argument_program() {
        let stdout = compile_and_run(
            r#"
            function make(): [3]int {
                return [3]int::fill(2);
            }

            function sum(values: [3]int): int {
                return values[0] + values[1] + values[2];
            }

            function main(): void {
                values: [3]int = make();
                values[1] = 5;
                print(sum(values));
            }
            "#,
        )
        .unwrap();

        assert_eq!(stdout, "9\n");
    }

    #[test]
    fn runs_compiled_struct_field_program() {
        let stdout = compile_and_run(
            r#"
            struct Point {
                x: int;
                y: int;
            }

            function main(): void {
                p: Point = Point { x: 1, y: 2 };
                p.x = p.x + 9;
                print(p.x);
                print(p.y);
            }
            "#,
        )
        .unwrap();

        assert_eq!(stdout, "10\n2\n");
    }

    #[test]
    fn runs_compiled_nested_struct_program() {
        let stdout = compile_and_run(
            r#"
            struct Point {
                x: int;
                y: int;
            }

            struct Line {
                start: Point;
                end: Point;
            }

            function main(): void {
                line: Line = Line {
                    start: Point { x: 3, y: 4 },
                    end: Point { x: 5, y: 6 }
                };
                line.start.x = line.start.x + line.end.y;
                print(line.start.x);
                print(line.end.x);
            }
            "#,
        )
        .unwrap();

        assert_eq!(stdout, "9\n5\n");
    }

    #[test]
    fn runs_compiled_struct_method_program() {
        let stdout = compile_and_run(
            r#"
            struct Point {
                x: int;
                y: int;
            }

            attach Point {
                function set_x(mut self, x: int): void {
                    self.x = x;
                }

                function sum(self): int {
                    return self.x + self.y;
                }
            }

            function main(): void {
                p: Point = Point { x: 1, y: 2 };
                p.set_x(10);
                print(p.sum());
            }
            "#,
        )
        .unwrap();

        assert_eq!(stdout, "12\n");
    }

    #[test]
    fn runs_compiled_static_attached_function_program() {
        let stdout = compile_and_run(
            r#"
            struct Point {
                x: int;
                y: int;
            }

            attach Point {
                function new(x: int, y: int): Point {
                    return Point { x: x, y: y };
                }

                function origin(): Point {
                    return Point { x: 0, y: 0 };
                }

                function sum(self): int {
                    return self.x + self.y;
                }
            }

            function main(): void {
                point: Point = Point::new(4, 9);
                origin: Point = Point::origin();
                print(point.sum());
                print(origin.sum());
            }
            "#,
        )
        .unwrap();

        assert_eq!(stdout, "13\n0\n");
    }

    #[test]
    fn runs_compiled_mut_self_method_program() {
        let stdout = compile_and_run(
            r#"
            struct Counter {
                value: int;
            }

            attach Counter {
                function bump(mut self): void {
                    self.value = self.value + 1;
                }

                function get(self): int {
                    return self.value;
                }
            }

            function main(): void {
                counter: Counter = Counter { value: 9 };
                counter.bump();
                print(counter.get());
            }
            "#,
        )
        .unwrap();

        assert_eq!(stdout, "10\n");
    }

    #[test]
    fn runs_compiled_struct_destructure_program() {
        let stdout = compile_and_run(
            r#"
            struct Point {
                x: int;
                y: int;
            }

            function main(): void {
                point: Point = Point { x: 5, y: 8 };
                Point { x, y: py } = point;
                print(x);
                print(py);
                print(x + py);
            }
            "#,
        )
        .unwrap();

        assert_eq!(stdout, "5\n8\n13\n");
    }

    #[test]
    fn runs_compiled_struct_match_pattern_program() {
        let stdout = compile_and_run(
            r#"
            struct Point {
                x: int;
                y: int;
            }

            function sum(point: Point): int {
                match (point) {
                    case Point { x, y }: {
                        return x + y;
                    }
                }
            }

            function main(): void {
                print(sum(Point { x: 2, y: 9 }));
            }
            "#,
        )
        .unwrap();

        assert_eq!(stdout, "11\n");
    }

    #[test]
    fn rejects_calling_mut_self_method_on_const_binding() {
        let result = compile_and_run(
            r#"
            struct Counter {
                value: int;
            }

            attach Counter {
                function bump(mut self): void {
                    self.value = self.value + 1;
                }
            }

            function main(): void {
                const counter: Counter = Counter { value: 0 };
                counter.bump();
            }
            "#,
        );

        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("cannot call mutating method through const or immutable receiver"));
    }

    #[test]
    fn runs_compiled_const_pointer_readonly_method_program() {
        let stdout = compile_and_run(
            r#"
            struct Point {
                x: int;
            }

            attach Point {
                function get(self): int {
                    return self.x;
                }
            }

            function print_point(point: *const Point): void {
                print(point.get());
            }

            function main(): void {
                heap: Allocator = System::allocator();
                point: *Point = Point::create(heap);
                point.x = 7;
                print_point(point);
                heap.destroy(point);
            }
            "#,
        )
        .unwrap();

        assert_eq!(stdout, "7\n");
    }

    #[test]
    fn runs_compiled_slice_program() {
        let stdout = compile_and_run(
            r#"
            function sum_pair(values: []int): int {
                pair: []int = values[1:3];
                return pair[0] + pair[1];
            }

            function main(): void {
                values: [5]int = [10, 20, 30, 40, 50];
                middle: []int = values[1:4];
                head: []int = values[:2];
                tail: []int = middle[1:];
                print(middle.len);
                print(head[1]);
                print(tail[0]);
                print(sum_pair(values[:4]));
            }
            "#,
        )
        .unwrap();

        assert_eq!(stdout, "3\n20\n30\n50\n");
    }

    #[test]
    fn runs_compiled_slice_literal_program() {
        let stdout = compile_and_run(
            r#"
            function main(): void {
                values: []int = [1, 2, 3];
                print(values.len);
                print(values[2]);
            }
            "#,
        )
        .unwrap();

        assert_eq!(stdout, "3\n3\n");
    }

    #[test]
    fn runs_compiled_lambda_program() {
        let stdout = compile_and_run(
            r#"
            function main(): void {
                id: (int) -> int = function(a: int): int {
                    return a;
                };
                print(id(7));
            }
            "#,
        )
        .unwrap();

        assert_eq!(stdout, "7\n");
    }

    #[test]
    fn runs_compiled_closure_program() {
        let stdout = compile_and_run(
            r#"
            function counter(): () -> int {
                c: int = 0;
                return function(): int {
                    c = c + 1;
                    return c;
                };
            }

            function main(): void {
                count: () -> int = counter();
                print(count());
                print(count());
            }
            "#,
        )
        .unwrap();

        assert_eq!(stdout, "1\n2\n");
    }

    #[test]
    fn runs_compiled_recursive_lambda_program() {
        let stdout = compile_and_run(
            r#"
            function main(): void {
                factorial: (int) -> int = function(n: int): int {
                    if (n == 0) {
                        return 1;
                    } else {
                        return n * factorial(n - 1);
                    }
                };
                print(factorial(5));
            }
            "#,
        )
        .unwrap();

        assert_eq!(stdout, "120\n");
    }

    #[test]
    fn runs_compiled_method_returning_lambda_program() {
        let stdout = compile_and_run(
            r#"
            struct Foo {
                factor: int;
            }

            attach Foo {
                function make(self): (int) -> int {
                    return function(i: int): int {
                        return self.factor + i;
                    };
                }
            }

            function main(): void {
                foo: Foo = Foo { factor: 4 };
                print(foo.make()(3));
            }
            "#,
        )
        .unwrap();

        assert_eq!(stdout, "7\n");
    }

    #[test]
    fn runs_compiled_pointer_allocator_program() {
        let stdout = compile_and_run(
            r#"
            struct Point {
                x: int;
                y: int;
            }

            attach Point {
                function sum(self): int {
                    return self.x + self.y;
                }
            }

            function main(): void {
                heap: Allocator = System::allocator();
                arena: Arena = Arena::init(heap);
                alloc: Allocator = arena.allocator();
                p: *Point = Point::create(alloc);
                p.x = 4;
                p.y = 5;
                print(p.sum());
                arena.deinit();
            }
            "#,
        )
        .unwrap();

        assert_eq!(stdout, "9\n");
    }

    #[test]
    fn runs_compiled_function_returning_pointer_program() {
        let stdout = compile_and_run(
            r#"
            struct Point {
                x: int;
                y: int;
            }

            function create_point(alloc: Allocator): *Point {
                point: *Point = Point::create(alloc);
                point.x = 11;
                point.y = 31;
                return point;
            }

            function main(): void {
                heap: Allocator = System::allocator();
                point: *Point = create_point(heap);
                print(point.x);
                print(point.y);
                heap.destroy(point);
            }
            "#,
        )
        .unwrap();

        assert_eq!(stdout, "11\n31\n");
    }

    #[test]
    fn runs_compiled_generic_struct_and_function_program() {
        let stdout = compile_and_run(
            r#"
            struct Box[T] {
                value: T;
            }

            attach[T] Box[T] {
                function get(self): T {
                    return self.value;
                }
            }

            function wrap[T](value: T): Box[T] {
                return Box[T] { value: value };
            }

            function unwrap[T](box: Box[T]): T {
                return box.get();
            }

            function main(): void {
                int_box: Box[int] = wrap(41);
                string_box: Box[string] = wrap("done");
                print(unwrap(int_box) + 1);
                print(unwrap(string_box));
            }
            "#,
        )
        .unwrap();

        assert_eq!(stdout, "42\ndone\n");
    }

    #[test]
    fn runs_compiled_generic_static_attached_function_program() {
        let stdout = compile_and_run(
            r#"
            struct Box[T] {
                value: T;
            }

            attach[T] Box[T] {
                function wrap(value: T): Box[T] {
                    return Box[T] { value: value };
                }

                function get(self): T {
                    return self.value;
                }
            }

            function main(): void {
                int_box: Box[int] = Box[int]::wrap(41);
                string_box: Box[string] = Box[string]::wrap("ok");
                print(int_box.get() + 1);
                print(string_box.get());
            }
            "#,
        )
        .unwrap();

        assert_eq!(stdout, "42\nok\n");
    }

    #[test]
    fn runs_compiled_nested_generic_program() {
        let stdout = compile_and_run(
            r#"
            struct Box[T] {
                value: T;
            }

            function wrap[T](value: T): Box[T] {
                return Box[T] { value: value };
            }

            function main(): void {
                inner: Box[int] = wrap(7);
                outer: Box[Box[int]] = wrap(inner);
                print(outer.value.value);
            }
            "#,
        )
        .unwrap();

        assert_eq!(stdout, "7\n");
    }

    #[test]
    fn runs_compiled_slice_allocator_program() {
        let stdout = compile_and_run(
            r#"
            function main(): void {
                heap: Allocator = System::allocator();
                arena: Arena = Arena::init(heap);
                alloc: Allocator = arena.allocator();
                values: []int = []int::alloc(alloc, 3);
                values[1] = 7;
                print(values.len);
                print(values[1]);
                arena.deinit();
            }
            "#,
        )
        .unwrap();

        assert_eq!(stdout, "3\n7\n");
    }

    #[test]
    fn runs_compiled_allocator_destroy_and_free_program() {
        let stdout = compile_and_run(
            r#"
            struct Point {
                x: int;
            }

            function main(): void {
                heap: Allocator = System::allocator();
                p: *Point = Point::create(heap);
                p.x = 12;
                print(p.x);
                heap.destroy(p);

                values: []int = []int::alloc(heap, 2);
                values[0] = 3;
                print(values[0]);
                heap.free(values);
            }
            "#,
        )
        .unwrap();

        assert_eq!(stdout, "12\n3\n");
    }

    #[test]
    fn runs_compiled_const_slice_copy_program() {
        let stdout = compile_and_run(
            r#"
            function copy_into(const dst: []int, src: []const int): void {
                for (i: int = 0; i < src.len; i = i + 1) {
                    dst[i] = src[i];
                }
            }

            function main(): void {
                heap: Allocator = System::allocator();
                dst: []int = []int::alloc(heap, 2);
                src: []int = []int::alloc(heap, 2);
                src[0] = 7;
                src[1] = 11;
                copy_into(dst, src);
                print(dst[0]);
                print(dst[1]);
                heap.free(dst);
                heap.free(src);
            }
            "#,
        )
        .unwrap();

        assert_eq!(stdout, "7\n11\n");
    }

    #[test]
    fn rejects_reassigning_const_variable() {
        let result = compile_and_run(
            r#"
            function main(): void {
                const answer: int = 41;
                answer = 42;
            }
            "#,
        );

        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("cannot assign to const binding `answer`"));
    }

    #[test]
    fn rejects_assigning_to_const_struct_field() {
        let result = compile_and_run(
            r#"
            struct Counter {
                const value: int;
            }

            attach Counter {
                function reset(mut self): void {
                    self.value = 0;
                }
            }

            function main(): void {
                counter: Counter = Counter { value: 7 };
                counter.reset();
            }
            "#,
        );

        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("cannot assign through const-qualified target `int`"));
    }

    #[test]
    fn rejects_reassigning_const_parameter() {
        let result = compile_and_run(
            r#"
            function bump(const n: int): int {
                n = n + 1;
                return n;
            }

            function main(): void {
                print(bump(1));
            }
            "#,
        );

        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("cannot assign to const binding `n`"));
    }

    #[test]
    fn rejects_writing_through_const_slice() {
        let result = compile_and_run(
            r#"
            function overwrite(values: []const int): void {
                values[0] = 7;
            }

            function main(): void {
                heap: Allocator = System::allocator();
                values: []int = []int::alloc(heap, 1);
                overwrite(values);
                heap.free(values);
            }
            "#,
        );

        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("cannot assign through const-qualified target `int`"));
    }

    #[test]
    fn rejects_writing_through_const_pointer() {
        let result = compile_and_run(
            r#"
            struct Point {
                x: int;
            }

            function set_x(point: *const Point): void {
                point.x = 9;
            }

            function main(): void {
                heap: Allocator = System::allocator();
                point: *Point = Point::create(heap);
                set_x(point);
                heap.destroy(point);
            }
            "#,
        );

        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("cannot assign through const-qualified target `int`"));
    }

    #[test]
    fn runs_compiled_arena_destroy_and_free_program() {
        let stdout = compile_and_run(
            r#"
            struct Point {
                x: int;
            }

            function main(): void {
                heap: Allocator = System::allocator();
                arena: Arena = Arena::init(heap);
                alloc: Allocator = arena.allocator();

                p: *Point = Point::create(alloc);
                p.x = 2;
                print(p.x);
                alloc.destroy(p);

                values: []int = []int::alloc(alloc, 2);
                values[1] = 8;
                print(values[1]);
                alloc.free(values);

                arena.reset();
                arena.deinit();
            }
            "#,
        )
        .unwrap();

        assert_eq!(stdout, "2\n8\n");
    }

    #[test]
    fn compiles_generic_enum_to_ir() {
        let program = ast::parse(
            r#"
            enum Option[T] {
                None;
                Some(T);
            }

            function main(): void {
                value: Option[int] = Option[int]::Some(7);
                match (value) {
                    case None: {
                        print(0);
                    }
                    case Some(v): {
                        print(v);
                    }
                }
            }
            "#,
        );
        let program = monomorphize::prepare_program(&program).unwrap();

        let ir = compile_to_llvm_ir(&program).unwrap();
        assert!(ir.contains("%enum.Option__int = type { i32, i32 }"));
        assert!(ir.contains("switch i32"));
    }

    #[test]
    fn runs_compiled_generic_enum_match_program() {
        let stdout = compile_and_run(
            r#"
            enum Option[T] {
                None;
                Some(T);
            }

            function unwrap(value: Option[int]): int {
                match (value) {
                    case None: {
                        return 0;
                    }
                    case Some(v): {
                        return v + 1;
                    }
                }
            }

            function main(): void {
                print(unwrap(Option[int]::None()));
                print(unwrap(Option[int]::Some(41)));
            }
            "#,
        )
        .unwrap();

        assert_eq!(stdout, "0\n42\n");
    }

    #[test]
    fn runs_compiled_multi_payload_enum_match_program() {
        let stdout = compile_and_run(
            r#"
            enum PairOrNone[A, B] {
                None;
                Pair(A, B);
            }

            function unwrap(value: PairOrNone[int, int]): int {
                match (value) {
                    case None: {
                        return 0;
                    }
                    case Pair(a, b): {
                        return a + b;
                    }
                }
            }

            function main(): void {
                print(unwrap(PairOrNone[int, int]::None()));
                print(unwrap(PairOrNone[int, int]::Pair(20, 22)));
            }
            "#,
        )
        .unwrap();

        assert_eq!(stdout, "0\n42\n");
    }

    #[test]
    fn runs_compiled_imported_generic_enum_module_program() {
        let stdout = compile_project_and_run(
            &[
                (
                    "std/option.skunk",
                    r#"
                    module std.option;

                    enum Option[T] {
                        None;
                        Some(T);
                    }

                    function wrap[T](value: T): Option[T] {
                        return Option[T]::Some(value);
                    }
                    "#,
                ),
                (
                    "main.skunk",
                    r#"
                    import std.option;

                    function main(): void {
                        value: Option[int] = wrap(9);
                        match (value) {
                            case None: {
                                print(0);
                            }
                            case Some(v): {
                                print(v);
                            }
                        }
                    }
                    "#,
                ),
            ],
            "main.skunk",
        )
        .unwrap();

        assert_eq!(stdout, "9\n");
    }

    #[test]
    fn runs_compiled_trait_bound_program() {
        let stdout = compile_and_run(
            r#"
            trait Writer {
                function write(mut self, value: int): int;
            }

            trait Resettable {
                function reset(mut self): void;
            }

            struct Counter {
                value: int;
            }

            conform Writer for Counter {
                function write(mut self, value: int): int {
                    self.value = self.value + value;
                    return self.value;
                }
            }

            conform Resettable for Counter {
                function reset(mut self): void {
                    self.value = 0;
                }
            }

            function use_counter[T: Writer + Resettable](counter: *T): int {
                counter.reset();
                return counter.write(41);
            }

            function main(): void {
                heap: Allocator = System::allocator();
                counter: *Counter = Counter::create(heap);
                counter.value = 9;
                print(use_counter(counter));
                heap.destroy(counter);
            }
            "#,
        )
        .unwrap();

        assert_eq!(stdout, "41\n");
    }

    #[test]
    fn runs_compiled_where_clause_trait_bound_program() {
        let stdout = compile_and_run(
            r#"
            trait Writer {
                function write(mut self, value: int): int;
            }

            trait Resettable {
                function reset(mut self): void;
            }

            struct Counter {
                value: int;
            }

            conform Writer for Counter {
                function write(mut self, value: int): int {
                    self.value = self.value + value;
                    return self.value;
                }
            }

            conform Resettable for Counter {
                function reset(mut self): void {
                    self.value = 0;
                }
            }

            function use_counter[T](counter: *T): int
            where T: Writer + Resettable {
                counter.reset();
                return counter.write(41);
            }

            function main(): void {
                heap: Allocator = System::allocator();
                counter: *Counter = Counter::create(heap);
                counter.value = 9;
                print(use_counter(counter));
                heap.destroy(counter);
            }
            "#,
        )
        .unwrap();

        assert_eq!(stdout, "41\n");
    }

    #[test]
    fn runs_compiled_shape_bound_program() {
        let stdout = compile_and_run(
            r#"
            shape WriterLike {
                function write(mut self, value: int): int;
            }

            struct BufferWriter {
                value: int;
            }

            attach BufferWriter {
                function write(mut self, value: int): int {
                    self.value = self.value + value;
                    return self.value;
                }
            }

            function use_writer_like[T: WriterLike](writer: *T): int {
                return writer.write(5);
            }

            function main(): void {
                heap: Allocator = System::allocator();
                writer: *BufferWriter = BufferWriter::create(heap);
                writer.value = 7;
                print(use_writer_like(writer));
                heap.destroy(writer);
            }
            "#,
        )
        .unwrap();

        assert_eq!(stdout, "12\n");
    }

    #[test]
    fn runs_compiled_trait_object_dynamic_dispatch_program() {
        let stdout = compile_and_run(
            r#"
            trait Writer {
                function write(mut self, value: int): int;
            }

            struct Counter {
                value: int;
            }

            conform Writer for Counter {
                function write(mut self, value: int): int {
                    self.value = self.value + value;
                    return self.value;
                }
            }

            function main(): void {
                writer: Writer = Counter { value: 1 };
                print(writer.write(4));
                print(writer.write(7));
            }
            "#,
        )
        .unwrap();

        assert_eq!(stdout, "5\n12\n");
    }

    #[test]
    fn runs_compiled_trait_object_borrowed_local_program() {
        let stdout = compile_and_run(
            r#"
            trait Writer {
                function write(mut self, value: int): int;
            }

            struct IntWriter {
                i: int;
            }

            attach IntWriter {
                function get_i(self): int {
                    return self.i;
                }
            }

            conform Writer for IntWriter {
                function write(mut self, value: int): int {
                    self.i = value;
                    return self.i;
                }
            }

            function main(): void {
                iw: IntWriter = IntWriter { i: 0 };
                w: Writer = iw;
                w.write(1);
                print(iw.get_i());
                print(iw.i);
            }
            "#,
        )
        .unwrap();

        assert_eq!(stdout, "1\n1\n");
    }

    #[test]
    fn runs_compiled_trait_object_parameter_program() {
        let stdout = compile_and_run(
            r#"
            trait Writer {
                function write(mut self, value: int): int;
            }

            struct Counter {
                value: int;
            }

            conform Writer for Counter {
                function write(mut self, value: int): int {
                    self.value = self.value + value;
                    return self.value;
                }
            }

            function use_writer(writer: Writer): int {
                return writer.write(9);
            }

            function main(): void {
                writer: Writer = Counter { value: 3 };
                print(use_writer(writer));
                print(writer.write(1));
            }
            "#,
        )
        .unwrap();

        assert_eq!(stdout, "12\n13\n");
    }

    #[test]
    fn runs_compiled_safe_reference_program() {
        let stdout = compile_and_run(
            r#"
            struct Counter {
                value: int;
            }

            attach Counter {
                function get(self): int {
                    return self.value;
                }
            }

            function bump(counter: &mut Counter): void {
                counter.value = counter.value + 1;
            }

            function read(counter: &Counter): int {
                return counter.get();
            }

            function main(): void {
                counter: Counter = Counter { value: 4 };
                bump(&mut counter);

                reader: &Counter = &counter;
                print(read(reader));
                print(reader.*.value);
            }
            "#,
        )
        .unwrap();

        assert_eq!(stdout, "5\n5\n");
    }

    #[test]
    fn rejects_mutating_through_immutable_reference() {
        let result = compile_and_run(
            r#"
            struct Counter {
                value: int;
            }

            function bump(counter: &Counter): void {
                counter.value = counter.value + 1;
            }

            function main(): void {
                counter: Counter = Counter { value: 4 };
                bump(&counter);
            }
            "#,
        );

        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("cannot assign through const-qualified target"));
    }

    #[test]
    fn runs_compiled_trait_default_method_program() {
        let stdout = compile_and_run(
            r#"
            trait Writer {
                function write(mut self, value: int): int;

                function write_twice(mut self, value: int): int {
                    self.write(value);
                    return self.write(value);
                }
            }

            struct Counter {
                value: int;
            }

            conform Writer for Counter {
                function write(mut self, value: int): int {
                    self.value = self.value + value;
                    return self.value;
                }
            }

            function use_counter[T: Writer](counter: *T): int {
                return counter.write_twice(3);
            }

            function main(): void {
                heap: Allocator = System::allocator();
                counter: *Counter = Counter::create(heap);
                counter.value = 1;
                print(use_counter(counter));

                writer: Writer = Counter { value: 10 };
                print(writer.write_twice(2));

                heap.destroy(counter);
            }
            "#,
        )
        .unwrap();

        assert_eq!(stdout, "7\n14\n");
    }

    #[test]
    fn runs_compiled_supertrait_program() {
        let stdout = compile_and_run(
            r#"
            trait Readable {
                function value(self): int;
            }

            trait Writer: Readable {
                function write(mut self, value: int): int;
            }

            struct Counter {
                value: int;
            }

            conform Writer for Counter {
                function value(self): int {
                    return self.value;
                }

                function write(mut self, value: int): int {
                    self.value = self.value + value;
                    return self.value;
                }
            }

            function read_once[T: Readable](counter: *T): int {
                return counter.value();
            }

            function main(): void {
                heap: Allocator = System::allocator();
                counter: *Counter = Counter::create(heap);
                counter.value = 5;
                print(read_once(counter));

                readable: Readable = Counter { value: 9 };
                print(readable.value());

                heap.destroy(counter);
            }
            "#,
        )
        .unwrap();

        assert_eq!(stdout, "5\n9\n");
    }

    #[test]
    fn runs_compiled_function_returning_trait_object_program() {
        let stdout = compile_and_run(
            r#"
            trait Writer {
                function write(mut self, value: int): int;
            }

            struct Counter {
                value: int;
            }

            conform Writer for Counter {
                function write(mut self, value: int): int {
                    self.value = self.value + value;
                    return self.value;
                }
            }

            function create_writer(): Writer {
                return Counter { value: 10 };
            }

            function main(): void {
                writer: Writer = create_writer();
                print(writer.write(2));
                print(writer.write(5));
            }
            "#,
        )
        .unwrap();

        assert_eq!(stdout, "12\n17\n");
    }

    #[test]
    fn rejects_trait_object_assignment_for_non_impl_type() {
        let result = compile_and_run(
            r#"
            trait Writer {
                function write(mut self, value: int): int;
            }

            struct Counter {
                value: int;
            }

            conform Writer for Counter {
                function write(mut self, value: int): int {
                    self.value = self.value + value;
                    return self.value;
                }
            }

            struct Plain {
                value: int;
            }

            function main(): void {
                writer: Writer = Plain { value: 1 };
                print(writer.write(1));
            }
            "#,
        );

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Writer"));
    }

    #[test]
    fn runs_compiled_unsafe_address_of_and_dereference_program() {
        let stdout = compile_and_run(
            r#"
            function main(): void {
                value: int = 41;
                unsafe {
                    ptr: *int = &value;
                    print(ptr.*);
                    ptr.* = 42;
                }
                print(value);
            }
            "#,
        )
        .unwrap();

        assert_eq!(stdout, "41\n42\n");
    }

    #[test]
    fn runs_compiled_size_of_and_align_of_program() {
        let stdout = compile_and_run(
            r#"
            struct Pair {
                left: int;
                right: int;
            }

            function main(): void {
                print(int::size_of());
                print(int::align_of());
                print(Pair::size_of());
                print(Pair::align_of());
            }
            "#,
        )
        .unwrap();

        assert_eq!(stdout, "4\n4\n8\n4\n");
    }

    #[test]
    fn runs_compiled_unsafe_memory_copy_and_set_program() {
        let stdout = compile_and_run(
            r#"
            function main(): void {
                src: int = 123;
                dst: int = 0;
                bytes: [4]byte;
                unsafe {
                    Memory::copy(*byte::cast(&dst), *byte::cast(&src), int::size_of());
                    Memory::set(&bytes[0], 7, 4);
                }
                print(dst);
                print(bytes[0]);
                print(bytes[3]);
            }
            "#,
        )
        .unwrap();

        assert_eq!(stdout, "123\n7\n7\n");
    }

    #[test]
    fn runs_compiled_headless_window_program() {
        let stdout = compile_and_run_with_env(
            r#"
            function main(): void {
                window: Window = Window::create(96, 64, "Headless");
                print(window.is_open());
                window.poll();
                window.clear(Color::black());
                window.draw_rect(4.0, 6.0, 18.0, 10.0, Color::white());
                window.present();
                print(Keyboard::is_down(window, 'w'));
                print(window.delta_time() > 0.0);
                window.close();
                print(window.is_open());
                window.deinit();
            }
            "#,
            &[("SKUNK_WINDOW_HEADLESS", "1")],
        )
        .unwrap();

        assert_eq!(stdout, "true\nfalse\ntrue\nfalse\n");
    }

    #[test]
    fn runs_headless_pong_example() {
        let stdout = compile_and_run_with_env(
            include_str!("../examples/pong.skunk"),
            &[("SKUNK_WINDOW_HEADLESS", "1")],
        )
        .unwrap();

        assert_eq!(stdout, "0\n5\n");
    }

    #[test]
    fn runs_compiled_unsafe_byte_pointer_offset_program() {
        let stdout = compile_and_run(
            r#"
            function main(): void {
                bytes: [4]byte;
                unsafe {
                    start: *byte = &bytes[0];
                    second: *byte = *byte::offset(start, 1);
                    second.* = 9;
                }
                print(bytes[1]);
            }
            "#,
        )
        .unwrap();

        assert_eq!(stdout, "9\n");
    }

    #[test]
    fn rejects_address_of_outside_unsafe_block() {
        let result = compile_and_run(
            r#"
            function main(): void {
                value: int = 1;
                ptr: *int = &value;
                print(ptr.*);
            }
            "#,
        );

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("address-of"));
    }

    #[test]
    fn rejects_memory_copy_outside_unsafe_block() {
        let result = compile_and_run(
            r#"
            function main(): void {
                src: int = 1;
                dst: int = 0;
                Memory::copy(*byte::cast(&dst), *byte::cast(&src), int::size_of());
            }
            "#,
        );

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("unsafe"));
    }

    #[test]
    fn runs_compiled_generic_impl_target_program() {
        let stdout = compile_and_run(
            r#"
            trait SizedThing {
                function size(self): int;
            }

            struct Box[T] {
                value: T;
            }

            conform[T] SizedThing for Box[T] {
                function size(self): int {
                    return 1;
                }
            }

            function measure[T: SizedThing](value: T): int {
                return value.size();
            }

            function main(): void {
                print(measure(Box[int] { value: 7 }));
                print(measure(Box[string] { value: "x" }));
            }
            "#,
        )
        .unwrap();

        assert_eq!(stdout, "1\n1\n");
    }

    #[test]
    fn runs_compiled_explicit_generic_function_call_program() {
        let stdout = compile_and_run(
            r#"
            function id[T](value: T): T {
                return value;
            }

            function main(): void {
                print(id[int](42));
            }
            "#,
        )
        .unwrap();

        assert_eq!(stdout, "42\n");
    }

    #[test]
    fn rejects_trait_impl_with_receiver_mutability_mismatch() {
        let result = compile_and_run(
            r#"
            trait Writer {
                function write(mut self, value: int): int;
            }

            struct Counter {
                value: int;
            }

            conform Writer for Counter {
                function write(self, value: int): int {
                    return self.value + value;
                }
            }

            function main(): void {}
            "#,
        );

        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("trait method `Writer.write` expects"));
    }

    #[test]
    fn rejects_call_when_trait_bound_is_not_implemented() {
        let result = compile_and_run(
            r#"
            trait Writer {
                function write(mut self, value: int): int;
            }

            struct Counter {
                value: int;
            }

            attach Counter {
                function write(mut self, value: int): int {
                    self.value = self.value + value;
                    return self.value;
                }
            }

            function use_counter[T: Writer](counter: *T): int {
                return counter.write(41);
            }

            function main(): void {
                heap: Allocator = System::allocator();
                counter: *Counter = Counter::create(heap);
                print(use_counter(counter));
                heap.destroy(counter);
            }
            "#,
        );

        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("generic function `use_counter` requires `T` to implement trait `Writer`"));
    }

    #[test]
    fn runs_compiled_imported_module_with_exported_api_and_private_helper_program() {
        let stdout = compile_project_and_run(
            &[
                (
                    "std/math.skunk",
                    r#"
                    module std.math;

                    function helper(n: int): int {
                        return n + 1;
                    }

                    export function inc(n: int): int {
                        return helper(n);
                    }
                    "#,
                ),
                (
                    "main.skunk",
                    r#"
                    import std.math;

                    function main(): void {
                        print(inc(41));
                    }
                    "#,
                ),
            ],
            "main.skunk",
        )
        .unwrap();

        assert_eq!(stdout, "42\n");
    }

    #[test]
    fn rejects_private_imported_module_symbols() {
        let result = compile_project_and_run(
            &[
                (
                    "std/math.skunk",
                    r#"
                    module std.math;

                    function helper(n: int): int {
                        return n + 1;
                    }

                    export function inc(n: int): int {
                        return helper(n);
                    }
                    "#,
                ),
                (
                    "main.skunk",
                    r#"
                    import std.math;

                    function main(): void {
                        print(helper(41));
                    }
                    "#,
                ),
            ],
            "main.skunk",
        );

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("unknown function `helper`"));
    }

    #[test]
    fn runs_compiled_imported_module_program() {
        let stdout = compile_project_and_run(
            &[
                (
                    "std/math.skunk",
                    r#"
                    module std.math;

                    function inc(n: int): int {
                        return n + 1;
                    }
                    "#,
                ),
                (
                    "main.skunk",
                    r#"
                    import std.math;

                    function main(): void {
                        print(inc(41));
                    }
                    "#,
                ),
            ],
            "main.skunk",
        )
        .unwrap();

        assert_eq!(stdout, "42\n");
    }

    #[test]
    fn runs_compiled_imported_generic_module_program() {
        let stdout = compile_project_and_run(
            &[
                (
                    "std/box.skunk",
                    r#"
                    module std.box;

                    struct Box[T] {
                        value: T;
                    }

                    function wrap[T](value: T): Box[T] {
                        return Box[T] { value: value };
                    }
                    "#,
                ),
                (
                    "main.skunk",
                    r#"
                    import std.box;

                    function main(): void {
                        value: Box[int] = wrap(7);
                        print(value.value);
                    }
                    "#,
                ),
            ],
            "main.skunk",
        )
        .unwrap();

        assert_eq!(stdout, "7\n");
    }
}
