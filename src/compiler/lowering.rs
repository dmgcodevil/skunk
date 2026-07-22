//! Core statement, control-flow, expression, and aggregate lowering.

use super::*;

impl<'a> FunctionCompiler<'a> {
    /// Creates the per-function lowering context that owns local scopes,
    /// temporary names, deferred work, and generated auxiliary IR.
    pub(super) fn new(
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
            deferred_scopes: vec![Vec::new()],
            lines: Vec::new(),
            temp_counter: 0,
            label_counter: 0,
            terminated: false,
            unsafe_depth: 0,
        }
    }

    /// Lowers one complete function body, including parameter storage and the
    /// implicit return for a fall-through `void` function.
    pub(super) fn compile(
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
                LlvmType::Void => {
                    self.compile_current_scope_defers()?;
                    self.emit_line("ret void".to_string());
                }
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

    pub(super) fn compile_statements(&mut self, statements: &[Node]) -> Result<(), String> {
        for statement in statements {
            if self.terminated {
                break;
            }
            self.compile_statement(statement)?;
        }
        Ok(())
    }

    /// Lowers one checked statement and updates control-flow termination and
    /// lexical-scope state.
    pub(super) fn compile_statement(&mut self, node: &Node) -> Result<(), String> {
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
                if !self.terminated {
                    self.compile_current_scope_defers()?;
                }
                self.pop_scope();
                Ok(())
            }
            Node::UnsafeBlock { statements } => {
                self.push_scope();
                self.enter_unsafe();
                let result = self.compile_statements(statements).and_then(|_| {
                    if !self.terminated {
                        self.compile_current_scope_defers()?;
                    }
                    Ok(())
                });
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
            Node::Defer(expression) => {
                let mut locals = HashMap::new();
                for scope in &self.scopes {
                    locals.extend(scope.clone());
                }
                self.deferred_scopes
                    .last_mut()
                    .expect("defer scope stack should never be empty")
                    .push(DeferredExpression {
                        expression: expression.as_ref().clone(),
                        locals,
                        unsafe_depth: self.unsafe_depth,
                    });
                Ok(())
            }
            Node::Return(value) => {
                match value {
                    Some(value) => {
                        let return_type = self.return_type.clone();
                        let expr = self.compile_expr_with_expected(value, Some(&return_type))?;
                        let return_type = self.return_type.clone();
                        let expr = self.coerce_expr(expr, &return_type, "return")?;
                        self.compile_all_scope_defers()?;
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
                        self.compile_all_scope_defers()?;
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

    /// Lowers an if/else-if/else chain and merges its reachable control-flow
    /// paths at a shared continuation block.
    pub(super) fn compile_if(
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
        if !self.terminated {
            self.compile_current_scope_defers()?;
        }
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
            if !self.terminated {
                self.compile_current_scope_defers()?;
            }
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

    /// Lowers pattern matching into tag/value tests, case-local bindings, and
    /// a shared continuation block.
    pub(super) fn compile_match(
        &mut self,
        value: &Node,
        cases: &[ast::MatchCase],
    ) -> Result<(), String> {
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
                    if !self.terminated {
                        self.compile_current_scope_defers()?;
                    }
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
                if !self.terminated {
                    self.compile_current_scope_defers()?;
                }
                self.pop_scope();
                Ok(())
            }
            _ => Err("match expects an enum or struct value in LLVM backend".to_string()),
        }
    }

    /// Lowers a range-based loop with explicit condition, body, step, and exit
    /// blocks.
    pub(super) fn compile_for(
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
        self.push_scope();
        self.compile_statements(body)?;
        if !self.terminated {
            self.compile_current_scope_defers()?;
        }
        self.pop_scope();
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
        self.compile_current_scope_defers()?;
        self.pop_scope();
        Ok(())
    }

    pub(super) fn compile_print(&mut self, value: &Node) -> Result<(), String> {
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
            | LlvmType::TraitIntersection(_)
            | LlvmType::Reference { .. }
            | LlvmType::Pointer { .. } => {
                Err("cannot print a pointer-like value directly".to_string())
            }
            LlvmType::Struct(_) => Err("cannot print a struct value directly".to_string()),
            LlvmType::Enum(_) => Err("cannot print an enum value directly".to_string()),
            LlvmType::Union(_) => Err("cannot print a union value directly".to_string()),
            LlvmType::Function { .. } => Err("cannot print a function value directly".to_string()),
            LlvmType::Slice { .. } => Err("cannot print a slice value directly".to_string()),
            LlvmType::Array { .. } => Err("cannot print an array value directly".to_string()),
            LlvmType::Void => Err("cannot print a void value".to_string()),
        }
    }

    pub(super) fn compile_expr(&mut self, node: &Node) -> Result<ExprValue, String> {
        self.compile_expr_with_expected(node, None)
    }

    /// Compiles an expression while optionally using an expected target type to
    /// guide lowering decisions.
    ///
    /// This is the main expression entry point used by variable initializers,
    /// assignments, returns, and struct field construction.
    /// Lowers an expression using an optional expected LLVM type to guide
    /// literals, generic runtime representations, and coercion decisions.
    pub(super) fn compile_expr_with_expected(
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
                    if !formatted.contains('.')
                        && !formatted.contains('e')
                        && !formatted.contains('E')
                    {
                        formatted.push_str(".0");
                    }
                    formatted
                },
            }),
            Node::Literal(Literal::Double(value)) => Ok(ExprValue {
                llvm_type: LlvmType::F64,
                value: {
                    let mut formatted = value.to_string();
                    if !formatted.contains('.')
                        && !formatted.contains('e')
                        && !formatted.contains('E')
                    {
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
                    Some(
                        LlvmType::TraitObject(_)
                        | LlvmType::TraitIntersection(_)
                        | LlvmType::Union(_),
                    ) => {
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
            Node::UnaryOp { operator, operand } => match operator {
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
            },
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

    pub(super) fn try_compile_borrowed_trait_object(
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

        let trait_value = self.trait_object_from_ptr(trait_name, &concrete_name, ptr, context)?;
        Ok(Some(trait_value))
    }

    /// Lowers a namespace-qualified static call, including built-in allocator,
    /// bounds, window, and pointer operations.
    pub(super) fn compile_static_function_call(
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
            Type::Custom(bounds_name) if bounds_name == "Bounds" => {
                if name != "check" {
                    return Err(format!(
                        "LLVM backend does not support static function call `Bounds::{}`",
                        name
                    ));
                }
                if arguments.len() != 2 {
                    return Err("Bounds::check expects index and length".to_string());
                }
                let index = self.compile_expr(&arguments[0])?;
                let length = self.compile_expr(&arguments[1])?;
                self.emit_index_bounds_check(&index, &length)?;
                Ok(ExprValue {
                    llvm_type: LlvmType::Void,
                    value: "void".to_string(),
                })
            }
            Type::Custom(memory_name) if memory_name == "Memory" => {
                if !self.unsafe_allowed() {
                    return Err(format!("Memory::{} requires an unsafe block", name));
                }
                match name {
                    "copy" => {
                        if arguments.len() != 3 {
                            return Err("Memory::copy expects dst, src, and byte count".to_string());
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
                        self.emit_line(format!("{} = and i32 {}, 255", masked, channel.value));
                        channels.push(masked);
                    }
                    let alpha = if name == "rgb" {
                        "255".to_string()
                    } else {
                        channels[3].clone()
                    };
                    let shifted_alpha = self.next_temp();
                    self.emit_line(format!("{} = shl i32 {}, 24", shifted_alpha, alpha));
                    let shifted_red = self.next_temp();
                    self.emit_line(format!("{} = shl i32 {}, 16", shifted_red, channels[0]));
                    let shifted_green = self.next_temp();
                    self.emit_line(format!("{} = shl i32 {}, 8", shifted_green, channels[1]));
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
                    self.emit_line(format!("{} = or i32 {}, {}", full, with_green, channels[2]));
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
                let width = self.compile_expr_with_expected(&arguments[0], Some(&LlvmType::I32))?;
                let width = self.coerce_expr(width, &LlvmType::I32, "Window::create width")?;
                let height =
                    self.compile_expr_with_expected(&arguments[1], Some(&LlvmType::I32))?;
                let height = self.coerce_expr(height, &LlvmType::I32, "Window::create height")?;
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
                let window =
                    self.coerce_expr(window, &LlvmType::Window, "Keyboard::is_down window")?;
                let key =
                    self.compile_expr_with_expected(&arguments[1], Some(&LlvmType::Char16))?;
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
                let is_variant = self.enums.get(enum_name).is_some_and(|layout| {
                    layout.variants.iter().any(|variant| variant.name == name)
                });
                if is_variant {
                    self.compile_enum_constructor(enum_name, name, arguments)
                } else {
                    let signature_key = format!("{}::{}", enum_name, name);
                    let signature =
                        self.signatures
                            .get(&signature_key)
                            .cloned()
                            .ok_or_else(|| {
                                format!(
                                    "unknown static function `{}` on `{}` in LLVM backend",
                                    name, enum_name
                                )
                            })?;
                    self.compile_direct_call(
                        &signature_key,
                        &signature,
                        arguments,
                        "static function",
                    )
                }
            }
            Type::Custom(struct_name)
                if name != "create" && self.structs.contains_key(struct_name) =>
            {
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

    /// Constructs the flattened runtime enum representation from a variant tag
    /// and its coerced payload values.
    pub(super) fn compile_enum_constructor(
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

    /// Allocates and initializes a fixed-size array literal value.
    pub(super) fn compile_array_literal(
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

    pub(super) fn compile_array_fill(
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

    /// Builds a slice header over freshly allocated literal element storage.
    pub(super) fn compile_slice_literal(
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

    pub(super) fn compile_slice_from_ptr(
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

    pub(super) fn build_slice_from_array_ptr(
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

    pub(super) fn build_slice_header_from_data_ptr(
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

        self.emit_slice_range_bounds_check(&start_value, &end_value, base_len)?;

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

    pub(super) fn load_from_ptr(
        &mut self,
        ptr: &str,
        llvm_type: &LlvmType,
    ) -> Result<ExprValue, String> {
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

    pub(super) fn extract_slice_data(&mut self, slice: &ExprValue) -> Result<String, String> {
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

    pub(super) fn extract_slice_len(&mut self, slice: &ExprValue) -> Result<ExprValue, String> {
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

    /// Emits the runtime guard used before array or slice element access.
    pub(super) fn emit_index_bounds_check(
        &mut self,
        index: &ExprValue,
        length: &ExprValue,
    ) -> Result<(), String> {
        let index = self.coerce_expr(index.clone(), &LlvmType::I64, "bounds-check index")?;
        let length = self.coerce_expr(length.clone(), &LlvmType::I64, "bounds-check length")?;
        let non_negative = self.next_temp();
        self.emit_line(format!(
            "{} = icmp sge i64 {}, 0",
            non_negative, index.value
        ));
        let below_length = self.next_temp();
        self.emit_line(format!(
            "{} = icmp slt i64 {}, {}",
            below_length, index.value, length.value
        ));
        let valid = self.next_temp();
        self.emit_line(format!(
            "{} = and i1 {}, {}",
            valid, non_negative, below_length
        ));

        let ok_label = self.next_label("bounds_ok");
        let panic_label = self.next_label("bounds_panic");
        self.emit_line(format!(
            "br i1 {}, label %{}, label %{}",
            valid, ok_label, panic_label
        ));
        self.emit_label(&panic_label);
        self.emit_line(format!(
            "call void @skunk_panic_index_out_of_bounds(i64 {}, i64 {})",
            index.value, length.value
        ));
        self.emit_line("unreachable".to_string());
        self.emit_label(&ok_label);
        Ok(())
    }

    /// Emits runtime guards for a half-open slice range and its source length.
    pub(super) fn emit_slice_range_bounds_check(
        &mut self,
        start: &ExprValue,
        end: &ExprValue,
        length: &ExprValue,
    ) -> Result<(), String> {
        let start = self.coerce_expr(start.clone(), &LlvmType::I64, "slice start")?;
        let end = self.coerce_expr(end.clone(), &LlvmType::I64, "slice end")?;
        let length = self.coerce_expr(length.clone(), &LlvmType::I64, "slice length")?;

        let start_non_negative = self.next_temp();
        self.emit_line(format!(
            "{} = icmp sge i64 {}, 0",
            start_non_negative, start.value
        ));
        let ordered = self.next_temp();
        self.emit_line(format!(
            "{} = icmp sle i64 {}, {}",
            ordered, start.value, end.value
        ));
        let end_in_bounds = self.next_temp();
        self.emit_line(format!(
            "{} = icmp sle i64 {}, {}",
            end_in_bounds, end.value, length.value
        ));
        let valid_start = self.next_temp();
        self.emit_line(format!(
            "{} = and i1 {}, {}",
            valid_start, start_non_negative, ordered
        ));
        let valid = self.next_temp();
        self.emit_line(format!(
            "{} = and i1 {}, {}",
            valid, valid_start, end_in_bounds
        ));

        let ok_label = self.next_label("slice_bounds_ok");
        let panic_label = self.next_label("slice_bounds_panic");
        self.emit_line(format!(
            "br i1 {}, label %{}, label %{}",
            valid, ok_label, panic_label
        ));
        self.emit_label(&panic_label);
        self.emit_line(format!(
            "call void @skunk_panic_slice_range_out_of_bounds(i64 {}, i64 {}, i64 {})",
            start.value, end.value, length.value
        ));
        self.emit_line("unreachable".to_string());
        self.emit_label(&ok_label);
        Ok(())
    }

    /// Lowers a typed struct literal into an LLVM aggregate value.
    ///
    /// Field expressions are compiled one by one and inserted into a
    /// zero-initialized aggregate using the resolved struct layout.
    /// Constructs a struct value in declaration field order after coercing each
    /// field expression to its layout type.
    pub(super) fn compile_struct_literal(
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
}
