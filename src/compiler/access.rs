//! Member access, method dispatch, closures, and function calls.

use super::*;

impl<'a> FunctionCompiler<'a> {
    /// Lowers an access chain as a value, giving method and `.len` access their
    /// specialized handling before falling back to pointer resolution.
    pub(super) fn compile_access_expr(&mut self, nodes: &[Node]) -> Result<ExprValue, String> {
        if let Some(expr) = self.try_compile_member_call(nodes)? {
            return Ok(expr);
        }
        if let Some(expr) = self.compile_length_member(nodes)? {
            return Ok(expr);
        }
        let (ptr, llvm_type) = self.resolve_access_ptr(nodes)?;
        self.load_from_ptr(&ptr, &llvm_type)
    }

    /// Resolves and lowers concrete, trait-object, and trait-intersection method
    /// dispatch for the final step of an access chain.
    pub(super) fn try_compile_member_call(
        &mut self,
        nodes: &[Node],
    ) -> Result<Option<ExprValue>, String> {
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

        if let LlvmType::TraitIntersection(trait_names) = receiver_type {
            let mut found = Vec::new();
            for (trait_index, trait_name) in trait_names.iter().enumerate() {
                let trait_layout = self
                    .traits
                    .get(trait_name)
                    .ok_or_else(|| format!("unknown trait `{}` in LLVM backend", trait_name))?;
                if let Some((method_index, method)) = trait_layout
                    .methods
                    .iter()
                    .enumerate()
                    .find(|(_, method)| method.name == method_name)
                {
                    found.push((
                        trait_index,
                        trait_name.clone(),
                        method_index,
                        method.clone(),
                    ));
                }
            }
            if found.is_empty() {
                return Err(format!(
                    "unknown method `{}` on trait intersection",
                    method_name
                ));
            }
            if found.len() > 1 {
                return Err(format!(
                    "ambiguous method `{}` on trait intersection",
                    method_name
                ));
            }
            let (trait_index, trait_name, method_index, method) = found.pop().unwrap();
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
                "{} = extractvalue {} {}, {}",
                vtable_ptr,
                receiver_value.llvm_type.ir(),
                receiver_value.value,
                trait_index + 1
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
                    llvm_type: method.return_type,
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

        let (nominal_name, nominal_type) = match receiver_type {
            LlvmType::Struct(name) => (name.clone(), LlvmType::Struct(name)),
            LlvmType::Enum(name) => (name.clone(), LlvmType::Enum(name)),
            LlvmType::Reference { target_type, .. } => match *target_type {
                LlvmType::Struct(name) => (name.clone(), LlvmType::Struct(name)),
                LlvmType::Enum(name) => (name.clone(), LlvmType::Enum(name)),
                other => {
                    return Err(format!(
                        "method `{}` requires a nominal receiver, found `{}`",
                        method_name,
                        other.ir()
                    ))
                }
            },
            LlvmType::Pointer { target_type } => match *target_type {
                LlvmType::Struct(name) => (name.clone(), LlvmType::Struct(name)),
                LlvmType::Enum(name) => (name.clone(), LlvmType::Enum(name)),
                other => {
                    return Err(format!(
                        "method `{}` requires a nominal receiver, found `{}`",
                        method_name,
                        other.ir()
                    ))
                }
            },
            other => {
                return Err(format!(
                    "method `{}` requires a nominal receiver, found `{}`",
                    method_name,
                    other.ir()
                ))
            }
        };

        let signature_key = format!("{}::{}", nominal_name, method_name);
        let signature = self
            .signatures
            .get(&signature_key)
            .cloned()
            .ok_or_else(|| format!("unknown method `{}` on `{}`", method_name, nominal_name))?;

        let receiver_ptr = match self.resolve_access_ptr(receiver_nodes) {
            Ok((ptr, llvm_type)) => {
                if is_pointer_like_llvm_type(&llvm_type) {
                    self.load_from_ptr(&ptr, &llvm_type)?.value
                } else {
                    ptr
                }
            }
            Err(_) => {
                let temp_var = self.emit_heap_alloc(nominal_type, "receiver_tmp");
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

    pub(super) fn compile_length_member(
        &mut self,
        nodes: &[Node],
    ) -> Result<Option<ExprValue>, String> {
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

    pub(super) fn access_base_type(&self, nodes: &[Node]) -> Result<LlvmType, String> {
        match nodes.first() {
            Some(Node::Identifier(name)) => self
                .lookup_local(name)
                .map(|local| local.llvm_type.clone())
                .ok_or_else(|| format!("unknown variable `{}` in LLVM backend", name)),
            _ => Err("LLVM backend currently supports only identifier-rooted access".to_string()),
        }
    }

    pub(super) fn apply_access_type_step(
        &self,
        current: LlvmType,
        node: &Node,
    ) -> Result<LlvmType, String> {
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

    /// Walks an addressable access chain and returns the final storage pointer
    /// together with its LLVM value type.
    pub(super) fn resolve_access_ptr(
        &mut self,
        nodes: &[Node],
    ) -> Result<(String, LlvmType), String> {
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
                Node::Dereference { .. } => match current_type.clone() {
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
                },
                Node::ArrayAccess { coordinates } => {
                    for coordinate in coordinates {
                        let index = self.compile_expr(coordinate)?;
                        let index = self.coerce_expr(index, &LlvmType::I64, "array index")?;
                        match current_type.clone() {
                            LlvmType::Array { elem_type, len } => {
                                let length = ExprValue {
                                    llvm_type: LlvmType::I64,
                                    value: len.to_string(),
                                };
                                self.emit_index_bounds_check(&index, &length)?;
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
                                let length = self.extract_slice_len(&slice_value)?;
                                self.emit_index_bounds_check(&index, &length)?;
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

    pub(super) fn struct_field_info(
        &self,
        struct_name: &str,
        field_name: &str,
    ) -> Option<(usize, LlvmType)> {
        self.structs.get(struct_name).and_then(|layout| {
            layout
                .fields
                .iter()
                .enumerate()
                .find(|(_, (name, _))| name == field_name)
                .map(|(index, (_, llvm_type))| (index, llvm_type.clone()))
        })
    }

    pub(super) fn visible_locals(&self) -> Vec<(String, LocalVar)> {
        let mut locals = BTreeMap::<String, LocalVar>::new();
        for scope in &self.scopes {
            for (name, local) in scope {
                locals.insert(name.clone(), local.clone());
            }
        }
        locals.into_iter().collect()
    }

    /// Lifts a lambda into a generated function, builds its capture environment,
    /// and returns the closure pair consumed by indirect calls.
    pub(super) fn compile_lambda_expr(
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
                return_type: Box::new(llvm_type(
                    return_type,
                    self.structs,
                    self.enums,
                    self.traits,
                )?),
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

        let dependencies = FunctionCompilerDependencies {
            signatures: self.signatures,
            structs: self.structs,
            enums: self.enums,
            traits: self.traits,
            trait_vtables: self.trait_vtables,
            globals: self.globals,
            extra_type_decls: self.extra_type_decls,
            extra_function_irs: self.extra_function_irs,
            lambda_counter: self.lambda_counter,
        };
        let nested_compiler = FunctionCompiler::new(
            &symbol_name,
            lambda_return_type.clone(),
            dependencies,
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

    /// Emits a direct call to a known function signature and applies any later
    /// curried argument groups to the returned closure.
    pub(super) fn compile_direct_call(
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

    /// Invokes a closure pair through its function pointer and environment.
    pub(super) fn compile_closure_call(
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

    /// Selects direct or closure dispatch for a source-level function call.
    pub(super) fn compile_function_call(
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
}
