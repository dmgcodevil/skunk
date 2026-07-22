//! Operators, coercions, runtime representations, scopes, and IR emission.

use super::*;

impl<'a> FunctionCompiler<'a> {
    /// Lowers arithmetic, comparison, and logical operators after promoting
    /// compatible numeric operands to a shared representation.
    pub(super) fn compile_binary_expr(
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

    pub(super) fn load_local(&mut self, name: &str) -> Result<ExprValue, String> {
        let local = self
            .lookup_local(name)
            .cloned()
            .ok_or_else(|| format!("unknown variable `{}` in LLVM backend", name))?;
        self.load_from_ptr(&local.ptr, &local.llvm_type)
    }

    pub(super) fn resolve_local_from_access(&mut self, node: &Node) -> Result<LocalVar, String> {
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

    pub(super) fn default_value(&self, llvm_type: &LlvmType) -> ExprValue {
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
            LlvmType::TraitIntersection(_) | LlvmType::Union(_) => ExprValue {
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

    pub(super) fn extract_enum_payloads(
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

    pub(super) fn extract_struct_field_value(
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

    pub(super) fn bind_struct_pattern_fields(
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
    /// Converts a lowered value to the expected runtime representation,
    /// including numeric widening, trait objects, intersections, and unions.
    pub(super) fn coerce_expr(
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
                LlvmType::TraitIntersection(traits) => {
                    self.trait_object_from_intersection(trait_name, &traits, value, context)
                }
                other => Err(format!(
                    "type mismatch in {}: expected `{}`, got `{}`",
                    context,
                    expected.ir(),
                    other.ir()
                )),
            };
        }

        if let LlvmType::TraitIntersection(traits) = expected {
            return match value.llvm_type.clone() {
                LlvmType::Struct(concrete_name) => {
                    self.box_trait_intersection(traits, &concrete_name, value, context)
                }
                LlvmType::TraitIntersection(actual_traits) => {
                    self.project_trait_intersection(traits, &actual_traits, value, context)
                }
                other => Err(format!(
                    "type mismatch in {}: expected `{}`, got `{}`",
                    context,
                    expected.ir(),
                    other.ir()
                )),
            };
        }

        if let LlvmType::Union(members) = expected {
            let mut last_error = None;
            let exact_index = members.iter().position(|member| member == &value.llvm_type);
            let candidates = exact_index
                .into_iter()
                .chain((0..members.len()).filter(|index| Some(*index) != exact_index));
            for index in candidates {
                match self.coerce_expr(value.clone(), &members[index], context) {
                    Ok(member_value) => {
                        return self.box_union_value(members, index, member_value);
                    }
                    Err(error) => last_error = Some(error),
                }
            }
            return Err(last_error.unwrap_or_else(|| {
                format!(
                    "type mismatch in {}: expected `{}`, got `{}`",
                    context,
                    expected.ir(),
                    value.llvm_type.ir()
                )
            }));
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
            | (LlvmType::Pointer { .. }, LlvmType::Pointer { .. })
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

    /// Packages a concrete value as a trait object with data and vtable pointers.
    pub(super) fn box_trait_object(
        &mut self,
        trait_name: &str,
        concrete_name: &str,
        value: ExprValue,
        context: &str,
    ) -> Result<ExprValue, String> {
        let vtable_symbol = self.trait_vtable_symbol(trait_name, concrete_name, context)?;

        let allocator = self.next_temp();
        self.emit_line(format!(
            "{} = call ptr @skunk_system_allocator()",
            allocator
        ));
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

    /// Packages one concrete value with the vtable pointers required by every
    /// member of an intersection type.
    pub(super) fn box_trait_intersection(
        &mut self,
        trait_names: &[String],
        concrete_name: &str,
        value: ExprValue,
        context: &str,
    ) -> Result<ExprValue, String> {
        let vtables = trait_names
            .iter()
            .map(|trait_name| self.trait_vtable_symbol(trait_name, concrete_name, context))
            .collect::<Result<Vec<_>, _>>()?;
        let allocator = self.next_temp();
        self.emit_line(format!(
            "{} = call ptr @skunk_system_allocator()",
            allocator
        ));
        let boxed_ptr = self.next_temp();
        self.emit_line(format!(
            "{} = call ptr @skunk_alloc_create(ptr {}, i64 {})",
            boxed_ptr,
            allocator,
            self.size_of(&value.llvm_type)
        ));
        self.emit_store(&boxed_ptr, &value);

        let intersection_type = LlvmType::TraitIntersection(trait_names.to_vec());
        let mut aggregate = self.next_temp();
        self.emit_line(format!(
            "{} = insertvalue {} zeroinitializer, ptr {}, 0",
            aggregate,
            intersection_type.ir(),
            boxed_ptr
        ));
        for (index, vtable) in vtables.iter().enumerate() {
            let next = self.next_temp();
            self.emit_line(format!(
                "{} = insertvalue {} {}, ptr @{}, {}",
                next,
                intersection_type.ir(),
                aggregate,
                vtable,
                index + 1
            ));
            aggregate = next;
        }
        Ok(ExprValue {
            llvm_type: intersection_type,
            value: aggregate,
        })
    }

    pub(super) fn trait_object_from_intersection(
        &mut self,
        trait_name: &str,
        trait_names: &[String],
        value: ExprValue,
        context: &str,
    ) -> Result<ExprValue, String> {
        let index = trait_names
            .iter()
            .position(|candidate| candidate == trait_name)
            .ok_or_else(|| {
                format!(
                    "type mismatch in {}: intersection does not contain trait `{}`",
                    context, trait_name
                )
            })?;
        let data_ptr = self.next_temp();
        self.emit_line(format!(
            "{} = extractvalue {} {}, 0",
            data_ptr,
            value.llvm_type.ir(),
            value.value
        ));
        let vtable_ptr = self.next_temp();
        self.emit_line(format!(
            "{} = extractvalue {} {}, {}",
            vtable_ptr,
            value.llvm_type.ir(),
            value.value,
            index + 1
        ));
        let trait_type = LlvmType::TraitObject(trait_name.to_string());
        let with_data = self.next_temp();
        self.emit_line(format!(
            "{} = insertvalue {} zeroinitializer, ptr {}, 0",
            with_data,
            trait_type.ir(),
            data_ptr
        ));
        let result = self.next_temp();
        self.emit_line(format!(
            "{} = insertvalue {} {}, ptr {}, 1",
            result,
            trait_type.ir(),
            with_data,
            vtable_ptr
        ));
        Ok(ExprValue {
            llvm_type: trait_type,
            value: result,
        })
    }

    pub(super) fn project_trait_intersection(
        &mut self,
        expected_traits: &[String],
        actual_traits: &[String],
        value: ExprValue,
        context: &str,
    ) -> Result<ExprValue, String> {
        let indices = expected_traits
            .iter()
            .map(|trait_name| {
                actual_traits
                    .iter()
                    .position(|candidate| candidate == trait_name)
                    .ok_or_else(|| {
                        format!(
                            "type mismatch in {}: intersection does not contain trait `{}`",
                            context, trait_name
                        )
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let data_ptr = self.next_temp();
        self.emit_line(format!(
            "{} = extractvalue {} {}, 0",
            data_ptr,
            value.llvm_type.ir(),
            value.value
        ));
        let projected_type = LlvmType::TraitIntersection(expected_traits.to_vec());
        let mut aggregate = self.next_temp();
        self.emit_line(format!(
            "{} = insertvalue {} zeroinitializer, ptr {}, 0",
            aggregate,
            projected_type.ir(),
            data_ptr
        ));
        for (output_index, input_index) in indices.into_iter().enumerate() {
            let vtable_ptr = self.next_temp();
            self.emit_line(format!(
                "{} = extractvalue {} {}, {}",
                vtable_ptr,
                value.llvm_type.ir(),
                value.value,
                input_index + 1
            ));
            let next = self.next_temp();
            self.emit_line(format!(
                "{} = insertvalue {} {}, ptr {}, {}",
                next,
                projected_type.ir(),
                aggregate,
                vtable_ptr,
                output_index + 1
            ));
            aggregate = next;
        }
        Ok(ExprValue {
            llvm_type: projected_type,
            value: aggregate,
        })
    }

    /// Stores a union member behind an erased pointer and pairs it with the
    /// stable member tag from the union layout.
    pub(super) fn box_union_value(
        &mut self,
        members: &[LlvmType],
        member_index: usize,
        value: ExprValue,
    ) -> Result<ExprValue, String> {
        let allocator = self.next_temp();
        self.emit_line(format!(
            "{} = call ptr @skunk_system_allocator()",
            allocator
        ));
        let boxed_ptr = self.next_temp();
        self.emit_line(format!(
            "{} = call ptr @skunk_alloc_create(ptr {}, i64 {})",
            boxed_ptr,
            allocator,
            self.size_of(&value.llvm_type)
        ));
        self.emit_store(&boxed_ptr, &value);
        let union_type = LlvmType::Union(members.to_vec());
        let with_tag = self.next_temp();
        self.emit_line(format!(
            "{} = insertvalue {} zeroinitializer, i32 {}, 0",
            with_tag,
            union_type.ir(),
            member_index
        ));
        let result = self.next_temp();
        self.emit_line(format!(
            "{} = insertvalue {} {}, ptr {}, 1",
            result,
            union_type.ir(),
            with_tag,
            boxed_ptr
        ));
        Ok(ExprValue {
            llvm_type: union_type,
            value: result,
        })
    }

    pub(super) fn trait_vtable_symbol(
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

    pub(super) fn trait_object_from_ptr(
        &mut self,
        trait_name: &str,
        concrete_name: &str,
        data_ptr: String,
        context: &str,
    ) -> Result<ExprValue, String> {
        let vtable_symbol = self.trait_vtable_symbol(trait_name, concrete_name, context)?;
        self.trait_object_from_data_ptr(trait_name, data_ptr, &vtable_symbol)
    }

    pub(super) fn trait_object_from_data_ptr(
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

    pub(super) fn global_c_string(&mut self, prefix: &str, value: &str) -> String {
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

    pub(super) fn string_ptr(&self, global: &str) -> String {
        format!("getelementptr inbounds (i8, ptr @{}, i64 0)", global)
    }

    pub(super) fn emit_store(&mut self, ptr: &str, expr: &ExprValue) {
        self.emit_line(format!(
            "store {} {}, ptr {}, align {}",
            expr.llvm_type.ir(),
            expr.value,
            ptr,
            self.align_of(&expr.llvm_type)
        ));
    }

    pub(super) fn emit_alloca(&mut self, llvm_type: LlvmType, hint: &str) -> String {
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

    pub(super) fn emit_heap_alloc(&mut self, llvm_type: LlvmType, hint: &str) -> String {
        let ptr = format!("%{}_{}", sanitize_name(hint), self.temp_counter);
        self.temp_counter += 1;
        self.emit_line(format!(
            "{} = call ptr @malloc(i64 {})",
            ptr,
            self.size_of(&llvm_type)
        ));
        ptr
    }

    pub(super) fn emit_label(&mut self, label: &str) {
        self.lines.push(format!("{}:", label));
    }

    pub(super) fn emit_line(&mut self, line: String) {
        self.lines.push(format!("  {}", line));
    }

    pub(super) fn next_temp(&mut self) -> String {
        let name = format!("%t{}", self.temp_counter);
        self.temp_counter += 1;
        name
    }

    pub(super) fn next_label(&mut self, prefix: &str) -> String {
        let label = format!(
            "{}_{}_{}",
            sanitize_name(self.function_name),
            prefix,
            self.label_counter
        );
        self.label_counter += 1;
        label
    }

    pub(super) fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
        self.deferred_scopes.push(Vec::new());
    }

    pub(super) fn pop_scope(&mut self) {
        self.scopes.pop();
        self.deferred_scopes.pop();
    }

    pub(super) fn compile_current_scope_defers(&mut self) -> Result<(), String> {
        let deferred = self
            .deferred_scopes
            .last()
            .expect("defer scope stack should never be empty")
            .iter()
            .rev()
            .cloned()
            .collect::<Vec<_>>();
        self.compile_deferred_expressions(deferred)
    }

    pub(super) fn compile_all_scope_defers(&mut self) -> Result<(), String> {
        let deferred = self
            .deferred_scopes
            .iter()
            .rev()
            .flat_map(|scope| scope.iter().rev())
            .cloned()
            .collect::<Vec<_>>();
        self.compile_deferred_expressions(deferred)
    }

    /// Emits deferred expressions in reverse registration order while preserving
    /// the surrounding block's termination state.
    pub(super) fn compile_deferred_expressions(
        &mut self,
        deferred: Vec<DeferredExpression>,
    ) -> Result<(), String> {
        for deferred_expression in deferred {
            let active_scopes =
                std::mem::replace(&mut self.scopes, vec![deferred_expression.locals]);
            let active_unsafe_depth =
                std::mem::replace(&mut self.unsafe_depth, deferred_expression.unsafe_depth);
            let result = self.compile_expr(&deferred_expression.expression);
            self.scopes = active_scopes;
            self.unsafe_depth = active_unsafe_depth;
            let _ = result?;
        }
        Ok(())
    }

    pub(super) fn enter_unsafe(&mut self) {
        self.unsafe_depth += 1;
    }

    pub(super) fn exit_unsafe(&mut self) {
        if self.unsafe_depth > 0 {
            self.unsafe_depth -= 1;
        }
    }

    pub(super) fn unsafe_allowed(&self) -> bool {
        self.unsafe_depth > 0
    }

    pub(super) fn declare_local(&mut self, name: String, ptr: String, llvm_type: LlvmType) {
        self.scopes
            .last_mut()
            .expect("scope stack should never be empty")
            .insert(name, LocalVar { ptr, llvm_type });
    }

    pub(super) fn lookup_local(&self, name: &str) -> Option<&LocalVar> {
        self.scopes.iter().rev().find_map(|scope| scope.get(name))
    }

    pub(super) fn align_of(&self, llvm_type: &LlvmType) -> usize {
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
            | LlvmType::TraitIntersection(_)
            | LlvmType::Union(_)
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

    pub(super) fn size_of(&self, llvm_type: &LlvmType) -> usize {
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
            LlvmType::TraitIntersection(traits) => 8 * (traits.len() + 1),
            LlvmType::Union(_) => 16,
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
