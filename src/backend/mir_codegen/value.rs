//! MIR rvalues, aggregates, slices, operands, constants, and unary operations.

use super::*;

impl<'a> FunctionCompiler<'a> {
    pub(super) fn compile_mir_rvalue(
        &mut self,
        value: &ir::Rvalue,
        locals: &[LocalVar],
        context: &CodegenContext<'_>,
    ) -> Result<ExprValue, String> {
        let expected = context.llvm_type(value.ty)?;
        let compiled = match &value.kind {
            ir::RvalueKind::Use(operand) => self.compile_mir_operand(operand, locals, context)?,
            ir::RvalueKind::Coerce(operand) => {
                if let (LlvmType::TraitObject(trait_name), ir::OperandKind::Copy(place)) =
                    (&expected, &operand.kind)
                {
                    let place = self.mir_place(place, locals, context)?;
                    if let LlvmType::Struct(concrete_name) = &place.llvm_type {
                        return self.trait_object_from_ptr(
                            trait_name,
                            concrete_name,
                            place.ptr,
                            "MIR trait coercion",
                        );
                    }
                }
                let operand = self.compile_mir_operand(operand, locals, context)?;
                self.coerce_expr(operand, &expected, "explicit MIR coercion")?
            }
            ir::RvalueKind::Unary { operator, operand } => {
                let operand = self.compile_mir_operand(operand, locals, context)?;
                self.compile_mir_unary(*operator, operand)?
            }
            ir::RvalueKind::Binary {
                left,
                operator,
                right,
            } => {
                let left = self.compile_mir_operand(left, locals, context)?;
                let right = self.compile_mir_operand(right, locals, context)?;
                self.compile_binary_values(left, *operator, right)?
            }
            ir::RvalueKind::Aggregate(aggregate) => {
                self.compile_mir_aggregate(aggregate, &expected, locals, context)?
            }
            ir::RvalueKind::Length(operand) => {
                let operand_type = context.llvm_type(operand.ty)?;
                match operand_type {
                    LlvmType::Array { len, .. } => ExprValue {
                        llvm_type: LlvmType::I32,
                        value: len.to_string(),
                    },
                    LlvmType::Slice { .. } => {
                        let slice = self.compile_mir_operand(operand, locals, context)?;
                        self.extract_slice_len(&slice)?
                    }
                    other => {
                        return Err(format!(
                            "direct MIR length requires an array or slice, found `{}`",
                            other.ir()
                        ))
                    }
                }
            }
            ir::RvalueKind::Reference { place, .. } => {
                let place = self.mir_place(place, locals, context)?;
                if !matches!(
                    expected,
                    LlvmType::Reference { .. } | LlvmType::Pointer { .. }
                ) {
                    return Err(format!(
                        "MIR reference cannot initialize `{}`",
                        expected.ir()
                    ));
                }
                ExprValue {
                    llvm_type: expected.clone(),
                    value: place.ptr,
                }
            }
            ir::RvalueKind::Slice {
                receiver,
                start,
                end,
            } => self.compile_mir_slice(receiver, start.as_ref(), end.as_ref(), locals, context)?,
            ir::RvalueKind::Closure { function, captures } => {
                self.compile_mir_closure(*function, captures, &expected, locals, context)?
            }
        };
        self.coerce_expr(compiled, &expected, "MIR rvalue")
    }

    fn compile_mir_unary(
        &mut self,
        operator: MirUnaryOperator,
        operand: ExprValue,
    ) -> Result<ExprValue, String> {
        match operator {
            MirUnaryOperator::Plus => Ok(operand),
            MirUnaryOperator::Minus => {
                if !is_numeric_llvm_type(&operand.llvm_type)
                    || operand.llvm_type == LlvmType::Char16
                {
                    return Err("unary `-` requires a numeric MIR operand".to_string());
                }
                let target = if matches!(operand.llvm_type, LlvmType::I8 | LlvmType::I16) {
                    LlvmType::I32
                } else {
                    operand.llvm_type.clone()
                };
                let operand = self.coerce_expr(operand, &target, "MIR unary `-`")?;
                let temp = self.next_temp();
                let (instruction, zero) = if matches!(target, LlvmType::F32 | LlvmType::F64) {
                    ("fsub", "0.0")
                } else {
                    ("sub", "0")
                };
                self.emit_line(format!(
                    "{} = {} {} {}, {}",
                    temp,
                    instruction,
                    target.ir(),
                    zero,
                    operand.value
                ));
                Ok(ExprValue {
                    llvm_type: target,
                    value: temp,
                })
            }
            MirUnaryOperator::Not => {
                if operand.llvm_type != LlvmType::I1 {
                    return Err("unary `!` requires a boolean MIR operand".to_string());
                }
                let temp = self.next_temp();
                self.emit_line(format!("{} = xor i1 {}, true", temp, operand.value));
                Ok(ExprValue {
                    llvm_type: LlvmType::I1,
                    value: temp,
                })
            }
            MirUnaryOperator::AddressOf
            | MirUnaryOperator::AddressOfMut
            | MirUnaryOperator::Dereference => {
                Err("address operations are not in the direct MIR slice yet".to_string())
            }
        }
    }

    fn compile_mir_aggregate(
        &mut self,
        aggregate: &ir::Aggregate,
        expected: &LlvmType,
        locals: &[LocalVar],
        context: &CodegenContext<'_>,
    ) -> Result<ExprValue, String> {
        if let (ir::Aggregate::Array(elements), LlvmType::Slice { elem_type }) =
            (aggregate, expected)
        {
            let backing_type = LlvmType::Array {
                elem_type: elem_type.clone(),
                len: elements.len(),
            };
            let backing = self.compile_mir_aggregate(aggregate, &backing_type, locals, context)?;
            let backing_ptr = self.emit_heap_alloc(backing_type.clone(), "mir_slice_literal");
            self.emit_store(&backing_ptr, &backing);
            return self.build_slice_from_array_ptr(&backing_ptr, &backing_type);
        }

        let mut result = ExprValue {
            llvm_type: expected.clone(),
            value: "zeroinitializer".to_string(),
        };
        match aggregate {
            ir::Aggregate::Struct { definition, fields } => {
                let LlvmType::Struct(expected_name) = expected else {
                    return Err(format!(
                        "MIR struct aggregate cannot initialize `{}`",
                        expected.ir()
                    ));
                };
                let actual_name = definition_name(context.model, *definition)?;
                if actual_name != expected_name {
                    return Err(format!(
                        "MIR struct aggregate `{actual_name}` cannot initialize `{expected_name}`"
                    ));
                }
                for (field, operand) in fields {
                    let index = *context
                        .field_indices
                        .get(field)
                        .ok_or_else(|| format!("unknown MIR field {}", field.index()))?;
                    let (_, field_type) = self
                        .structs
                        .get(expected_name)
                        .and_then(|layout| layout.fields.get(index))
                        .ok_or_else(|| {
                            format!(
                                "field {} is outside struct `{expected_name}` layout",
                                field.index()
                            )
                        })?;
                    let field_type = field_type.clone();
                    let value = self.compile_mir_operand(operand, locals, context)?;
                    let value = self.coerce_expr(value, &field_type, "MIR struct field")?;
                    let next = self.next_temp();
                    self.emit_line(format!(
                        "{} = insertvalue {} {}, {} {}, {}",
                        next,
                        expected.ir(),
                        result.value,
                        value.llvm_type.ir(),
                        value.value,
                        index
                    ));
                    result.value = next;
                }
            }
            ir::Aggregate::Array(elements) => {
                let LlvmType::Array { elem_type, len } = expected else {
                    return Err(format!(
                        "MIR array aggregate cannot initialize `{}`",
                        expected.ir()
                    ));
                };
                if elements.len() != *len {
                    return Err(format!(
                        "MIR array aggregate expected {len} elements, found {}",
                        elements.len()
                    ));
                }
                let elem_type = elem_type.as_ref().clone();
                for (index, operand) in elements.iter().enumerate() {
                    let value = self.compile_mir_operand(operand, locals, context)?;
                    let value = self.coerce_expr(value, &elem_type, "MIR array element")?;
                    let next = self.next_temp();
                    self.emit_line(format!(
                        "{} = insertvalue {} {}, {} {}, {}",
                        next,
                        expected.ir(),
                        result.value,
                        value.llvm_type.ir(),
                        value.value,
                        index
                    ));
                    result.value = next;
                }
            }
        }
        Ok(result)
    }

    fn compile_mir_slice(
        &mut self,
        receiver: &ir::Operand,
        start: Option<&ir::Operand>,
        end: Option<&ir::Operand>,
        locals: &[LocalVar],
        context: &CodegenContext<'_>,
    ) -> Result<ExprValue, String> {
        let ir::OperandKind::Copy(receiver_place) = &receiver.kind else {
            return Err("direct MIR slice receiver must be addressable".to_string());
        };
        let receiver = self.mir_place(receiver_place, locals, context)?;
        let (data_ptr, elem_type, base_len) = match &receiver.llvm_type {
            LlvmType::Array { elem_type, len } => {
                let data_ptr = if *len == 0 {
                    "null".to_string()
                } else {
                    let data_ptr = self.next_temp();
                    self.emit_line(format!(
                        "{} = getelementptr inbounds {}, ptr {}, i64 0, i64 0",
                        data_ptr,
                        receiver.llvm_type.ir(),
                        receiver.ptr
                    ));
                    data_ptr
                };
                (
                    data_ptr,
                    elem_type.as_ref().clone(),
                    ExprValue {
                        llvm_type: LlvmType::I32,
                        value: len.to_string(),
                    },
                )
            }
            LlvmType::Slice { elem_type } => {
                let value = self.load_from_ptr(&receiver.ptr, &receiver.llvm_type)?;
                let data_ptr = self.extract_slice_data(&value)?;
                let base_len = self.extract_slice_len(&value)?;
                (data_ptr, elem_type.as_ref().clone(), base_len)
            }
            other => {
                return Err(format!(
                    "cannot take a direct MIR slice of `{}`",
                    other.ir()
                ))
            }
        };
        let start = match start {
            Some(start) => {
                let start = self.compile_mir_operand(start, locals, context)?;
                self.coerce_expr(start, &LlvmType::I32, "MIR slice start")?
            }
            None => ExprValue {
                llvm_type: LlvmType::I32,
                value: "0".to_string(),
            },
        };
        let end = match end {
            Some(end) => {
                let end = self.compile_mir_operand(end, locals, context)?;
                self.coerce_expr(end, &LlvmType::I32, "MIR slice end")?
            }
            None => base_len.clone(),
        };
        self.build_slice_header_from_values(&data_ptr, &elem_type, &base_len, start, end)
    }

    pub(super) fn compile_mir_operand(
        &mut self,
        operand: &ir::Operand,
        locals: &[LocalVar],
        context: &CodegenContext<'_>,
    ) -> Result<ExprValue, String> {
        match &operand.kind {
            ir::OperandKind::Copy(place) => {
                let place = self.mir_place(place, locals, context)?;
                self.load_from_ptr(&place.ptr, &place.llvm_type)
            }
            ir::OperandKind::Constant(constant) => {
                self.compile_mir_constant(constant, operand.ty, context)
            }
            ir::OperandKind::Definition(_) => {
                Err("function definitions are valid only as direct MIR call targets".to_string())
            }
        }
    }

    fn compile_mir_constant(
        &mut self,
        constant: &ir::Constant,
        ty: TypeId,
        context: &CodegenContext<'_>,
    ) -> Result<ExprValue, String> {
        let llvm_type = context.llvm_type(ty)?;
        let value = match constant {
            ir::Constant::Unit => "void".to_string(),
            ir::Constant::Literal(MirLiteral::Integer(value))
            | ir::Constant::Literal(MirLiteral::Long(value)) => value.to_string(),
            ir::Constant::Literal(MirLiteral::Float(value)) => format_float(*value as f64),
            ir::Constant::Literal(MirLiteral::Double(value)) => format_float(*value),
            ir::Constant::Literal(MirLiteral::Boolean(value)) => {
                if *value { "1" } else { "0" }.to_string()
            }
            ir::Constant::Literal(MirLiteral::Char(value)) => (*value as u32 as u16).to_string(),
            ir::Constant::Literal(MirLiteral::String(value)) => {
                let parsed = parse_string_literal(value)?;
                let global = self.global_c_string("str", &parsed);
                self.string_ptr(&global)
            }
        };
        Ok(ExprValue { llvm_type, value })
    }
}

fn parse_string_literal(literal: &str) -> Result<String, String> {
    let inner = literal
        .strip_prefix('"')
        .and_then(|value| value.strip_suffix('"'))
        .ok_or_else(|| format!("invalid string literal `{literal}`"))?;
    let mut output = String::new();
    let mut chars = inner.chars();
    while let Some(character) = chars.next() {
        if character != '\\' {
            output.push(character);
            continue;
        }
        let escaped = chars
            .next()
            .ok_or_else(|| format!("unterminated escape sequence in `{literal}`"))?;
        output.push(match escaped {
            'n' => '\n',
            'r' => '\r',
            't' => '\t',
            '0' => '\0',
            '"' => '"',
            '\\' => '\\',
            other => {
                return Err(format!(
                    "unsupported escape sequence `\\{other}` in `{literal}`"
                ))
            }
        });
    }
    Ok(output)
}
