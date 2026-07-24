//! Direct, indirect, enum-constructor, intrinsic, and receiver calls.

use super::*;

impl<'a> FunctionCompiler<'a> {
    pub(super) fn compile_mir_direct_call(
        &mut self,
        definition: DefId,
        receiver: Option<String>,
        arguments: Vec<ExprValue>,
        context: &CodegenContext<'_>,
    ) -> Result<ExprValue, String> {
        let key = context
            .direct_signature_keys
            .get(&definition)
            .ok_or_else(|| {
                format!(
                    "definition {} is not a directly callable MIR function",
                    definition.index()
                )
            })?;
        let signature = self
            .signatures
            .get(key)
            .cloned()
            .ok_or_else(|| format!("missing MIR call signature for `{key}`"))?;
        if arguments.len() != signature.parameters.len() {
            return Err(format!(
                "MIR call to `{key}` expected {} arguments, found {}",
                signature.parameters.len(),
                arguments.len()
            ));
        }
        let arguments = arguments
            .into_iter()
            .zip(&signature.parameters)
            .map(|(argument, expected)| self.coerce_expr(argument, expected, "MIR call argument"))
            .collect::<Result<Vec<_>, _>>()?;
        let mut rendered = arguments
            .iter()
            .map(|argument| format!("{} {}", argument.llvm_type.ir(), argument.value))
            .collect::<Vec<_>>();
        if let Some(receiver) = receiver {
            rendered.insert(0, format!("ptr {receiver}"));
        }
        let rendered = rendered.join(", ");
        if signature.return_type == LlvmType::Void {
            self.emit_line(format!("call void @{}({rendered})", signature.symbol_name));
            Ok(ExprValue {
                llvm_type: LlvmType::Void,
                value: "void".to_string(),
            })
        } else {
            let temp = self.next_temp();
            self.emit_line(format!(
                "{} = call {} @{}({rendered})",
                temp,
                signature.return_type.ir(),
                signature.symbol_name
            ));
            Ok(ExprValue {
                llvm_type: signature.return_type,
                value: temp,
            })
        }
    }

    pub(super) fn compile_mir_dynamic_call(
        &mut self,
        receiver: ExprValue,
        method: DefId,
        arguments: Vec<ExprValue>,
        context: &CodegenContext<'_>,
    ) -> Result<ExprValue, String> {
        let method_name = definition_name(context.model, method)?;
        let (trait_name, vtable_field, method_index, signature) = match &receiver.llvm_type {
            LlvmType::TraitObject(trait_name) => {
                let (method_index, signature) = self
                    .traits
                    .get(trait_name)
                    .and_then(|layout| {
                        layout
                            .methods
                            .iter()
                            .enumerate()
                            .find(|(_, candidate)| candidate.name == method_name)
                    })
                    .map(|(index, signature)| (index, signature.clone()))
                    .ok_or_else(|| {
                        format!("unknown dynamic method `{method_name}` on trait `{trait_name}`")
                    })?;
                (trait_name.clone(), 1usize, method_index, signature)
            }
            LlvmType::TraitIntersection(trait_names) => {
                let mut matches =
                    trait_names
                        .iter()
                        .enumerate()
                        .filter_map(|(trait_index, trait_name)| {
                            self.traits.get(trait_name).and_then(|layout| {
                                layout
                                    .methods
                                    .iter()
                                    .enumerate()
                                    .find(|(_, candidate)| candidate.name == method_name)
                                    .map(|(method_index, signature)| {
                                        (
                                            trait_name.clone(),
                                            trait_index + 1,
                                            method_index,
                                            signature.clone(),
                                        )
                                    })
                            })
                        });
                let found = matches.next().ok_or_else(|| {
                    format!("unknown dynamic method `{method_name}` on trait intersection")
                })?;
                if matches.next().is_some() {
                    return Err(format!(
                        "ambiguous dynamic method `{method_name}` on trait intersection"
                    ));
                }
                found
            }
            other => {
                return Err(format!(
                    "dynamic MIR call requires a trait object, found `{}`",
                    other.ir()
                ))
            }
        };
        if arguments.len() != signature.parameters.len() {
            return Err(format!(
                "dynamic MIR method `{method_name}` expected {} arguments, found {}",
                signature.parameters.len(),
                arguments.len()
            ));
        }
        let data_ptr = self.next_temp();
        self.emit_line(format!(
            "{} = extractvalue {} {}, 0",
            data_ptr,
            receiver.llvm_type.ir(),
            receiver.value
        ));
        let vtable_ptr = self.next_temp();
        self.emit_line(format!(
            "{} = extractvalue {} {}, {}",
            vtable_ptr,
            receiver.llvm_type.ir(),
            receiver.value,
            vtable_field
        ));
        let slot_ptr = self.next_temp();
        self.emit_line(format!(
            "{} = getelementptr inbounds %vtable.{}, ptr {}, i32 0, i32 {}",
            slot_ptr,
            sanitize_name(&trait_name),
            vtable_ptr,
            method_index
        ));
        let function_ptr = self.next_temp();
        self.emit_line(format!(
            "{} = load ptr, ptr {}, align 8",
            function_ptr, slot_ptr
        ));
        let mut rendered = vec![format!("ptr {data_ptr}")];
        for (argument, expected) in arguments.into_iter().zip(&signature.parameters) {
            let argument = self.coerce_expr(argument, expected, "dynamic MIR method argument")?;
            rendered.push(format!("{} {}", argument.llvm_type.ir(), argument.value));
        }
        if signature.return_type == LlvmType::Void {
            self.emit_line(format!(
                "call void {}({})",
                function_ptr,
                rendered.join(", ")
            ));
            Ok(ExprValue {
                llvm_type: LlvmType::Void,
                value: "void".to_string(),
            })
        } else {
            let result = self.next_temp();
            self.emit_line(format!(
                "{} = call {} {}({})",
                result,
                signature.return_type.ir(),
                function_ptr,
                rendered.join(", ")
            ));
            Ok(ExprValue {
                llvm_type: signature.return_type,
                value: result,
            })
        }
    }

    pub(super) fn compile_mir_variant(
        &mut self,
        variant: VariantId,
        arguments: Vec<ExprValue>,
        expected: &LlvmType,
        context: &CodegenContext<'_>,
    ) -> Result<ExprValue, String> {
        let info = context
            .variants
            .get(&variant)
            .ok_or_else(|| format!("unknown MIR variant {}", variant.index()))?;
        let LlvmType::Enum(expected_name) = expected else {
            return Err(format!(
                "MIR enum variant cannot initialize `{}`",
                expected.ir()
            ));
        };
        if expected_name != &info.enum_name {
            return Err(format!(
                "MIR variant of `{}` cannot initialize `{expected_name}`",
                info.enum_name
            ));
        }
        if arguments.len() != info.payload_types.len() {
            return Err(format!(
                "MIR variant {} expected {} payload values, found {}",
                variant.index(),
                info.payload_types.len(),
                arguments.len()
            ));
        }

        let with_tag = self.next_temp();
        self.emit_line(format!(
            "{} = insertvalue {} zeroinitializer, i32 {}, 0",
            with_tag,
            expected.ir(),
            info.tag
        ));
        let mut result = ExprValue {
            llvm_type: expected.clone(),
            value: with_tag,
        };
        for (((argument, payload_type), field_index), payload_index) in arguments
            .into_iter()
            .zip(&info.payload_types)
            .zip(&info.field_indices)
            .zip(0..)
        {
            let argument = self.coerce_expr(
                argument,
                payload_type,
                &format!("MIR variant payload {payload_index}"),
            )?;
            let next = self.next_temp();
            self.emit_line(format!(
                "{} = insertvalue {} {}, {} {}, {}",
                next,
                expected.ir(),
                result.value,
                argument.llvm_type.ir(),
                argument.value,
                field_index
            ));
            result.value = next;
        }
        Ok(result)
    }

    pub(super) fn compile_mir_receiver(
        &mut self,
        receiver: &ir::Operand,
        locals: &[LocalVar],
        context: &CodegenContext<'_>,
    ) -> Result<String, String> {
        let ir::OperandKind::Copy(place) = &receiver.kind else {
            return Err("direct MIR method receiver must be addressable".to_string());
        };
        let receiver = self.mir_place(place, locals, context)?;
        match &receiver.llvm_type {
            LlvmType::Struct(_) | LlvmType::Enum(_) => Ok(receiver.ptr),
            LlvmType::Reference { target_type, .. }
                if matches!(
                    target_type.as_ref(),
                    LlvmType::Struct(_) | LlvmType::Enum(_)
                ) =>
            {
                Ok(self
                    .load_from_ptr(&receiver.ptr, &receiver.llvm_type)?
                    .value)
            }
            LlvmType::Pointer { target_type }
                if matches!(
                    target_type.as_ref(),
                    LlvmType::Struct(_) | LlvmType::Enum(_)
                ) =>
            {
                Ok(self
                    .load_from_ptr(&receiver.ptr, &receiver.llvm_type)?
                    .value)
            }
            other => Err(format!(
                "direct MIR method receiver must be nominal or a nominal reference, found `{}`",
                other.ir()
            )),
        }
    }
}
