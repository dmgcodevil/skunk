//! Address computation for MIR locals, projections, and bounds-checked indexes.

use super::*;

impl<'a> FunctionCompiler<'a> {
    pub(super) fn mir_place(
        &mut self,
        place: &ir::Place,
        locals: &[LocalVar],
        context: &CodegenContext<'_>,
    ) -> Result<LocalVar, String> {
        let mut current = match place.base {
            ir::PlaceBase::Local(local) => locals
                .get(local.index())
                .cloned()
                .ok_or_else(|| format!("unknown MIR local {}", local.index()))?,
            ir::PlaceBase::Definition(definition) => {
                let global = context.globals.get(&definition).ok_or_else(|| {
                    format!("unknown MIR global definition {}", definition.index())
                })?;
                LocalVar {
                    ptr: format!("@{}", global.symbol_name),
                    llvm_type: global.llvm_type.clone(),
                }
            }
        };

        for projection in &place.projections {
            match &projection.kind {
                ir::ProjectionKind::Field(field) => {
                    let LlvmType::Struct(struct_name) = &current.llvm_type else {
                        return Err(format!(
                            "cannot project MIR field {} from `{}`",
                            field.index(),
                            current.llvm_type.ir()
                        ));
                    };
                    let index = *context
                        .field_indices
                        .get(field)
                        .ok_or_else(|| format!("unknown MIR field {}", field.index()))?;
                    if self
                        .structs
                        .get(struct_name)
                        .and_then(|layout| layout.fields.get(index))
                        .is_none()
                    {
                        return Err(format!(
                            "field {} is outside struct `{struct_name}` layout",
                            field.index()
                        ));
                    }
                    let ptr = self.next_temp();
                    self.emit_line(format!(
                        "{} = getelementptr inbounds {}, ptr {}, i32 0, i32 {}",
                        ptr,
                        current.llvm_type.ir(),
                        current.ptr,
                        index
                    ));
                    current = LocalVar {
                        ptr,
                        llvm_type: context.llvm_type(projection.ty)?,
                    };
                }
                ir::ProjectionKind::Index(coordinates) => {
                    for coordinate in coordinates {
                        let index = self.compile_mir_operand(coordinate, locals, context)?;
                        let index = self.coerce_expr(index, &LlvmType::I64, "MIR array index")?;
                        let (ptr, elem_type) = match &current.llvm_type {
                            LlvmType::Array { elem_type, len } => {
                                let elem_type = elem_type.as_ref().clone();
                                let length = ExprValue {
                                    llvm_type: LlvmType::I64,
                                    value: len.to_string(),
                                };
                                self.emit_index_bounds_check(&index, &length)?;
                                let ptr = self.next_temp();
                                self.emit_line(format!(
                                    "{} = getelementptr inbounds {}, ptr {}, i64 0, i64 {}",
                                    ptr,
                                    current.llvm_type.ir(),
                                    current.ptr,
                                    index.value
                                ));
                                (ptr, elem_type)
                            }
                            LlvmType::Slice { elem_type } => {
                                let elem_type = elem_type.as_ref().clone();
                                let slice = self.load_from_ptr(&current.ptr, &current.llvm_type)?;
                                let length = self.extract_slice_len(&slice)?;
                                self.emit_index_bounds_check(&index, &length)?;
                                let data_ptr = self.extract_slice_data(&slice)?;
                                let ptr = self.next_temp();
                                self.emit_line(format!(
                                    "{} = getelementptr inbounds {}, ptr {}, i64 {}",
                                    ptr,
                                    elem_type.ir(),
                                    data_ptr,
                                    index.value
                                ));
                                (ptr, elem_type)
                            }
                            other => {
                                return Err(format!(
                                    "cannot index MIR place of type `{}`",
                                    other.ir()
                                ))
                            }
                        };
                        current = LocalVar {
                            ptr,
                            llvm_type: elem_type,
                        };
                    }
                    let projected_type = context.llvm_type(projection.ty)?;
                    if current.llvm_type != projected_type {
                        return Err(format!(
                            "MIR index projection produced `{}` instead of `{}`",
                            current.llvm_type.ir(),
                            projected_type.ir()
                        ));
                    }
                }
                ir::ProjectionKind::Dereference => {
                    let target_type = match &current.llvm_type {
                        LlvmType::Reference { target_type, .. } => target_type.as_ref().clone(),
                        LlvmType::Pointer { target_type } => target_type.as_ref().clone(),
                        other => {
                            return Err(format!(
                                "cannot dereference MIR place of type `{}`",
                                other.ir()
                            ))
                        }
                    };
                    let pointer = self.load_from_ptr(&current.ptr, &current.llvm_type)?;
                    current = LocalVar {
                        ptr: pointer.value,
                        llvm_type: target_type,
                    };
                    let projected_type = context.llvm_type(projection.ty)?;
                    if current.llvm_type != projected_type {
                        return Err(format!(
                            "MIR dereference produced `{}` instead of `{}`",
                            current.llvm_type.ir(),
                            projected_type.ir()
                        ));
                    }
                }
                ir::ProjectionKind::VariantField { variant, index } => {
                    let LlvmType::Enum(enum_name) = &current.llvm_type else {
                        return Err(format!(
                            "cannot project a MIR variant field from `{}`",
                            current.llvm_type.ir()
                        ));
                    };
                    let info = context
                        .variants
                        .get(variant)
                        .ok_or_else(|| format!("unknown MIR variant {}", variant.index()))?;
                    if enum_name != &info.enum_name {
                        return Err(format!(
                            "variant {} belongs to `{}`, not `{enum_name}`",
                            variant.index(),
                            info.enum_name
                        ));
                    }
                    let field_index =
                        *info.field_indices.get(*index as usize).ok_or_else(|| {
                            format!("variant {} has no payload field {index}", variant.index())
                        })?;
                    let ptr = self.next_temp();
                    self.emit_line(format!(
                        "{} = getelementptr inbounds {}, ptr {}, i32 0, i32 {}",
                        ptr,
                        current.llvm_type.ir(),
                        current.ptr,
                        field_index
                    ));
                    current = LocalVar {
                        ptr,
                        llvm_type: context.llvm_type(projection.ty)?,
                    };
                }
            }
        }
        Ok(current)
    }
}
