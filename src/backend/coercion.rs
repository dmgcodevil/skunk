//! MIR value operations, representation coercions, and trait-object packing.

use super::*;
use crate::syntax::ast::BinaryOperator;

impl<'a> FunctionCompiler<'a> {
    pub(super) fn compile_binary_values(
        &mut self,
        left: ExprValue,
        operator: BinaryOperator,
        right: ExprValue,
    ) -> Result<ExprValue, String> {
        if operator == BinaryOperator::Power {
            return Err("power is not supported by LLVM codegen yet".to_string());
        }
        if let Some(promoted) = promoted_numeric_llvm_type(&left.llvm_type, &right.llvm_type) {
            let left = self.coerce_expr(left, &promoted, "binary operand")?;
            let right = self.coerce_expr(right, &promoted, "binary operand")?;
            let instruction = if matches!(promoted, LlvmType::F32 | LlvmType::F64) {
                match operator {
                    BinaryOperator::Add => "fadd",
                    BinaryOperator::Subtract => "fsub",
                    BinaryOperator::Multiply => "fmul",
                    BinaryOperator::Divide => "fdiv",
                    BinaryOperator::Equals => "fcmp oeq",
                    BinaryOperator::NotEquals => "fcmp one",
                    BinaryOperator::LessThan => "fcmp olt",
                    BinaryOperator::LessThanOrEqual => "fcmp ole",
                    BinaryOperator::GreaterThan => "fcmp ogt",
                    BinaryOperator::GreaterThanOrEqual => "fcmp oge",
                    _ => {
                        return Err(format!(
                            "operator `{operator:?}` is invalid for floating-point values"
                        ))
                    }
                }
            } else {
                match operator {
                    BinaryOperator::Add => "add",
                    BinaryOperator::Subtract => "sub",
                    BinaryOperator::Multiply => "mul",
                    BinaryOperator::Divide => "sdiv",
                    BinaryOperator::Modulo => "srem",
                    BinaryOperator::Equals => "icmp eq",
                    BinaryOperator::NotEquals => "icmp ne",
                    BinaryOperator::LessThan => "icmp slt",
                    BinaryOperator::LessThanOrEqual => "icmp sle",
                    BinaryOperator::GreaterThan => "icmp sgt",
                    BinaryOperator::GreaterThanOrEqual => "icmp sge",
                    _ => {
                        return Err(format!(
                            "operator `{operator:?}` is invalid for integer values"
                        ))
                    }
                }
            };
            let result = self.next_temp();
            self.emit_line(format!(
                "{} = {} {} {}, {}",
                result,
                instruction,
                promoted.ir(),
                left.value,
                right.value
            ));
            return Ok(ExprValue {
                llvm_type: if is_comparison(operator) {
                    LlvmType::I1
                } else {
                    promoted
                },
                value: result,
            });
        }

        let (instruction, result_type) = match (&left.llvm_type, operator, &right.llvm_type) {
            (LlvmType::I1, BinaryOperator::And, LlvmType::I1) => ("and", LlvmType::I1),
            (LlvmType::I1, BinaryOperator::Or, LlvmType::I1) => ("or", LlvmType::I1),
            (LlvmType::I1, BinaryOperator::Equals, LlvmType::I1) => ("icmp eq", LlvmType::I1),
            (LlvmType::I1, BinaryOperator::NotEquals, LlvmType::I1) => ("icmp ne", LlvmType::I1),
            (LlvmType::Char16, operator, LlvmType::Char16) if is_comparison(operator) => {
                let instruction = match operator {
                    BinaryOperator::Equals => "icmp eq",
                    BinaryOperator::NotEquals => "icmp ne",
                    BinaryOperator::LessThan => "icmp ult",
                    BinaryOperator::LessThanOrEqual => "icmp ule",
                    BinaryOperator::GreaterThan => "icmp ugt",
                    BinaryOperator::GreaterThanOrEqual => "icmp uge",
                    _ => {
                        return Err(format!(
                            "operator `{operator:?}` is invalid for character values"
                        ))
                    }
                };
                (instruction, LlvmType::I1)
            }
            (LlvmType::PtrI8, BinaryOperator::Equals, LlvmType::PtrI8) => ("icmp eq", LlvmType::I1),
            (LlvmType::PtrI8, BinaryOperator::NotEquals, LlvmType::PtrI8) => {
                ("icmp ne", LlvmType::I1)
            }
            _ => {
                return Err(format!(
                    "operator `{operator:?}` is invalid for `{}` and `{}`",
                    left.llvm_type.ir(),
                    right.llvm_type.ir()
                ))
            }
        };
        let result = self.next_temp();
        self.emit_line(format!(
            "{} = {} {} {}, {}",
            result,
            instruction,
            left.llvm_type.ir(),
            left.value,
            right.value
        ));
        Ok(ExprValue {
            llvm_type: result_type,
            value: result,
        })
    }

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
                LlvmType::TraitIntersection(actual_traits) => {
                    self.trait_object_from_intersection(trait_name, &actual_traits, value, context)
                }
                actual => Err(type_mismatch(context, expected, &actual)),
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
                actual => Err(type_mismatch(context, expected, &actual)),
            };
        }
        if let LlvmType::Union(members) = expected {
            let exact = members.iter().position(|member| member == &value.llvm_type);
            let candidates = exact
                .into_iter()
                .chain((0..members.len()).filter(|index| Some(*index) != exact));
            let mut last_error = None;
            for index in candidates {
                match self.coerce_expr(value.clone(), &members[index], context) {
                    Ok(member) => return self.box_union_value(members, index, member),
                    Err(error) => last_error = Some(error),
                }
            }
            return Err(
                last_error.unwrap_or_else(|| type_mismatch(context, expected, &value.llvm_type))
            );
        }
        if let (
            LlvmType::Reference {
                target_type: actual,
                mutable: true,
            },
            LlvmType::Reference {
                target_type: expected_target,
                mutable: false,
            },
        ) = (&value.llvm_type, expected)
        {
            if actual == expected_target {
                return Ok(ExprValue {
                    llvm_type: expected.clone(),
                    value: value.value,
                });
            }
        }

        let result = self.next_temp();
        let instruction = match (&value.llvm_type, expected) {
            (LlvmType::I8, LlvmType::I16) => format!("sext i8 {} to i16", value.value),
            (LlvmType::I8, LlvmType::I32) => format!("sext i8 {} to i32", value.value),
            (LlvmType::I8, LlvmType::I64) => format!("sext i8 {} to i64", value.value),
            (LlvmType::I16, LlvmType::I32) => format!("sext i16 {} to i32", value.value),
            (LlvmType::I16, LlvmType::I64) => format!("sext i16 {} to i64", value.value),
            (LlvmType::I32, LlvmType::I64) => format!("sext i32 {} to i64", value.value),
            (LlvmType::Char16, LlvmType::I32) => format!("zext i16 {} to i32", value.value),
            (LlvmType::I32, LlvmType::I16) => format!("trunc i32 {} to i16", value.value),
            (LlvmType::I32, LlvmType::I8) => format!("trunc i32 {} to i8", value.value),
            (LlvmType::I64, LlvmType::I32) => format!("trunc i64 {} to i32", value.value),
            (LlvmType::I64, LlvmType::I16) => format!("trunc i64 {} to i16", value.value),
            (LlvmType::I64, LlvmType::I8) => format!("trunc i64 {} to i8", value.value),
            (LlvmType::I8, LlvmType::F32) => format!("sitofp i8 {} to float", value.value),
            (LlvmType::I8, LlvmType::F64) => format!("sitofp i8 {} to double", value.value),
            (LlvmType::I16, LlvmType::F32) => format!("sitofp i16 {} to float", value.value),
            (LlvmType::I16, LlvmType::F64) => format!("sitofp i16 {} to double", value.value),
            (LlvmType::I32, LlvmType::F32) => format!("sitofp i32 {} to float", value.value),
            (LlvmType::I32, LlvmType::F64) => format!("sitofp i32 {} to double", value.value),
            (LlvmType::I64, LlvmType::F32) => format!("sitofp i64 {} to float", value.value),
            (LlvmType::I64, LlvmType::F64) => format!("sitofp i64 {} to double", value.value),
            (LlvmType::F32, LlvmType::F64) => format!("fpext float {} to double", value.value),
            (LlvmType::F64, LlvmType::F32) => format!("fptrunc double {} to float", value.value),
            _ => return Err(type_mismatch(context, expected, &value.llvm_type)),
        };
        self.emit_line(format!("{} = {}", result, instruction));
        Ok(ExprValue {
            llvm_type: expected.clone(),
            value: result,
        })
    }

    fn box_trait_object(
        &mut self,
        trait_name: &str,
        concrete_name: &str,
        value: ExprValue,
        context: &str,
    ) -> Result<ExprValue, String> {
        let vtable = self.trait_vtable_symbol(trait_name, concrete_name, context)?;
        let data_ptr = self.box_value(value);
        self.trait_object_from_data_ptr(trait_name, data_ptr, &format!("@{vtable}"))
    }

    fn box_trait_intersection(
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
        let data_ptr = self.box_value(value);
        let intersection_type = LlvmType::TraitIntersection(trait_names.to_vec());
        let mut result = self.insert_pointer(&intersection_type, "zeroinitializer", &data_ptr, 0);
        for (index, vtable) in vtables.iter().enumerate() {
            result = self.insert_pointer(
                &intersection_type,
                &result,
                &format!("@{vtable}"),
                index + 1,
            );
        }
        Ok(ExprValue {
            llvm_type: intersection_type,
            value: result,
        })
    }

    fn trait_object_from_intersection(
        &mut self,
        trait_name: &str,
        actual_traits: &[String],
        value: ExprValue,
        context: &str,
    ) -> Result<ExprValue, String> {
        let index = actual_traits
            .iter()
            .position(|candidate| candidate == trait_name)
            .ok_or_else(|| {
                format!(
                    "type mismatch in {context}: intersection does not contain trait `{trait_name}`"
                )
            })?;
        let data_ptr = self.extract_pointer(&value, 0);
        let vtable_ptr = self.extract_pointer(&value, index + 1);
        self.trait_object_from_data_ptr(trait_name, data_ptr, &vtable_ptr)
    }

    fn project_trait_intersection(
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
                            "type mismatch in {context}: intersection does not contain trait `{trait_name}`"
                        )
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let projected_type = LlvmType::TraitIntersection(expected_traits.to_vec());
        let data_ptr = self.extract_pointer(&value, 0);
        let mut result = self.insert_pointer(&projected_type, "zeroinitializer", &data_ptr, 0);
        for (output_index, input_index) in indices.into_iter().enumerate() {
            let vtable_ptr = self.extract_pointer(&value, input_index + 1);
            result = self.insert_pointer(&projected_type, &result, &vtable_ptr, output_index + 1);
        }
        Ok(ExprValue {
            llvm_type: projected_type,
            value: result,
        })
    }

    fn box_union_value(
        &mut self,
        members: &[LlvmType],
        member_index: usize,
        value: ExprValue,
    ) -> Result<ExprValue, String> {
        let data_ptr = self.box_value(value);
        let union_type = LlvmType::Union(members.to_vec());
        let with_tag = self.next_temp();
        self.emit_line(format!(
            "{} = insertvalue {} zeroinitializer, i32 {}, 0",
            with_tag,
            union_type.ir(),
            member_index
        ));
        let result = self.insert_pointer(&union_type, &with_tag, &data_ptr, 1);
        Ok(ExprValue {
            llvm_type: union_type,
            value: result,
        })
    }

    fn box_value(&mut self, value: ExprValue) -> String {
        let allocator = self.next_temp();
        self.emit_line(format!(
            "{} = call ptr @skunk_system_allocator()",
            allocator
        ));
        let data_ptr = self.next_temp();
        self.emit_line(format!(
            "{} = call ptr @skunk_alloc_create(ptr {}, i64 {})",
            data_ptr,
            allocator,
            self.size_of(&value.llvm_type)
        ));
        self.emit_store(&data_ptr, &value);
        data_ptr
    }

    fn trait_vtable_symbol(
        &self,
        trait_name: &str,
        concrete_name: &str,
        context: &str,
    ) -> Result<String, String> {
        self.trait_vtables
            .get(&format!("{trait_name}=>{concrete_name}"))
            .cloned()
            .ok_or_else(|| {
                format!(
                    "type mismatch in {context}: `{concrete_name}` does not implement trait `{trait_name}`"
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
        let vtable = self.trait_vtable_symbol(trait_name, concrete_name, context)?;
        self.trait_object_from_data_ptr(trait_name, data_ptr, &format!("@{vtable}"))
    }

    fn trait_object_from_data_ptr(
        &mut self,
        trait_name: &str,
        data_ptr: String,
        vtable_ptr: &str,
    ) -> Result<ExprValue, String> {
        let trait_type = LlvmType::TraitObject(trait_name.to_string());
        let with_data = self.insert_pointer(&trait_type, "zeroinitializer", &data_ptr, 0);
        let result = self.insert_pointer(&trait_type, &with_data, vtable_ptr, 1);
        Ok(ExprValue {
            llvm_type: trait_type,
            value: result,
        })
    }

    fn insert_pointer(
        &mut self,
        aggregate_type: &LlvmType,
        aggregate: &str,
        pointer: &str,
        index: usize,
    ) -> String {
        let result = self.next_temp();
        self.emit_line(format!(
            "{} = insertvalue {} {}, ptr {}, {}",
            result,
            aggregate_type.ir(),
            aggregate,
            pointer,
            index
        ));
        result
    }

    fn extract_pointer(&mut self, aggregate: &ExprValue, index: usize) -> String {
        let result = self.next_temp();
        self.emit_line(format!(
            "{} = extractvalue {} {}, {}",
            result,
            aggregate.llvm_type.ir(),
            aggregate.value,
            index
        ));
        result
    }
}

fn is_comparison(operator: BinaryOperator) -> bool {
    matches!(
        operator,
        BinaryOperator::Equals
            | BinaryOperator::NotEquals
            | BinaryOperator::LessThan
            | BinaryOperator::LessThanOrEqual
            | BinaryOperator::GreaterThan
            | BinaryOperator::GreaterThanOrEqual
    )
}

fn type_mismatch(context: &str, expected: &LlvmType, actual: &LlvmType) -> String {
    format!(
        "type mismatch in {context}: expected `{}`, got `{}`",
        expected.ir(),
        actual.ir()
    )
}
