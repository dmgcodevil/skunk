//! Direct lowering of validated MIR control flow into LLVM instructions.
//!
//! Every executable function body reaches LLVM through this module. The MIR
//! validator owns structural invariants; instruction selection reports an
//! explicit error if a new MIR operation has no backend implementation yet.

use super::*;
use crate::ids::{FieldId, NodeId, VariantId};
use crate::mir as ir;
use crate::syntax::ast::{Literal as MirLiteral, UnaryOperator as MirUnaryOperator};
mod calls;
mod closure;
mod intrinsics;
mod place;
mod value;

mod support;
pub(super) use support::{CodegenContext, GlobalCodegenInfo};

impl<'a> FunctionCompiler<'a> {
    pub(super) fn compile_mir(
        self,
        function: &ir::Function,
        context: &CodegenContext<'_>,
    ) -> Result<Vec<String>, String> {
        self.compile_mir_with_env(function, context, None)
    }

    fn compile_mir_with_env(
        mut self,
        function: &ir::Function,
        context: &CodegenContext<'_>,
        environment_type: Option<&str>,
    ) -> Result<Vec<String>, String> {
        let receiver_local = mir_function_has_receiver(function, context.model)
            .then(|| function.parameters.first().copied())
            .flatten();
        let mut locals = Vec::with_capacity(function.locals.len());
        for local in &function.locals {
            let llvm_type = context.llvm_type(local.ty)?;
            if llvm_type == LlvmType::Void {
                return Err("MIR local cannot have LLVM void storage".to_string());
            }
            if receiver_local == Some(local.id) {
                locals.push(LocalVar {
                    ptr: "%arg0".to_string(),
                    llvm_type,
                });
                continue;
            }
            if let Some(capture_index) = function
                .captures
                .iter()
                .position(|capture| *capture == local.id)
            {
                let environment_type = environment_type.ok_or_else(|| {
                    "MIR closure capture is missing its environment type".to_string()
                })?;
                let field_ptr = self.next_temp();
                self.emit_line(format!(
                    "{} = getelementptr inbounds %env.{}, ptr %env, i32 0, i32 {}",
                    field_ptr,
                    sanitize_name(environment_type),
                    capture_index
                ));
                let capture_ptr = self.next_temp();
                self.emit_line(format!(
                    "{} = load ptr, ptr {}, align 8",
                    capture_ptr, field_ptr
                ));
                locals.push(LocalVar {
                    ptr: capture_ptr,
                    llvm_type,
                });
                continue;
            }
            let ptr = self.emit_heap_alloc(llvm_type.clone(), &format!("mir_{}", local.id.index()));
            let local_value = LocalVar {
                ptr,
                llvm_type: llvm_type.clone(),
            };
            if local.kind == ir::LocalKind::User {
                let initial = self.default_value(&llvm_type);
                self.emit_store(&local_value.ptr, &initial);
            }
            locals.push(local_value);
        }

        for (argument_index, parameter) in function.parameters.iter().enumerate() {
            if receiver_local == Some(*parameter) {
                continue;
            }
            let local = locals
                .get(parameter.index())
                .ok_or_else(|| format!("unknown MIR parameter local {}", parameter.index()))?;
            self.emit_line(format!(
                "store {} %arg{}, ptr {}, align {}",
                local.llvm_type.ir(),
                argument_index + usize::from(environment_type.is_some()),
                local.ptr,
                self.align_of(&local.llvm_type)
            ));
        }

        self.emit_line("; lowered directly from MIR".to_string());
        self.emit_line(format!("br label %mir_bb{}", function.entry.index()));
        for block in &function.blocks {
            self.emit_label(&format!("mir_bb{}", block.id.index()));
            for statement in &block.statements {
                self.compile_mir_statement(statement, &locals, context)?;
            }
            self.compile_mir_terminator(&block.terminator, &locals, context)?;
        }
        Ok(self.lines)
    }

    fn compile_mir_statement(
        &mut self,
        statement: &ir::Statement,
        locals: &[LocalVar],
        context: &CodegenContext<'_>,
    ) -> Result<(), String> {
        match &statement.kind {
            ir::StatementKind::Assign { destination, value } => {
                let destination = self.mir_place(destination, locals, context)?;
                let value = self.compile_mir_rvalue(value, locals, context)?;
                let value = self.coerce_expr(value, &destination.llvm_type, "MIR assignment")?;
                self.emit_store(&destination.ptr, &value);
                Ok(())
            }
            ir::StatementKind::Call {
                destination,
                target,
                argument_groups,
                result,
            } => {
                let (mut value, remaining_argument_groups) = match target {
                    ir::CallTarget::Operand(ir::Operand {
                        kind: ir::OperandKind::Definition(definition),
                        ..
                    })
                    | ir::CallTarget::Static(ir::StaticCallee::Definition(definition)) => {
                        let arguments = self.compile_mir_argument_group(
                            argument_groups.first().ok_or_else(|| {
                                "direct MIR call has no argument group".to_string()
                            })?,
                            locals,
                            context,
                        )?;
                        (
                            self.compile_mir_direct_call(*definition, None, arguments, context)?,
                            &argument_groups[1..],
                        )
                    }
                    ir::CallTarget::Method {
                        receiver,
                        method: ir::MethodCallee::Definition(definition),
                    } => {
                        let arguments = self.compile_mir_argument_group(
                            argument_groups.first().ok_or_else(|| {
                                "direct MIR method call has no argument group".to_string()
                            })?,
                            locals,
                            context,
                        )?;
                        let receiver = self.compile_mir_receiver(receiver, locals, context)?;
                        (
                            self.compile_mir_direct_call(
                                *definition,
                                Some(receiver),
                                arguments,
                                context,
                            )?,
                            &argument_groups[1..],
                        )
                    }
                    ir::CallTarget::Method {
                        receiver,
                        method: ir::MethodCallee::Dynamic { method, .. },
                    } => {
                        let arguments = self.compile_mir_argument_group(
                            argument_groups.first().ok_or_else(|| {
                                "dynamic MIR method call has no argument group".to_string()
                            })?,
                            locals,
                            context,
                        )?;
                        let receiver = self.compile_mir_operand(receiver, locals, context)?;
                        (
                            self.compile_mir_dynamic_call(receiver, *method, arguments, context)?,
                            &argument_groups[1..],
                        )
                    }
                    ir::CallTarget::Method {
                        receiver,
                        method: ir::MethodCallee::Intrinsic { name, .. },
                    } => {
                        let arguments = self.compile_mir_argument_group(
                            argument_groups.first().ok_or_else(|| {
                                "intrinsic MIR method call has no argument group".to_string()
                            })?,
                            locals,
                            context,
                        )?;
                        let receiver = self.compile_mir_operand(receiver, locals, context)?;
                        (
                            self.compile_mir_method_intrinsic(receiver, name, arguments)?,
                            &argument_groups[1..],
                        )
                    }
                    ir::CallTarget::Static(ir::StaticCallee::Variant(variant)) => {
                        let arguments = self.compile_mir_argument_group(
                            argument_groups.first().ok_or_else(|| {
                                "direct MIR variant call has no argument group".to_string()
                            })?,
                            locals,
                            context,
                        )?;
                        (
                            self.compile_mir_variant(
                                *variant,
                                arguments,
                                &context.llvm_type(*result)?,
                                context,
                            )?,
                            &argument_groups[1..],
                        )
                    }
                    ir::CallTarget::Static(ir::StaticCallee::Intrinsic { owner, name }) => {
                        let arguments = self.compile_mir_argument_group(
                            argument_groups.first().ok_or_else(|| {
                                "direct MIR intrinsic call has no argument group".to_string()
                            })?,
                            locals,
                            context,
                        )?;
                        (
                            self.compile_mir_static_intrinsic(
                                *owner,
                                name,
                                arguments,
                                &context.llvm_type(*result)?,
                                context,
                            )?,
                            &argument_groups[1..],
                        )
                    }
                    ir::CallTarget::Operand(callee) => {
                        let mut current = self.compile_mir_operand(callee, locals, context)?;
                        for group in argument_groups {
                            let arguments =
                                self.compile_mir_argument_group(group, locals, context)?;
                            current = self.compile_mir_closure_call(current, arguments)?;
                        }
                        (current, &argument_groups[argument_groups.len()..])
                    }
                };
                for group in remaining_argument_groups {
                    let arguments = self.compile_mir_argument_group(group, locals, context)?;
                    value = self.compile_mir_closure_call(value, arguments)?;
                }
                if let Some(destination) = destination {
                    let destination = self.mir_place(destination, locals, context)?;
                    let value =
                        self.coerce_expr(value, &destination.llvm_type, "MIR call result")?;
                    self.emit_store(&destination.ptr, &value);
                }
                Ok(())
            }
            ir::StatementKind::Print(operand) => {
                let value = self.compile_mir_operand(operand, locals, context)?;
                self.compile_print_value(value)
            }
            ir::StatementKind::Input => Ok(()),
        }
    }

    fn compile_mir_argument_group(
        &mut self,
        arguments: &[ir::Operand],
        locals: &[LocalVar],
        context: &CodegenContext<'_>,
    ) -> Result<Vec<ExprValue>, String> {
        arguments
            .iter()
            .map(|argument| self.compile_mir_operand(argument, locals, context))
            .collect()
    }

    fn compile_mir_terminator(
        &mut self,
        terminator: &ir::Terminator,
        locals: &[LocalVar],
        context: &CodegenContext<'_>,
    ) -> Result<(), String> {
        match &terminator.kind {
            ir::TerminatorKind::Goto { target } => {
                self.emit_line(format!("br label %mir_bb{}", target.index()));
            }
            ir::TerminatorKind::If {
                condition,
                then_target,
                else_target,
            } => {
                let condition = self.compile_mir_operand(condition, locals, context)?;
                if condition.llvm_type != LlvmType::I1 {
                    return Err("MIR branch condition is not boolean".to_string());
                }
                self.emit_line(format!(
                    "br i1 {}, label %mir_bb{}, label %mir_bb{}",
                    condition.value,
                    then_target.index(),
                    else_target.index()
                ));
            }
            ir::TerminatorKind::Return(value) => match value {
                Some(value) => {
                    let value = self.compile_mir_operand(value, locals, context)?;
                    let return_type = self.return_type.clone();
                    let value = self.coerce_expr(value, &return_type, "MIR return")?;
                    self.emit_line(format!("ret {} {}", value.llvm_type.ir(), value.value));
                }
                None => self.emit_line("ret void".to_string()),
            },
            ir::TerminatorKind::Unreachable => self.emit_line("unreachable".to_string()),
            ir::TerminatorKind::SwitchEnum {
                discriminator,
                targets,
                otherwise,
            } => {
                let discriminator = self.compile_mir_operand(discriminator, locals, context)?;
                if !matches!(discriminator.llvm_type, LlvmType::Enum(_)) {
                    return Err(format!(
                        "MIR enum switch requires an enum, found `{}`",
                        discriminator.llvm_type.ir()
                    ));
                }
                let tag = self.next_temp();
                self.emit_line(format!(
                    "{} = extractvalue {} {}, 0",
                    tag,
                    discriminator.llvm_type.ir(),
                    discriminator.value
                ));
                let targets = targets
                    .iter()
                    .map(|(variant, target)| {
                        context
                            .variants
                            .get(variant)
                            .map(|info| {
                                format!("i32 {}, label %mir_bb{}", info.tag, target.index())
                            })
                            .ok_or_else(|| format!("unknown MIR variant {}", variant.index()))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                self.emit_line(format!(
                    "switch i32 {}, label %mir_bb{} [ {} ]",
                    tag,
                    otherwise.index(),
                    targets.join(" ")
                ));
            }
        }
        Ok(())
    }
}

fn format_float(value: f64) -> String {
    let mut formatted = value.to_string();
    if !formatted.contains('.') && !formatted.contains('e') && !formatted.contains('E') {
        formatted.push_str(".0");
    }
    formatted
}
