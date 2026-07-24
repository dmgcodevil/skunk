use super::*;

impl<'a> FunctionLowerer<'a> {
    pub(super) fn lower_for(&mut self, loop_statement: &'a hir::For, source: NodeId, span: Span) {
        if let Some(initializer) = loop_statement.initializer.as_deref() {
            self.lower_statement(initializer);
        }
        if !self.current_is_open_and_reachable() {
            return;
        }

        let Some(condition_target) = self.new_block(span) else {
            return;
        };
        let Some(body_target) = self.new_block(loop_statement.body.span) else {
            return;
        };
        let Some(update_target) = self.new_block(span) else {
            return;
        };
        let Some(exit_target) = self.new_block(span) else {
            return;
        };

        self.terminate(Terminator {
            source,
            span,
            kind: TerminatorKind::Goto {
                target: condition_target,
            },
        });

        self.current = condition_target;
        if let Some(condition) = &loop_statement.condition {
            let Some(condition) = self.lower_expression(condition) else {
                return;
            };
            self.terminate(Terminator {
                source,
                span,
                kind: TerminatorKind::If {
                    condition,
                    then_target: body_target,
                    else_target: exit_target,
                },
            });
        } else {
            self.terminate(Terminator {
                source,
                span,
                kind: TerminatorKind::Goto {
                    target: body_target,
                },
            });
        }

        self.current = body_target;
        self.lower_block(&loop_statement.body);
        self.goto_if_open(update_target, source, span);

        self.current = update_target;
        if self.current_is_open_and_reachable() {
            if let Some(update) = loop_statement.update.as_deref() {
                self.lower_statement(update);
            }
            self.goto_if_open(condition_target, source, span);
        }

        self.current = exit_target;
    }

    pub(super) fn lower_destructure(
        &mut self,
        value: &'a hir::Expr,
        bindings: &[(crate::ids::FieldId, LocalId)],
        source: NodeId,
        span: Span,
    ) {
        let Some(value) = self.lower_expression(value) else {
            return;
        };
        let value_ty = value.ty;
        let Some(mut source_place) = self.operand_into_place(value, source, span) else {
            return;
        };
        self.append_auto_dereferences(&mut source_place, value_ty, true);

        for (field, source_local) in bindings {
            let Some(ty) = self.model.local_types.get(source_local).copied() else {
                self.diagnostics.push(
                    Diagnostic::error(format!(
                        "destructure binding {} has no semantic type",
                        source_local.index()
                    ))
                    .with_code("E5008")
                    .at(span),
                );
                continue;
            };
            let mutable = self.source_local_is_mutable(*source_local);
            let Some(destination) =
                self.register_source_local(*source_local, ty, mutable, LocalKind::User, span)
            else {
                continue;
            };
            let mut field_place = source_place.clone();
            field_place.projections.push(Projection {
                ty,
                kind: ProjectionKind::Field(*field),
            });
            self.assign(
                Place::local(destination),
                Rvalue {
                    ty,
                    kind: RvalueKind::Use(Operand {
                        ty,
                        kind: OperandKind::Copy(field_place),
                    }),
                },
                source,
                span,
            );
        }
    }

    pub(super) fn lower_match(&mut self, branch: &'a hir::Match, source: NodeId, span: Span) {
        let Some(first) = branch.cases.first() else {
            self.diagnostics.push(
                Diagnostic::error("MIR lowering requires at least one match case")
                    .with_code("E5009")
                    .at(span),
            );
            return;
        };
        match first.pattern {
            hir::MatchPattern::EnumVariant { .. } => self.lower_enum_match(branch, source, span),
            hir::MatchPattern::Struct { .. } => self.lower_struct_match(branch, source, span),
        }
    }

    fn lower_enum_match(&mut self, branch: &'a hir::Match, source: NodeId, span: Span) {
        let Some(value) = self.lower_expression(&branch.value) else {
            return;
        };
        let value_ty = value.ty;
        let Some(value_place) = self.operand_into_place(value, source, span) else {
            return;
        };
        let Some(join_target) = self.new_block(span) else {
            return;
        };
        let Some(otherwise) = self.new_block(span) else {
            return;
        };
        let mut case_targets = Vec::with_capacity(branch.cases.len());
        for case in &branch.cases {
            let hir::MatchPattern::EnumVariant { variant } = case.pattern else {
                self.mixed_match_patterns(span);
                return;
            };
            let Some(target) = self.new_block(case.body.span) else {
                return;
            };
            case_targets.push((variant, target));
        }

        self.terminate(Terminator {
            source,
            span,
            kind: TerminatorKind::SwitchEnum {
                discriminator: Operand {
                    ty: value_ty,
                    kind: OperandKind::Copy(value_place.clone()),
                },
                targets: case_targets.clone(),
                otherwise,
            },
        });

        for (case, (variant, target)) in branch.cases.iter().zip(case_targets) {
            self.current = target;
            self.bind_variant_payloads(case, variant, &value_place, source, span);
            self.lower_block(&case.body);
            self.goto_if_open(join_target, source, span);
        }

        self.current = otherwise;
        self.terminate(Terminator {
            source,
            span,
            kind: TerminatorKind::Unreachable,
        });
        self.current = join_target;
    }

    fn lower_struct_match(&mut self, branch: &'a hir::Match, source: NodeId, span: Span) {
        if branch.cases.len() != 1 {
            self.diagnostics.push(
                Diagnostic::error("struct matches require exactly one irrefutable case")
                    .with_code("E5010")
                    .at(span),
            );
            return;
        }
        let case = &branch.cases[0];
        let hir::MatchPattern::Struct { fields, .. } = &case.pattern else {
            self.mixed_match_patterns(span);
            return;
        };
        let Some(value) = self.lower_expression(&branch.value) else {
            return;
        };
        let value_ty = value.ty;
        let Some(mut value_place) = self.operand_into_place(value, source, span) else {
            return;
        };
        self.append_auto_dereferences(&mut value_place, value_ty, true);
        let Some(case_target) = self.new_block(case.body.span) else {
            return;
        };
        let Some(join_target) = self.new_block(span) else {
            return;
        };
        self.terminate(Terminator {
            source,
            span,
            kind: TerminatorKind::Goto {
                target: case_target,
            },
        });

        self.current = case_target;
        for (field, binding) in fields.iter().zip(&case.bindings) {
            self.bind_pattern_local(
                *binding,
                ProjectionKind::Field(*field),
                &value_place,
                source,
                span,
            );
        }
        self.lower_block(&case.body);
        self.goto_if_open(join_target, source, span);
        self.current = join_target;
    }

    fn bind_variant_payloads(
        &mut self,
        case: &hir::MatchCase,
        variant: crate::ids::VariantId,
        value_place: &Place,
        source: NodeId,
        span: Span,
    ) {
        for (index, binding) in case.bindings.iter().enumerate() {
            let Ok(index) = u32::try_from(index) else {
                self.diagnostics.push(
                    Diagnostic::error("enum variant contains too many payload bindings")
                        .with_code("E5001")
                        .at(span),
                );
                return;
            };
            self.bind_pattern_local(
                *binding,
                ProjectionKind::VariantField { variant, index },
                value_place,
                source,
                span,
            );
        }
    }

    fn bind_pattern_local(
        &mut self,
        source_local: LocalId,
        projection: ProjectionKind,
        value_place: &Place,
        source: NodeId,
        span: Span,
    ) {
        let Some(ty) = self.model.local_types.get(&source_local).copied() else {
            self.diagnostics.push(
                Diagnostic::error(format!(
                    "pattern binding {} has no semantic type",
                    source_local.index()
                ))
                .with_code("E5008")
                .at(span),
            );
            return;
        };
        let mutable = self.source_local_is_mutable(source_local);
        let Some(destination) =
            self.register_source_local(source_local, ty, mutable, LocalKind::User, span)
        else {
            return;
        };
        let mut projected = value_place.clone();
        projected.projections.push(Projection {
            ty,
            kind: projection,
        });
        self.assign(
            Place::local(destination),
            Rvalue {
                ty,
                kind: RvalueKind::Use(Operand {
                    ty,
                    kind: OperandKind::Copy(projected),
                }),
            },
            source,
            span,
        );
    }

    fn source_local_is_mutable(&self, local: LocalId) -> bool {
        self.model
            .resolutions
            .locals
            .get(local.index())
            .is_none_or(|record| !record.is_const)
    }

    fn mixed_match_patterns(&mut self, span: Span) {
        self.diagnostics.push(
            Diagnostic::error("one MIR match cannot mix enum and struct patterns")
                .with_code("E5011")
                .at(span),
        );
    }
}
