use super::*;

impl<'a> FunctionLowerer<'a> {
    pub(super) fn lower_expression(&mut self, expression: &'a hir::Expr) -> Option<Operand> {
        match &expression.kind {
            hir::ExprKind::Literal(literal) => Some(Operand {
                ty: expression.ty,
                kind: OperandKind::Constant(Constant::Literal(literal.clone())),
            }),
            hir::ExprKind::Value(hir::Value::Definition(definition)) => {
                Some(self.definition_operand(*definition, expression.ty))
            }
            hir::ExprKind::Value(hir::Value::Local(local)) => {
                let mir_local = self.source_locals.get(local).copied();
                match mir_local {
                    Some(local) => Some(Operand {
                        ty: expression.ty,
                        kind: OperandKind::Copy(Place::local(local)),
                    }),
                    None => {
                        self.diagnostics.push(
                            Diagnostic::error(format!(
                                "HIR local {} is used before it is introduced in MIR",
                                local.index()
                            ))
                            .with_code("E5003")
                            .at(expression.span),
                        );
                        None
                    }
                }
            }
            hir::ExprKind::Unary { operator, operand } => match operator {
                crate::syntax::ast::UnaryOperator::Dereference => {
                    let operand = self.lower_expression(operand)?;
                    let mut place =
                        self.operand_into_place(operand, expression.source, expression.span)?;
                    place.projections.push(Projection {
                        ty: expression.ty,
                        kind: ProjectionKind::Dereference,
                    });
                    Some(Operand {
                        ty: expression.ty,
                        kind: OperandKind::Copy(place),
                    })
                }
                crate::syntax::ast::UnaryOperator::AddressOf
                | crate::syntax::ast::UnaryOperator::AddressOfMut => {
                    let place = self.lower_place(operand)?;
                    self.temporary(
                        expression,
                        RvalueKind::Reference {
                            mutable: matches!(
                                operator,
                                crate::syntax::ast::UnaryOperator::AddressOfMut
                            ),
                            place,
                        },
                    )
                }
                _ => {
                    let operand = self.lower_expression(operand)?;
                    self.temporary(
                        expression,
                        RvalueKind::Unary {
                            operator: *operator,
                            operand,
                        },
                    )
                }
            },
            hir::ExprKind::Binary {
                left,
                operator,
                right,
            } => {
                let left = self.lower_expression(left)?;
                let right = self.lower_expression(right)?;
                self.temporary(
                    expression,
                    RvalueKind::Binary {
                        left,
                        operator: *operator,
                        right,
                    },
                )
            }
            hir::ExprKind::Call {
                callee,
                argument_groups,
            } => {
                let callee = self.lower_expression(callee)?;
                let argument_groups = self.lower_argument_groups(argument_groups)?;
                self.call(expression, CallTarget::Operand(callee), argument_groups)
            }
            hir::ExprKind::MethodCall {
                receiver,
                method,
                argument_groups,
            } => {
                let receiver = self.lower_expression(receiver)?;
                let argument_groups = self.lower_argument_groups(argument_groups)?;
                self.call(
                    expression,
                    CallTarget::Method {
                        receiver,
                        method: lower_method_target(method),
                    },
                    argument_groups,
                )
            }
            hir::ExprKind::Field { .. } | hir::ExprKind::Index { .. } => {
                let place = self.lower_place(expression)?;
                Some(Operand {
                    ty: expression.ty,
                    kind: OperandKind::Copy(place),
                })
            }
            hir::ExprKind::Length { receiver } => {
                let receiver = self.lower_expression(receiver)?;
                self.temporary(expression, RvalueKind::Length(receiver))
            }
            hir::ExprKind::Slice {
                receiver,
                start,
                end,
            } => {
                let receiver = self.lower_expression(receiver)?;
                let start = match start.as_deref() {
                    Some(value) => Some(self.lower_expression(value)?),
                    None => None,
                };
                let end = match end.as_deref() {
                    Some(value) => Some(self.lower_expression(value)?),
                    None => None,
                };
                self.temporary(
                    expression,
                    RvalueKind::Slice {
                        receiver,
                        start,
                        end,
                    },
                )
            }
            hir::ExprKind::StructInit { definition, fields } => {
                let fields = fields
                    .iter()
                    .map(|(field, value)| self.lower_expression(value).map(|value| (*field, value)))
                    .collect::<Option<Vec<_>>>()?;
                self.temporary(
                    expression,
                    RvalueKind::Aggregate(Aggregate::Struct {
                        definition: *definition,
                        fields,
                    }),
                )
            }
            hir::ExprKind::StaticCall { target, arguments } => {
                let arguments = arguments
                    .iter()
                    .map(|argument| self.lower_expression(argument))
                    .collect::<Option<Vec<_>>>()?;
                self.call(
                    expression,
                    CallTarget::Static(lower_static_target(target)),
                    vec![arguments],
                )
            }
            hir::ExprKind::Array(elements) => {
                let elements = elements
                    .iter()
                    .map(|element| self.lower_expression(element))
                    .collect::<Option<Vec<_>>>()?;
                self.temporary(
                    expression,
                    RvalueKind::Aggregate(Aggregate::Array(elements)),
                )
            }
            hir::ExprKind::Block(block) => {
                self.lower_block(block);
                Some(Operand {
                    ty: expression.ty,
                    kind: OperandKind::Constant(Constant::Unit),
                })
            }
            hir::ExprKind::Lambda(function) => self.lower_closure(expression, function),
        }
    }

    pub(super) fn lower_place(&mut self, expression: &'a hir::Expr) -> Option<Place> {
        match &expression.kind {
            hir::ExprKind::Value(hir::Value::Local(local)) => self
                .source_locals
                .get(local)
                .copied()
                .map(Place::local)
                .or_else(|| {
                    self.diagnostics.push(
                        Diagnostic::error(format!(
                            "assignment target local {} is not declared in MIR",
                            local.index()
                        ))
                        .with_code("E5003")
                        .at(expression.span),
                    );
                    None
                }),
            hir::ExprKind::Value(hir::Value::Definition(definition))
                if self.definition_is_global(*definition) =>
            {
                Some(Place {
                    base: PlaceBase::Definition(*definition),
                    projections: Vec::new(),
                })
            }
            hir::ExprKind::Unary {
                operator: crate::syntax::ast::UnaryOperator::Dereference,
                operand,
            } => {
                let operand = self.lower_expression(operand)?;
                let mut place =
                    self.operand_into_place(operand, expression.source, expression.span)?;
                place.projections.push(Projection {
                    ty: expression.ty,
                    kind: ProjectionKind::Dereference,
                });
                Some(place)
            }
            hir::ExprKind::Field { receiver, field } => {
                let receiver = self.lower_expression(receiver)?;
                let receiver_ty = receiver.ty;
                let mut place =
                    self.operand_into_place(receiver, expression.source, expression.span)?;
                self.append_auto_dereferences(&mut place, receiver_ty, true);
                place.projections.push(Projection {
                    ty: expression.ty,
                    kind: ProjectionKind::Field(*field),
                });
                Some(place)
            }
            hir::ExprKind::Index {
                receiver,
                coordinates,
            } => {
                let receiver = self.lower_expression(receiver)?;
                let receiver_ty = receiver.ty;
                let coordinates = coordinates
                    .iter()
                    .map(|coordinate| self.lower_expression(coordinate))
                    .collect::<Option<Vec<_>>>()?;
                let mut place =
                    self.operand_into_place(receiver, expression.source, expression.span)?;
                // A pointer index is pointer arithmetic, not `(*pointer)[index]`.
                // References still need an explicit dereference to expose the
                // array, slice, or pointer value they borrow.
                self.append_auto_dereferences(&mut place, receiver_ty, false);
                place.projections.push(Projection {
                    ty: expression.ty,
                    kind: ProjectionKind::Index(coordinates),
                });
                Some(place)
            }
            _ => {
                self.diagnostics.push(
                    Diagnostic::error("expression is not an addressable MIR place")
                        .with_code("E5005")
                        .at(expression.span),
                );
                None
            }
        }
    }

    pub(super) fn operand_into_place(
        &mut self,
        operand: Operand,
        source: NodeId,
        span: Span,
    ) -> Option<Place> {
        if let OperandKind::Copy(place) = &operand.kind {
            return Some(place.clone());
        }
        let ty = operand.ty;
        let local = self.allocate_local(None, ty, true, LocalKind::Temporary, span)?;
        let place = Place::local(local);
        self.assign(
            place.clone(),
            Rvalue {
                ty,
                kind: RvalueKind::Use(operand),
            },
            source,
            span,
        );
        Some(place)
    }

    pub(super) fn append_auto_dereferences(
        &self,
        place: &mut Place,
        mut ty: TypeId,
        dereference_pointers: bool,
    ) {
        loop {
            let Some(kind) =
                (ty.index() < self.model.types.len()).then(|| self.model.types.kind(ty))
            else {
                return;
            };
            match kind {
                TypeKind::Const(inner) => ty = *inner,
                TypeKind::Pointer(target) if dereference_pointers => {
                    ty = *target;
                    place.projections.push(Projection {
                        ty,
                        kind: ProjectionKind::Dereference,
                    });
                }
                TypeKind::Reference { target, .. } => {
                    ty = *target;
                    place.projections.push(Projection {
                        ty,
                        kind: ProjectionKind::Dereference,
                    });
                }
                _ => return,
            }
        }
    }

    fn lower_argument_groups(&mut self, groups: &'a [Vec<hir::Expr>]) -> Option<Vec<Vec<Operand>>> {
        groups
            .iter()
            .map(|group| {
                group
                    .iter()
                    .map(|argument| self.lower_expression(argument))
                    .collect::<Option<Vec<_>>>()
            })
            .collect()
    }

    pub(super) fn temporary(
        &mut self,
        expression: &'a hir::Expr,
        kind: RvalueKind,
    ) -> Option<Operand> {
        let local = self.allocate_local(
            None,
            expression.ty,
            true,
            LocalKind::Temporary,
            expression.span,
        )?;
        self.assign(
            Place::local(local),
            Rvalue {
                ty: expression.ty,
                kind,
            },
            expression.source,
            expression.span,
        );
        Some(Operand {
            ty: expression.ty,
            kind: OperandKind::Copy(Place::local(local)),
        })
    }

    fn call(
        &mut self,
        expression: &'a hir::Expr,
        target: CallTarget,
        argument_groups: Vec<Vec<Operand>>,
    ) -> Option<Operand> {
        let destination = if self.type_is_void(expression.ty) {
            None
        } else {
            Some(Place::local(self.allocate_local(
                None,
                expression.ty,
                true,
                LocalKind::Temporary,
                expression.span,
            )?))
        };
        let result = destination.as_ref().map(|place| Operand {
            ty: expression.ty,
            kind: OperandKind::Copy(place.clone()),
        });
        self.emit(
            Statement {
                source: expression.source,
                span: expression.span,
                kind: StatementKind::Call {
                    destination,
                    target,
                    argument_groups,
                    result: expression.ty,
                },
            },
            expression.span,
        );
        if let Some(result) = result {
            Some(result)
        } else {
            Some(Operand {
                ty: expression.ty,
                kind: OperandKind::Constant(Constant::Unit),
            })
        }
    }
}
