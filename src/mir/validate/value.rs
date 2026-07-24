use super::*;

impl Validator<'_> {
    pub(super) fn rvalue(&mut self, function: &Function, value: &Rvalue, span: Span) {
        self.ty(value.ty, span);
        match &value.kind {
            RvalueKind::Use(operand) => {
                self.operand(function, operand, span);
                if value.ty != operand.ty {
                    self.diagnostics.push(
                        Diagnostic::error("MIR use rvalue changes its operand type")
                            .with_code("E5110")
                            .at(span),
                    );
                }
            }
            RvalueKind::Coerce(operand) => {
                self.operand(function, operand, span);
                if operand.ty == value.ty || !self.can_coerce(operand.ty, value.ty) {
                    self.diagnostics.push(
                        Diagnostic::error("MIR coercion has incompatible source and result types")
                            .with_code("E5179")
                            .at(span),
                    );
                }
            }
            RvalueKind::Unary { operand, .. } => self.operand(function, operand, span),
            RvalueKind::Binary { left, right, .. } => {
                self.operand(function, left, span);
                self.operand(function, right, span);
            }
            RvalueKind::Reference { mutable, place } => {
                let target = self.place(function, place, span);
                if target.is_some_and(|target| !self.reference_matches(value.ty, target, *mutable))
                {
                    self.diagnostics.push(
                        Diagnostic::error("MIR reference rvalue has an incompatible result type")
                            .with_code("E5121")
                            .at(span),
                    );
                }
            }
            RvalueKind::Length(receiver) => {
                self.operand(function, receiver, span);
                if !self.is_int(value.ty) || !self.is_sequence(receiver.ty) {
                    self.diagnostics.push(
                        Diagnostic::error(
                            "MIR length rvalue has incompatible operand or result types",
                        )
                        .with_code("E5122")
                        .at(span),
                    );
                }
            }
            RvalueKind::Slice {
                receiver,
                start,
                end,
            } => {
                self.operand(function, receiver, span);
                for bound in [start, end].into_iter().flatten() {
                    self.operand(function, bound, span);
                    if !self.is_integral(bound.ty) {
                        self.diagnostics.push(
                            Diagnostic::error("MIR slice bound is not an integer")
                                .with_code("E5123")
                                .at(span),
                        );
                    }
                }
                if !self.slice_matches(receiver.ty, value.ty) {
                    self.diagnostics.push(
                        Diagnostic::error("MIR slice result does not match its receiver")
                            .with_code("E5124")
                            .at(span),
                    );
                }
            }
            RvalueKind::Aggregate(aggregate) => self.aggregate(function, aggregate, value.ty, span),
            RvalueKind::Closure {
                function: closure,
                captures,
            } => self.closure(function, *closure, captures, value.ty, span),
        }
    }

    fn closure(
        &mut self,
        parent: &Function,
        closure: NodeId,
        captures: &[Place],
        result: TypeId,
        span: Span,
    ) {
        let Some(shape) = self.functions.get(&closure).cloned() else {
            self.diagnostics.push(
                Diagnostic::error("MIR closure references an unknown nested function")
                    .with_code("E5152")
                    .at(span),
            );
            return;
        };
        if shape.origin != FunctionOrigin::Closure {
            self.diagnostics.push(
                Diagnostic::error("MIR closure does not reference an anonymous function")
                    .with_code("E5182")
                    .at(span),
            );
        }
        if captures.len() != shape.captures.len() {
            self.diagnostics.push(
                Diagnostic::error("MIR closure capture count does not match its nested function")
                    .with_code("E5153")
                    .at(span),
            );
        }
        for (capture, expected) in captures.iter().zip(&shape.captures) {
            if self
                .place(parent, capture, span)
                .zip(*expected)
                .is_some_and(|(actual, expected)| actual != expected)
            {
                self.diagnostics.push(
                    Diagnostic::error("MIR closure capture has the wrong type")
                        .with_code("E5154")
                        .at(span),
                );
            }
        }
        match self.valid_type_kind(result) {
            Some(TypeKind::Function { parameters, result }) => {
                if parameters.len() != shape.parameters.len()
                    || parameters
                        .iter()
                        .zip(&shape.parameters)
                        .any(|(actual, expected)| {
                            expected.is_some_and(|expected| *actual != expected)
                        })
                    || *result != shape.result
                {
                    self.diagnostics.push(
                        Diagnostic::error(
                            "MIR closure type does not match its nested function signature",
                        )
                        .with_code("E5155")
                        .at(span),
                    );
                }
            }
            _ => self.diagnostics.push(
                Diagnostic::error("MIR closure rvalue does not have a function type")
                    .with_code("E5155")
                    .at(span),
            ),
        }
    }

    pub(super) fn call_target(
        &mut self,
        function: &Function,
        target: &CallTarget,
        groups: &[Vec<Operand>],
        result: TypeId,
        span: Span,
    ) {
        match target {
            CallTarget::Operand(callee) => {
                self.operand(function, callee, span);
                self.call_shape(callee.ty, groups, result, span);
            }
            CallTarget::Method { receiver, method } => {
                self.operand(function, receiver, span);
                match method {
                    MethodCallee::Definition(definition) => {
                        if let Some(signature) = self.definition_type(*definition, span) {
                            self.method_call_shape(signature, groups, result, span);
                        }
                    }
                    MethodCallee::Dynamic { owner, method } => {
                        self.ty(*owner, span);
                        if let Some(signature) = self.definition_type(*method, span) {
                            self.method_call_shape(signature, groups, result, span);
                        }
                    }
                    MethodCallee::Intrinsic { owner, .. } => self.ty(*owner, span),
                }
            }
            CallTarget::Static(target) => match target {
                StaticCallee::Definition(definition) => {
                    if let Some(signature) = self.definition_type(*definition, span) {
                        self.call_shape(signature, groups, result, span);
                    }
                }
                StaticCallee::Variant(variant) => {
                    if let Some(payload) = self.model.variant_payloads.get(variant) {
                        if groups.len() != 1 || groups[0].len() != payload.len() {
                            self.diagnostics.push(
                                Diagnostic::error(
                                    "MIR enum constructor arguments do not match its payload",
                                )
                                .with_code("E5136")
                                .at(span),
                            );
                        }
                        if self.model.variant_owners.get(variant).copied()
                            != self.nominal_definition(result)
                        {
                            self.diagnostics.push(
                                Diagnostic::error(
                                    "MIR enum constructor result belongs to a different enum",
                                )
                                .with_code("E5145")
                                .at(span),
                            );
                        }
                    } else {
                        self.invalid_id("variant", variant.index(), span);
                    }
                }
                StaticCallee::Intrinsic { owner, .. } => self.ty(*owner, span),
            },
        }
    }

    fn method_call_shape(
        &mut self,
        signature: TypeId,
        groups: &[Vec<Operand>],
        result: TypeId,
        span: Span,
    ) {
        let Some(TypeKind::Function {
            parameters,
            result: method_result,
        }) = self.valid_type_kind(signature)
        else {
            self.diagnostics.push(
                Diagnostic::error("MIR method target does not have a function type")
                    .with_code("E5111")
                    .at(span),
            );
            return;
        };
        let expected_argument_count = parameters.len().saturating_sub(1);
        let method_result = *method_result;
        let Some(first_group) = groups.first() else {
            self.diagnostics.push(
                Diagnostic::error("MIR method call has no argument group")
                    .with_code("E5112")
                    .at(span),
            );
            return;
        };
        if first_group.len() != expected_argument_count {
            self.diagnostics.push(
                Diagnostic::error("MIR method argument count does not match its function type")
                    .with_code("E5112")
                    .at(span),
            );
        }
        self.call_shape(method_result, &groups[1..], result, span);
    }

    fn aggregate(
        &mut self,
        function: &Function,
        aggregate: &Aggregate,
        result: TypeId,
        span: Span,
    ) {
        match aggregate {
            Aggregate::Struct { definition, fields } => {
                self.definition(*definition, span);
                if !matches!(
                    self.valid_type_kind(result),
                    Some(TypeKind::Nominal {
                        definition: result_definition,
                        ..
                    }) if result_definition == definition
                ) {
                    self.diagnostics.push(
                        Diagnostic::error("MIR struct aggregate has the wrong nominal result type")
                            .with_code("E5125")
                            .at(span),
                    );
                }
                let mut seen = HashSet::new();
                for (field, value) in fields {
                    if !self.model.field_types.contains_key(field) {
                        self.invalid_id("field", field.index(), span);
                    } else if self.model.field_owners.get(field) != Some(definition) {
                        self.diagnostics.push(
                            Diagnostic::error(
                                "MIR struct aggregate contains a field from another struct",
                            )
                            .with_code("E5146")
                            .at(span),
                        );
                    }
                    if !seen.insert(*field) {
                        self.diagnostics.push(
                            Diagnostic::error("MIR struct aggregate initializes a field twice")
                                .with_code("E5126")
                                .at(span),
                        );
                    }
                    self.operand(function, value, span);
                }
                let expected_fields = self
                    .model
                    .field_owners
                    .values()
                    .filter(|owner| **owner == *definition)
                    .count();
                if seen.len() != expected_fields {
                    self.diagnostics.push(
                        Diagnostic::error("MIR struct aggregate does not initialize every field")
                            .with_code("E5147")
                            .at(span),
                    );
                }
            }
            Aggregate::Array(elements) => {
                let result_kind = self.valid_type_kind(result).cloned();
                match &result_kind {
                    Some(TypeKind::Array { dimensions, .. }) => {
                        if dimensions
                            .first()
                            .is_some_and(|length| *length != elements.len() as u64)
                        {
                            self.diagnostics.push(
                                Diagnostic::error(
                                    "MIR array aggregate length does not match its type",
                                )
                                .with_code("E5127")
                                .at(span),
                            );
                        }
                    }
                    Some(TypeKind::Slice(_)) => {}
                    _ => {
                        self.diagnostics.push(
                            Diagnostic::error(
                                "MIR array aggregate does not have an array-like type",
                            )
                            .with_code("E5128")
                            .at(span),
                        );
                    }
                }
                for element in elements {
                    self.operand(function, element, span);
                    let matches = match &result_kind {
                        Some(TypeKind::Array {
                            element: expected,
                            dimensions,
                        }) => self.array_element_matches(*expected, dimensions, element.ty),
                        Some(TypeKind::Slice(expected)) => *expected == element.ty,
                        _ => true,
                    };
                    if !matches {
                        self.diagnostics.push(
                            Diagnostic::error("MIR array element does not match its element type")
                                .with_code("E5129")
                                .at(span),
                        );
                    }
                }
            }
        }
    }

    fn call_shape(
        &mut self,
        mut callee_ty: TypeId,
        groups: &[Vec<Operand>],
        result: TypeId,
        span: Span,
    ) {
        for group in groups {
            if callee_ty.index() >= self.model.types.len() {
                return;
            }
            let TypeKind::Function { parameters, result } = self.model.types.kind(callee_ty) else {
                self.diagnostics.push(
                    Diagnostic::error("MIR call applies arguments to a non-function value")
                        .with_code("E5111")
                        .at(span),
                );
                return;
            };
            if parameters.len() != group.len() {
                self.diagnostics.push(
                    Diagnostic::error("MIR call argument count does not match its function type")
                        .with_code("E5112")
                        .at(span),
                );
            }
            callee_ty = *result;
        }
        if callee_ty != result {
            self.diagnostics.push(
                Diagnostic::error("MIR call result does not match the resulting function type")
                    .with_code("E5113")
                    .at(span),
            );
        }
    }

    pub(super) fn operand(&mut self, function: &Function, operand: &Operand, span: Span) {
        self.ty(operand.ty, span);
        match &operand.kind {
            OperandKind::Copy(place) => {
                if self
                    .place(function, place, span)
                    .is_some_and(|ty| ty != operand.ty)
                {
                    self.diagnostics.push(
                        Diagnostic::error("MIR operand type differs from its local type")
                            .with_code("E5114")
                            .at(span),
                    );
                }
            }
            OperandKind::Definition(definition) => {
                if definition.index() >= self.model.resolutions.definitions.len() {
                    self.invalid_id("definition", definition.index(), span);
                } else if self
                    .model
                    .definition_types
                    .get(definition)
                    .is_some_and(|ty| *ty != operand.ty)
                {
                    self.diagnostics.push(
                        Diagnostic::error("MIR definition operand has the wrong semantic type")
                            .with_code("E5120")
                            .at(span),
                    );
                }
            }
            OperandKind::Constant(Constant::Literal(_)) => {}
            OperandKind::Constant(Constant::Unit) => {
                if !self.is_void(operand.ty) {
                    self.diagnostics.push(
                        Diagnostic::error("MIR unit constant does not have void type")
                            .with_code("E5115")
                            .at(span),
                    );
                }
            }
        }
    }

    pub(super) fn place(
        &mut self,
        function: &Function,
        place: &Place,
        span: Span,
    ) -> Option<TypeId> {
        let mut current = match place.base {
            PlaceBase::Local(local) => match function.locals.get(local.index()) {
                Some(local) => local.ty,
                None => {
                    self.invalid_id("MIR local", local.index(), span);
                    return None;
                }
            },
            PlaceBase::Definition(definition) => {
                let Some(record) = self.model.resolutions.definitions.get(definition.index())
                else {
                    self.invalid_id("definition", definition.index(), span);
                    return None;
                };
                if record.kind != DefinitionKind::Global {
                    self.diagnostics.push(
                        Diagnostic::error("MIR place definition is not a global value")
                            .with_code("E5130")
                            .at(span),
                    );
                }
                let Some(ty) = self.model.definition_types.get(&definition).copied() else {
                    self.diagnostics.push(
                        Diagnostic::error("MIR global place has no semantic type")
                            .with_code("E5131")
                            .at(span),
                    );
                    return None;
                };
                ty
            }
        };

        for projection in &place.projections {
            self.ty(projection.ty, span);
            match &projection.kind {
                ProjectionKind::Dereference => {
                    if !self.dereference_matches(current, projection.ty) {
                        self.diagnostics.push(
                            Diagnostic::error(
                                "MIR dereference projection has an incompatible type",
                            )
                            .with_code("E5132")
                            .at(span),
                        );
                    }
                }
                ProjectionKind::Field(field) => {
                    if !self.model.field_types.contains_key(field) {
                        self.invalid_id("field", field.index(), span);
                    }
                    match (
                        self.nominal_definition(current),
                        self.model.field_owners.get(field),
                    ) {
                        (Some(owner), Some(field_owner)) if owner == *field_owner => {}
                        (Some(_), Some(_)) => self.diagnostics.push(
                            Diagnostic::error("MIR field projection belongs to a different struct")
                                .with_code("E5135")
                                .at(span),
                        ),
                        (None, _) => self.diagnostics.push(
                            Diagnostic::error("MIR field projection has a non-struct receiver")
                                .with_code("E5135")
                                .at(span),
                        ),
                        (_, None) => {}
                    }
                }
                ProjectionKind::Index(coordinates) => {
                    for coordinate in coordinates {
                        self.operand(function, coordinate, span);
                        if !self.is_integral(coordinate.ty) {
                            self.diagnostics.push(
                                Diagnostic::error("MIR index coordinate is not an integer")
                                    .with_code("E5133")
                                    .at(span),
                            );
                        }
                    }
                    if coordinates.is_empty()
                        || !self.index_matches(current, coordinates.len(), projection.ty)
                    {
                        self.diagnostics.push(
                            Diagnostic::error("MIR index projection has incompatible types")
                                .with_code("E5134")
                                .at(span),
                        );
                    }
                }
                ProjectionKind::VariantField { variant, index } => {
                    let owner = self.nominal_definition(current);
                    match self.model.variant_owners.get(variant) {
                        Some(variant_owner) if Some(*variant_owner) == owner => {}
                        Some(_) => self.diagnostics.push(
                            Diagnostic::error(
                                "MIR variant payload projection belongs to a different enum",
                            )
                            .with_code("E5142")
                            .at(span),
                        ),
                        None => self.invalid_id("variant", variant.index(), span),
                    }
                    match self
                        .model
                        .variant_payloads
                        .get(variant)
                        .and_then(|payload| payload.get(*index as usize))
                    {
                        Some(expected)
                            if !self.contains_generic(*expected) && *expected != projection.ty =>
                        {
                            self.diagnostics.push(
                                Diagnostic::error(
                                    "MIR variant payload projection has the wrong type",
                                )
                                .with_code("E5143")
                                .at(span),
                            );
                        }
                        Some(_) => {}
                        None if self.model.variant_payloads.contains_key(variant) => {
                            self.diagnostics.push(
                                Diagnostic::error(
                                    "MIR variant payload projection index is out of range",
                                )
                                .with_code("E5144")
                                .at(span),
                            )
                        }
                        None => {}
                    }
                }
            }
            current = projection.ty;
        }
        Some(current)
    }
}
