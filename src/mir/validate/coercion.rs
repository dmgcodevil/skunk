use super::*;

impl Validator<'_> {
    pub(super) fn can_coerce(&self, actual: TypeId, expected: TypeId) -> bool {
        self.can_coerce_inner(actual, expected, &mut HashSet::new())
    }

    fn can_coerce_inner(
        &self,
        actual: TypeId,
        expected: TypeId,
        visiting: &mut HashSet<(TypeId, TypeId)>,
    ) -> bool {
        if actual == expected {
            return true;
        }
        if !visiting.insert((actual, expected)) {
            return false;
        }
        let result = match (self.valid_type_kind(actual), self.valid_type_kind(expected)) {
            (Some(TypeKind::Const(actual)), _) => {
                self.can_coerce_inner(*actual, expected, visiting)
            }
            (_, Some(TypeKind::Const(expected))) => {
                self.can_coerce_inner(actual, *expected, visiting)
            }
            (Some(TypeKind::Builtin(actual)), Some(TypeKind::Builtin(expected))) => {
                numeric_rank(*actual)
                    .zip(numeric_rank(*expected))
                    .is_some_and(|(actual, expected)| actual <= expected)
            }
            (
                Some(TypeKind::Reference {
                    target: actual_target,
                    mutable: actual_mutable,
                }),
                Some(TypeKind::Reference {
                    target: expected_target,
                    mutable: expected_mutable,
                }),
            ) => {
                (!expected_mutable || actual_mutable == expected_mutable)
                    && self.can_coerce_inner(*actual_target, *expected_target, visiting)
            }
            (Some(TypeKind::Pointer(actual)), Some(TypeKind::Pointer(expected)))
            | (Some(TypeKind::Slice(actual)), Some(TypeKind::Slice(expected))) => {
                self.can_coerce_inner(*actual, *expected, visiting)
            }
            (
                Some(TypeKind::Array {
                    element: actual_element,
                    dimensions: actual_dimensions,
                }),
                Some(TypeKind::Array {
                    element: expected_element,
                    dimensions: expected_dimensions,
                }),
            ) => {
                actual_dimensions == expected_dimensions
                    && self.can_coerce_inner(*actual_element, *expected_element, visiting)
            }
            (_, Some(TypeKind::Union(members))) => members
                .iter()
                .any(|member| self.can_coerce_inner(actual, *member, visiting)),
            (Some(TypeKind::Union(members)), _) => members
                .iter()
                .all(|member| self.can_coerce_inner(*member, expected, visiting)),
            (_, Some(TypeKind::Intersection(members))) => members
                .iter()
                .all(|member| self.can_coerce_inner(actual, *member, visiting)),
            (Some(TypeKind::Intersection(members)), _) => members
                .iter()
                .any(|member| self.can_coerce_inner(*member, expected, visiting)),
            (
                Some(TypeKind::Nominal {
                    definition: actual, ..
                }),
                Some(TypeKind::Nominal {
                    definition: expected,
                    ..
                }),
            ) if self.definition_kind(*expected) == Some(DefinitionKind::Trait) => {
                self.implements_or_extends(*actual, *expected, &mut HashSet::new())
            }
            _ => false,
        };
        visiting.remove(&(actual, expected));
        result
    }

    fn implements_or_extends(
        &self,
        actual: crate::ids::DefId,
        expected: crate::ids::DefId,
        visiting: &mut HashSet<crate::ids::DefId>,
    ) -> bool {
        if actual == expected || self.implementations.contains(&(expected, actual)) {
            return true;
        }
        if !visiting.insert(actual) {
            return false;
        }
        self.supertraits.get(&actual).is_some_and(|supertraits| {
            supertraits
                .iter()
                .any(|parent| self.implements_or_extends(*parent, expected, visiting))
        })
    }

    fn definition_kind(&self, definition: crate::ids::DefId) -> Option<DefinitionKind> {
        self.model
            .resolutions
            .definitions
            .get(definition.index())
            .map(|definition| definition.kind)
    }
}

fn numeric_rank(builtin: BuiltinType) -> Option<u8> {
    match builtin {
        BuiltinType::Byte => Some(0),
        BuiltinType::Short => Some(1),
        BuiltinType::Int => Some(2),
        BuiltinType::Long => Some(3),
        BuiltinType::Float => Some(4),
        BuiltinType::Double => Some(5),
        _ => None,
    }
}
