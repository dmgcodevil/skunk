//! Interned semantic types.
//!
//! A `TypeId` identifies a complete type. Nominal declarations retain their
//! separate `DefId`, so `Box[int]` and `Box[string]` share a definition while
//! remaining distinct semantic types.

use crate::ids::{DefId, TypeId};
use crate::intrinsics::IntrinsicType;
use crate::syntax::ast::BuiltinType;
use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum TypeKind {
    Error,
    Never,
    Builtin(BuiltinType),
    Intrinsic(IntrinsicType),
    GenericParameter(DefId),
    Nominal {
        definition: DefId,
        arguments: Vec<TypeId>,
    },
    Const(TypeId),
    Array {
        element: TypeId,
        dimensions: Vec<u64>,
    },
    Reference {
        target: TypeId,
        mutable: bool,
    },
    Pointer(TypeId),
    Slice(TypeId),
    Union(Vec<TypeId>),
    Intersection(Vec<TypeId>),
    Function {
        parameters: Vec<TypeId>,
        result: TypeId,
    },
}

#[derive(Debug)]
pub struct TypeStore {
    kinds: Vec<TypeKind>,
    interned: HashMap<TypeKind, TypeId>,
}

impl Default for TypeStore {
    fn default() -> Self {
        let mut store = Self {
            kinds: Vec::new(),
            interned: HashMap::new(),
        };
        store.intern(TypeKind::Error);
        store.intern(TypeKind::Never);
        for builtin in [
            BuiltinType::Void,
            BuiltinType::Byte,
            BuiltinType::Short,
            BuiltinType::Int,
            BuiltinType::Long,
            BuiltinType::Float,
            BuiltinType::Double,
            BuiltinType::String,
            BuiltinType::Boolean,
            BuiltinType::Char,
            BuiltinType::Allocator,
            BuiltinType::Arena,
        ] {
            store.intern(TypeKind::Builtin(builtin));
        }
        for intrinsic in IntrinsicType::ALL {
            store.intern(TypeKind::Intrinsic(intrinsic));
        }
        store
    }
}

impl TypeStore {
    pub fn intern(&mut self, kind: TypeKind) -> TypeId {
        if let Some(existing) = self.interned.get(&kind) {
            return *existing;
        }
        let id = TypeId::new(self.kinds.len() as u32);
        self.kinds.push(kind.clone());
        self.interned.insert(kind, id);
        id
    }

    pub fn kind(&self, id: TypeId) -> &TypeKind {
        &self.kinds[id.index()]
    }

    pub fn error(&self) -> TypeId {
        TypeId::new(0)
    }

    pub fn builtin(&mut self, builtin: BuiltinType) -> TypeId {
        self.intern(TypeKind::Builtin(builtin))
    }

    pub fn intrinsic(&mut self, intrinsic: IntrinsicType) -> TypeId {
        self.intern(TypeKind::Intrinsic(intrinsic))
    }

    pub fn function(&mut self, parameters: Vec<TypeId>, result: TypeId) -> TypeId {
        self.intern(TypeKind::Function { parameters, result })
    }

    pub fn nominal(&mut self, definition: DefId, arguments: Vec<TypeId>) -> TypeId {
        self.intern(TypeKind::Nominal {
            definition,
            arguments,
        })
    }

    pub fn len(&self) -> usize {
        self.kinds.len()
    }

    pub fn is_empty(&self) -> bool {
        self.kinds.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interns_equal_types_once_and_separates_instantiations() {
        let mut types = TypeStore::default();
        let int = types.builtin(BuiltinType::Int);
        let string = types.builtin(BuiltinType::String);
        let definition = DefId::new(7);

        let box_int_a = types.nominal(definition, vec![int]);
        let box_int_b = types.nominal(definition, vec![int]);
        let box_string = types.nominal(definition, vec![string]);

        assert_eq!(box_int_a, box_int_b);
        assert_ne!(box_int_a, box_string);
    }
}
