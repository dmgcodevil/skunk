//! Strongly typed identities shared by compiler phases.
//!
//! Each ID lives in a distinct namespace. Keeping these types separate prevents
//! source nodes, declarations, locals, and semantic types from being confused
//! simply because they are all represented by small integers internally.

macro_rules! define_id {
    ($name:ident) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(u32);

        impl $name {
            pub const fn new(raw: u32) -> Self {
                Self(raw)
            }

            pub const fn index(self) -> usize {
                self.0 as usize
            }
        }
    };
}

define_id!(FileId);
define_id!(NodeId);
define_id!(DefId);
define_id!(LocalId);
define_id!(TypeId);
define_id!(FieldId);
define_id!(VariantId);

/// Allocates identities for syntax nodes produced during one parse session.
#[derive(Debug, Default)]
pub struct NodeIdAllocator {
    next: u32,
}

impl NodeIdAllocator {
    pub fn allocate(&mut self) -> NodeId {
        let id = NodeId::new(self.next);
        self.next = self
            .next
            .checked_add(1)
            .expect("a source file cannot contain more than u32::MAX syntax nodes");
        id
    }
}
