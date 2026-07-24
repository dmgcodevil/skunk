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
define_id!(DefId);
define_id!(LocalId);
define_id!(TypeId);
define_id!(FieldId);
define_id!(VariantId);

/// Identity of a syntax node within a compilation.
///
/// The high 32 bits identify the source file and the low 32 bits identify the
/// node within that file. This keeps independently parsed modules collision
/// free without requiring a process-global allocator.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeId(u64);

impl NodeId {
    pub const fn in_file(file: FileId, raw: u32) -> Self {
        Self(((file.index() as u64) << 32) | raw as u64)
    }

    pub const fn file(self) -> FileId {
        FileId::new((self.0 >> 32) as u32)
    }

    pub const fn local_index(self) -> u32 {
        self.0 as u32
    }
}

/// Allocates identities for syntax nodes produced during one parse session.
#[derive(Debug)]
pub struct NodeIdAllocator {
    file: FileId,
    next: u32,
}

impl Default for NodeIdAllocator {
    fn default() -> Self {
        Self::for_file(FileId::new(0))
    }
}

impl NodeIdAllocator {
    pub const fn for_file(file: FileId) -> Self {
        Self { file, next: 0 }
    }

    pub fn allocate(&mut self) -> NodeId {
        let id = NodeId::in_file(self.file, self.next);
        self.next = self
            .next
            .checked_add(1)
            .expect("a source file cannot contain more than u32::MAX syntax nodes");
        id
    }
}
