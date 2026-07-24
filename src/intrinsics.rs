//! Canonical registry of compiler-provided nominal types.
//!
//! Frontend resolution and backend lowering must agree on these identities.
//! Keeping the names here avoids scattering string lists across phases.

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IntrinsicType {
    System,
    Memory,
    Bounds,
    Window,
    Color,
    Keyboard,
    Testing,
}

impl IntrinsicType {
    pub const ALL: [Self; 7] = [
        Self::System,
        Self::Memory,
        Self::Bounds,
        Self::Window,
        Self::Color,
        Self::Keyboard,
        Self::Testing,
    ];

    pub const fn name(self) -> &'static str {
        match self {
            Self::System => "System",
            Self::Memory => "Memory",
            Self::Bounds => "Bounds",
            Self::Window => "Window",
            Self::Color => "Color",
            Self::Keyboard => "Keyboard",
            Self::Testing => "Testing",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|intrinsic| intrinsic.name() == name)
    }
}
