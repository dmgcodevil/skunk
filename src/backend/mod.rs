//! LLVM backend and native build facade.
//!
//! This module defines LLVM value/layout models and assembles whole-module IR
//! from validated MIR. Instruction selection, representation coercions, and
//! low-level emission live in focused child modules.

use crate::analysis::model::SemanticModel;
use crate::analysis::resolver::DefinitionKind;
use crate::analysis::types::TypeKind as SemanticTypeKind;
use crate::ids::{DefId, TypeId};
use crate::intrinsics::IntrinsicType;
use crate::mir::{self, DeclarationKind};
use std::collections::HashMap;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

mod coercion;
mod emitter;
mod mir_codegen;

#[cfg(test)]
mod tests;

#[derive(Clone, Debug, PartialEq, Eq)]
enum LlvmType {
    I8,
    I16,
    I32,
    I64,
    F32,
    F64,
    Char16,
    I1,
    PtrI8,
    Allocator,
    Arena,
    Window,
    TraitObject(String),
    TraitIntersection(Vec<String>),
    Union(Vec<LlvmType>),
    Struct(String),
    Enum(String),
    Reference {
        target_type: Box<LlvmType>,
        mutable: bool,
    },
    Pointer {
        target_type: Box<LlvmType>,
    },
    Function {
        parameters: Vec<LlvmType>,
        return_type: Box<LlvmType>,
    },
    Slice {
        elem_type: Box<LlvmType>,
    },
    Array {
        elem_type: Box<LlvmType>,
        len: usize,
    },
    Void,
}

impl LlvmType {
    fn ir(&self) -> String {
        match self {
            LlvmType::I8 => "i8".to_string(),
            LlvmType::I16 => "i16".to_string(),
            LlvmType::I32 => "i32".to_string(),
            LlvmType::I64 => "i64".to_string(),
            LlvmType::F32 => "float".to_string(),
            LlvmType::F64 => "double".to_string(),
            LlvmType::Char16 => "i16".to_string(),
            LlvmType::I1 => "i1".to_string(),
            LlvmType::PtrI8 => "ptr".to_string(),
            LlvmType::Allocator => "ptr".to_string(),
            LlvmType::Arena => "ptr".to_string(),
            LlvmType::Window => "ptr".to_string(),
            LlvmType::TraitObject(name) => format!("%trait.{}", sanitize_name(name)),
            LlvmType::TraitIntersection(traits) => format!(
                "{{ {} }}",
                std::iter::repeat_n("ptr", traits.len() + 1)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            LlvmType::Union(_) => "{ i32, ptr }".to_string(),
            LlvmType::Struct(name) => format!("%struct.{}", sanitize_name(name)),
            LlvmType::Enum(name) => format!("%enum.{}", sanitize_name(name)),
            LlvmType::Reference { .. } => "ptr".to_string(),
            LlvmType::Pointer { .. } => "ptr".to_string(),
            LlvmType::Function { .. } => "{ ptr, ptr }".to_string(),
            LlvmType::Slice { .. } => "{ ptr, i32 }".to_string(),
            LlvmType::Array { elem_type, len } => format!("[{} x {}]", len, elem_type.ir()),
            LlvmType::Void => "void".to_string(),
        }
    }
}

#[derive(Clone, Debug)]
struct FunctionSignature {
    symbol_name: String,
    return_type: LlvmType,
    parameters: Vec<LlvmType>,
}

type BackendLayouts = (
    HashMap<String, StructLayout>,
    HashMap<String, EnumLayout>,
    HashMap<String, TraitLayout>,
);

#[derive(Clone, Debug)]
struct StructLayout {
    name: String,
    fields: Vec<(String, LlvmType)>,
}

#[derive(Clone, Debug)]
struct EnumVariantLayout {
    payload_types: Vec<LlvmType>,
}

#[derive(Clone, Debug)]
struct EnumLayout {
    name: String,
    variants: Vec<EnumVariantLayout>,
}

#[derive(Clone, Debug)]
struct TraitMethodLayout {
    name: String,
    return_type: LlvmType,
    parameters: Vec<LlvmType>,
}

#[derive(Clone, Debug)]
struct TraitLayout {
    name: String,
    methods: Vec<TraitMethodLayout>,
}

#[derive(Clone, Debug)]
struct LocalVar {
    ptr: String,
    llvm_type: LlvmType,
}

#[derive(Clone, Debug)]
struct ExprValue {
    llvm_type: LlvmType,
    value: String,
}

#[derive(Clone, Debug)]
struct GlobalString {
    name: String,
    bytes: Vec<u8>,
}

impl GlobalString {
    fn ir_decl(&self) -> String {
        format!(
            "@{} = private unnamed_addr constant [{} x i8] c\"{}\", align 1",
            self.name,
            self.bytes.len(),
            escape_llvm_bytes(&self.bytes)
        )
    }
}

fn escape_llvm_bytes(bytes: &[u8]) -> String {
    let mut out = String::new();
    for &byte in bytes {
        match byte {
            b' '..=b'~' if byte != b'\\' && byte != b'"' => out.push(byte as char),
            _ => {
                let _ = write!(out, "\\{:02X}", byte);
            }
        }
    }
    out
}

/// Builds native layouts from validated MIR declarations.
fn collect_typed_layouts(
    module: &mir::Module,
    model: &SemanticModel,
) -> Result<BackendLayouts, String> {
    let nominal_kinds = typed_nominal_kinds(module);

    let mut structs = HashMap::new();
    let mut enums = HashMap::new();
    for declaration in &module.declarations {
        match &declaration.kind {
            DeclarationKind::Struct {
                definition, fields, ..
            } => {
                let name = definition_name(model, *definition)?.to_string();
                let fields = fields
                    .iter()
                    .map(|field| {
                        Ok((
                            field.name.clone(),
                            llvm_type_id(field.ty, model, &nominal_kinds)?,
                        ))
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                structs.insert(name.clone(), StructLayout { name, fields });
            }
            DeclarationKind::Enum {
                definition,
                variants,
                ..
            } => {
                let name = definition_name(model, *definition)?.to_string();
                let variants = variants
                    .iter()
                    .map(|variant| {
                        let payload_types = variant
                            .payload
                            .iter()
                            .map(|ty| llvm_type_id(*ty, model, &nominal_kinds))
                            .collect::<Result<Vec<_>, String>>()?;
                        Ok(EnumVariantLayout { payload_types })
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                enums.insert(name.clone(), EnumLayout { name, variants });
            }
            _ => {}
        }
    }

    let trait_declarations = module
        .declarations
        .iter()
        .filter_map(|declaration| match &declaration.kind {
            DeclarationKind::Trait { definition, .. } => Some((*definition, declaration)),
            _ => None,
        })
        .collect::<HashMap<_, _>>();
    let mut traits = HashMap::new();
    for definition in trait_declarations.keys() {
        build_typed_trait_layout(
            *definition,
            &trait_declarations,
            model,
            &nominal_kinds,
            &mut traits,
            &mut Vec::new(),
        )?;
    }
    Ok((structs, enums, traits))
}

fn collect_typed_signatures(
    module: &mir::Module,
    model: &SemanticModel,
) -> Result<HashMap<String, FunctionSignature>, String> {
    let nominal_kinds = typed_nominal_kinds(module);
    let mut signatures = HashMap::new();
    for declaration in &module.declarations {
        let DeclarationKind::ExternFunction {
            definition,
            parameters,
            result,
        } = &declaration.kind
        else {
            continue;
        };
        let name = definition_name(model, *definition)?.to_string();
        signatures.insert(
            name.clone(),
            FunctionSignature {
                symbol_name: name,
                return_type: llvm_type_id(*result, model, &nominal_kinds)?,
                parameters: parameters
                    .iter()
                    .map(|ty| llvm_type_id(*ty, model, &nominal_kinds))
                    .collect::<Result<Vec<_>, String>>()?,
            },
        );
    }
    for function in &module.functions {
        let Some(definition) = function.definition() else {
            continue;
        };
        let name = definition_name(model, definition)?;
        let (key, symbol_name) = match function.owner() {
            Some(owner) => {
                let owner_name = definition_name(model, owner)?;
                (
                    format!("{owner_name}::{name}"),
                    format!(
                        "skunk_{}_{}",
                        sanitize_name(owner_name),
                        sanitize_name(name)
                    ),
                )
            }
            None => (name.to_string(), format!("skunk_{name}")),
        };
        signatures.insert(
            key,
            typed_function_signature(
                symbol_name,
                function,
                model,
                &nominal_kinds,
                mir_function_has_receiver(function, model),
            )?,
        );
    }
    Ok(signatures)
}

fn mir_function_has_receiver(function: &mir::Function, model: &SemanticModel) -> bool {
    function.owner().is_some()
        && function
            .parameters
            .first()
            .and_then(|parameter| function.locals.get(parameter.index()))
            .and_then(|local| local.source)
            .and_then(|source| model.resolutions.locals.get(source.index()))
            .is_some_and(|local| local.name == "self")
}

fn typed_function_signature(
    symbol_name: String,
    function: &mir::Function,
    model: &SemanticModel,
    nominal_kinds: &HashMap<DefId, DefinitionKind>,
    skip_receiver: bool,
) -> Result<FunctionSignature, String> {
    let parameters = function
        .parameters
        .iter()
        .skip(usize::from(skip_receiver))
        .map(|parameter| {
            let local = function
                .locals
                .get(parameter.index())
                .ok_or_else(|| format!("unknown MIR parameter local {}", parameter.index()))?;
            llvm_type_id(local.ty, model, nominal_kinds)
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(FunctionSignature {
        symbol_name,
        return_type: llvm_type_id(function.result, model, nominal_kinds)?,
        parameters,
    })
}

fn collect_typed_implementations(
    module: &mir::Module,
    model: &SemanticModel,
) -> Result<Vec<(String, String)>, String> {
    let mut implementations = Vec::new();
    for declaration in &module.declarations {
        let DeclarationKind::Implementation { traits, target } = &declaration.kind else {
            continue;
        };
        let target = nominal_type_name(*target, model)?.to_string();
        for trait_type in traits {
            implementations.push((
                nominal_type_name(*trait_type, model)?.to_string(),
                target.clone(),
            ));
        }
    }
    Ok(implementations)
}

fn nominal_type_name(ty: TypeId, model: &SemanticModel) -> Result<&str, String> {
    match model.types.kind(ty) {
        SemanticTypeKind::Nominal { definition, .. } => definition_name(model, *definition),
        SemanticTypeKind::Const(inner) => nominal_type_name(*inner, model),
        other => Err(format!(
            "runtime implementation requires a concrete nominal type, found {other:?}"
        )),
    }
}

fn add_trait_vtable(
    trait_name: &str,
    target_name: &str,
    traits: &HashMap<String, TraitLayout>,
    signatures: &HashMap<String, FunctionSignature>,
    trait_vtables: &mut HashMap<String, String>,
    globals: &mut Vec<String>,
) -> Result<(), String> {
    let trait_layout = traits
        .get(trait_name)
        .ok_or_else(|| format!("unknown trait `{trait_name}` in LLVM backend"))?;
    let vtable_symbol = format!(
        "skunk_vtable_{}_{}",
        sanitize_name(trait_name),
        sanitize_name(target_name)
    );
    trait_vtables.insert(
        format!("{trait_name}=>{target_name}"),
        vtable_symbol.clone(),
    );
    let entries = trait_layout
        .methods
        .iter()
        .map(|method| {
            let signature_key = format!("{target_name}::{}", method.name);
            let signature = signatures.get(&signature_key).ok_or_else(|| {
                format!(
                    "missing concrete method `{}` for trait `{trait_name}` implementation on `{target_name}`",
                    method.name
                )
            })?;
            Ok(format!("ptr @{}", signature.symbol_name))
        })
        .collect::<Result<Vec<_>, String>>()?;
    globals.push(format!(
        "@{} = private unnamed_addr constant %vtable.{} {{ {} }}",
        vtable_symbol,
        sanitize_name(trait_name),
        entries.join(", ")
    ));
    Ok(())
}

fn typed_nominal_kinds(module: &mir::Module) -> HashMap<DefId, DefinitionKind> {
    module
        .declarations
        .iter()
        .filter_map(|declaration| {
            let (definition, kind) = match declaration.kind {
                DeclarationKind::Struct { definition, .. } => (definition, DefinitionKind::Struct),
                DeclarationKind::Enum { definition, .. } => (definition, DefinitionKind::Enum),
                DeclarationKind::Trait { definition, .. } => (definition, DefinitionKind::Trait),
                DeclarationKind::Shape { definition, .. } => (definition, DefinitionKind::Shape),
                _ => return None,
            };
            Some((definition, kind))
        })
        .collect()
}

fn build_typed_trait_layout(
    definition: DefId,
    declarations: &HashMap<DefId, &mir::Declaration>,
    model: &SemanticModel,
    nominal_kinds: &HashMap<DefId, DefinitionKind>,
    layouts: &mut HashMap<String, TraitLayout>,
    visiting: &mut Vec<DefId>,
) -> Result<TraitLayout, String> {
    let name = definition_name(model, definition)?.to_string();
    if let Some(layout) = layouts.get(&name) {
        return Ok(layout.clone());
    }
    if let Some(index) = visiting
        .iter()
        .position(|candidate| *candidate == definition)
    {
        let mut cycle = visiting[index..]
            .iter()
            .map(|definition| definition_name(model, *definition).unwrap_or("<invalid>"))
            .collect::<Vec<_>>();
        cycle.push(&name);
        return Err(format!(
            "cyclic supertrait relationship detected: {}",
            cycle.join(" -> ")
        ));
    }
    let declaration = declarations
        .get(&definition)
        .ok_or_else(|| format!("missing MIR declaration for trait `{name}`"))?;
    let DeclarationKind::Trait {
        supertraits,
        methods: declared_methods,
        ..
    } = &declaration.kind
    else {
        return Err(format!(
            "MIR definition `{name}` is not a trait declaration"
        ));
    };
    visiting.push(definition);
    let mut methods = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for supertrait in supertraits {
        let inherited = build_typed_trait_layout(
            *supertrait,
            declarations,
            model,
            nominal_kinds,
            layouts,
            visiting,
        )?;
        for method in inherited.methods {
            if seen.insert(method.name.clone()) {
                methods.push(method);
            }
        }
    }
    for method in declared_methods {
        if !seen.insert(method.name.clone()) {
            return Err(format!(
                "trait `{name}` declares duplicate inherited method `{}`",
                method.name
            ));
        }
        methods.push(TraitMethodLayout {
            name: method.name.clone(),
            return_type: llvm_type_id(method.result, model, nominal_kinds)?,
            parameters: method
                .parameters
                .iter()
                .map(|ty| llvm_type_id(*ty, model, nominal_kinds))
                .collect::<Result<Vec<_>, String>>()?,
        });
    }
    visiting.pop();
    let layout = TraitLayout { name, methods };
    layouts.insert(layout.name.clone(), layout.clone());
    Ok(layout)
}

fn llvm_type_id(
    ty: TypeId,
    model: &SemanticModel,
    nominal_kinds: &HashMap<DefId, DefinitionKind>,
) -> Result<LlvmType, String> {
    match model.types.kind(ty) {
        SemanticTypeKind::Builtin(builtin) => match builtin {
            crate::syntax::ast::BuiltinType::Void => Ok(LlvmType::Void),
            crate::syntax::ast::BuiltinType::Byte => Ok(LlvmType::I8),
            crate::syntax::ast::BuiltinType::Short => Ok(LlvmType::I16),
            crate::syntax::ast::BuiltinType::Int => Ok(LlvmType::I32),
            crate::syntax::ast::BuiltinType::Long => Ok(LlvmType::I64),
            crate::syntax::ast::BuiltinType::Float => Ok(LlvmType::F32),
            crate::syntax::ast::BuiltinType::Double => Ok(LlvmType::F64),
            crate::syntax::ast::BuiltinType::String => Ok(LlvmType::PtrI8),
            crate::syntax::ast::BuiltinType::Boolean => Ok(LlvmType::I1),
            crate::syntax::ast::BuiltinType::Char => Ok(LlvmType::Char16),
            crate::syntax::ast::BuiltinType::Allocator => Ok(LlvmType::Allocator),
            crate::syntax::ast::BuiltinType::Arena => Ok(LlvmType::Arena),
        },
        SemanticTypeKind::Intrinsic(intrinsic) => match intrinsic {
            IntrinsicType::Color => Ok(LlvmType::I32),
            IntrinsicType::Window => Ok(LlvmType::Window),
            other => Err(format!(
                "intrinsic type `{}` has no runtime value layout",
                other.name()
            )),
        },
        SemanticTypeKind::Nominal { definition, .. } => {
            let name = definition_name(model, *definition)?.to_string();
            match nominal_kinds.get(definition).copied().or_else(|| {
                model
                    .resolutions
                    .definitions
                    .get(definition.index())
                    .map(|definition| definition.kind)
            }) {
                Some(DefinitionKind::Struct) => Ok(LlvmType::Struct(name)),
                Some(DefinitionKind::Enum) => Ok(LlvmType::Enum(name)),
                Some(DefinitionKind::Trait) => Ok(LlvmType::TraitObject(name)),
                Some(kind) => Err(format!(
                    "definition `{name}` of kind {kind:?} has no runtime nominal layout"
                )),
                None => Err(format!("unknown nominal definition `{name}`")),
            }
        }
        SemanticTypeKind::Const(inner) => llvm_type_id(*inner, model, nominal_kinds),
        SemanticTypeKind::Array {
            element,
            dimensions,
        } => {
            let mut element = llvm_type_id(*element, model, nominal_kinds)?;
            for dimension in dimensions.iter().rev() {
                element = LlvmType::Array {
                    elem_type: Box::new(element),
                    len: usize::try_from(*dimension)
                        .map_err(|_| format!("array dimension `{dimension}` is too large"))?,
                };
            }
            Ok(element)
        }
        SemanticTypeKind::Reference { target, mutable } => Ok(LlvmType::Reference {
            target_type: Box::new(llvm_type_id(*target, model, nominal_kinds)?),
            mutable: *mutable,
        }),
        SemanticTypeKind::Pointer(target) => Ok(LlvmType::Pointer {
            target_type: Box::new(llvm_type_id(*target, model, nominal_kinds)?),
        }),
        SemanticTypeKind::Slice(element) => Ok(LlvmType::Slice {
            elem_type: Box::new(llvm_type_id(*element, model, nominal_kinds)?),
        }),
        SemanticTypeKind::Union(members) => Ok(LlvmType::Union(
            members
                .iter()
                .map(|member| llvm_type_id(*member, model, nominal_kinds))
                .collect::<Result<Vec<_>, String>>()?,
        )),
        SemanticTypeKind::Intersection(members) => Ok(LlvmType::TraitIntersection(
            members
                .iter()
                .map(|member| match model.types.kind(*member) {
                    SemanticTypeKind::Nominal { definition, .. }
                        if nominal_kinds.get(definition) == Some(&DefinitionKind::Trait) =>
                    {
                        Ok(definition_name(model, *definition)?.to_string())
                    }
                    _ => Err("LLVM trait intersections require trait members".to_string()),
                })
                .collect::<Result<Vec<_>, String>>()?,
        )),
        SemanticTypeKind::Function { parameters, result } => Ok(LlvmType::Function {
            parameters: parameters
                .iter()
                .map(|parameter| llvm_type_id(*parameter, model, nominal_kinds))
                .collect::<Result<Vec<_>, String>>()?,
            return_type: Box::new(llvm_type_id(*result, model, nominal_kinds)?),
        }),
        SemanticTypeKind::Error => Err("error type reached LLVM layout lowering".to_string()),
        SemanticTypeKind::Never => Err("never type has no LLVM value layout yet".to_string()),
        SemanticTypeKind::GenericParameter(definition) => Err(format!(
            "unspecialized generic `{}` reached LLVM layout lowering",
            definition_name(model, *definition)?
        )),
    }
}

fn definition_name(model: &SemanticModel, definition: DefId) -> Result<&str, String> {
    model
        .resolutions
        .definitions
        .get(definition.index())
        .map(|definition| definition.name.as_str())
        .ok_or_else(|| format!("unknown definition id {}", definition.index()))
}

fn is_integer_llvm_type(llvm_type: &LlvmType) -> bool {
    matches!(
        llvm_type,
        LlvmType::I8 | LlvmType::I16 | LlvmType::I32 | LlvmType::I64 | LlvmType::Char16
    )
}

fn is_numeric_llvm_type(llvm_type: &LlvmType) -> bool {
    is_integer_llvm_type(llvm_type) || matches!(llvm_type, LlvmType::F32 | LlvmType::F64)
}

fn promoted_numeric_llvm_type(left: &LlvmType, right: &LlvmType) -> Option<LlvmType> {
    match (left, right) {
        (LlvmType::F64, _) | (_, LlvmType::F64) => Some(LlvmType::F64),
        (LlvmType::F32, _) | (_, LlvmType::F32) => Some(LlvmType::F32),
        (LlvmType::I64, _) | (_, LlvmType::I64) => Some(LlvmType::I64),
        (LlvmType::I8, LlvmType::I8)
        | (LlvmType::I8, LlvmType::I16)
        | (LlvmType::I16, LlvmType::I8)
        | (LlvmType::I16, LlvmType::I16)
        | (LlvmType::I8, LlvmType::I32)
        | (LlvmType::I32, LlvmType::I8)
        | (LlvmType::I16, LlvmType::I32)
        | (LlvmType::I32, LlvmType::I16)
        | (LlvmType::I32, LlvmType::I32) => Some(LlvmType::I32),
        _ => None,
    }
}

struct FunctionCompilerDependencies<'a> {
    signatures: &'a HashMap<String, FunctionSignature>,
    structs: &'a HashMap<String, StructLayout>,
    enums: &'a HashMap<String, EnumLayout>,
    traits: &'a HashMap<String, TraitLayout>,
    trait_vtables: &'a HashMap<String, String>,
    globals: &'a mut Vec<GlobalString>,
    extra_type_decls: &'a mut Vec<String>,
    extra_function_irs: &'a mut Vec<String>,
    lambda_counter: &'a mut usize,
}

struct FunctionCompiler<'a> {
    function_name: &'a str,
    return_type: LlvmType,
    signatures: &'a HashMap<String, FunctionSignature>,
    structs: &'a HashMap<String, StructLayout>,
    enums: &'a HashMap<String, EnumLayout>,
    traits: &'a HashMap<String, TraitLayout>,
    trait_vtables: &'a HashMap<String, String>,
    globals: &'a mut Vec<GlobalString>,
    extra_type_decls: &'a mut Vec<String>,
    extra_function_irs: &'a mut Vec<String>,
    lambda_counter: &'a mut usize,
    lines: Vec<String>,
    temp_counter: usize,
    label_counter: usize,
}

fn align_up(value: usize, align: usize) -> usize {
    if align <= 1 {
        value
    } else {
        let rem = value % align;
        if rem == 0 {
            value
        } else {
            value + (align - rem)
        }
    }
}

fn llvm_type_align(
    llvm_type: &LlvmType,
    structs: &HashMap<String, StructLayout>,
    enums: &HashMap<String, EnumLayout>,
) -> usize {
    match llvm_type {
        LlvmType::I8 | LlvmType::I1 => 1,
        LlvmType::I16 | LlvmType::Char16 => 2,
        LlvmType::I32 | LlvmType::F32 => 4,
        LlvmType::I64
        | LlvmType::F64
        | LlvmType::PtrI8
        | LlvmType::Allocator
        | LlvmType::Arena
        | LlvmType::Window
        | LlvmType::TraitObject(_)
        | LlvmType::TraitIntersection(_)
        | LlvmType::Union(_)
        | LlvmType::Reference { .. }
        | LlvmType::Pointer { .. }
        | LlvmType::Function { .. }
        | LlvmType::Slice { .. } => 8,
        LlvmType::Struct(name) => structs
            .get(name)
            .map(|layout| {
                layout
                    .fields
                    .iter()
                    .map(|(_, field)| llvm_type_align(field, structs, enums))
                    .max()
                    .unwrap_or(1)
            })
            .unwrap_or(8),
        LlvmType::Enum(name) => enums
            .get(name)
            .map(|layout| {
                layout
                    .variants
                    .iter()
                    .flat_map(|variant| &variant.payload_types)
                    .map(|payload| llvm_type_align(payload, structs, enums))
                    .max()
                    .unwrap_or(4)
                    .max(4)
            })
            .unwrap_or(8),
        LlvmType::Array { elem_type, .. } => llvm_type_align(elem_type, structs, enums),
        LlvmType::Void => 1,
    }
}

fn sanitize_name(name: &str) -> String {
    name.chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
        .collect()
}

pub struct CompiledArtifact {
    pub llvm_ir_path: PathBuf,
    pub binary_path: PathBuf,
}

/// Linker and optimization options for a native build, typically sourced from
/// a project's `skunk.toml`.
#[derive(Clone, Debug)]
pub struct BuildOptions {
    /// When true (the default), compile with `-O2`; otherwise `-O0`.
    pub optimize: bool,
    /// Extra libraries linked as `-l<name>`.
    pub libraries: Vec<String>,
    /// macOS frameworks linked as `-framework <name>`; ignored elsewhere.
    pub frameworks: Vec<String>,
}

impl Default for BuildOptions {
    fn default() -> Self {
        BuildOptions {
            optimize: true,
            libraries: Vec::new(),
            frameworks: Vec::new(),
        }
    }
}

// The C runtime sources are embedded into the compiler binary so an installed
// `skunk` works without a source checkout. They are materialized on demand
// into `$SKUNK_HOME/runtime/<version>/` (default `~/.skunk`).
const RUNTIME_C_SOURCE: &str = include_str!("../../runtime/skunk_runtime.c");
#[cfg(target_os = "macos")]
const RUNTIME_WINDOW_SOURCE: &str = include_str!("../../runtime/skunk_window_runtime.m");

static MATERIALIZED_FILE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Returns the Skunk home directory: `$SKUNK_HOME`, or `~/.skunk`, or a
/// temporary directory as a last resort.
pub fn skunk_home() -> PathBuf {
    if let Ok(home) = std::env::var("SKUNK_HOME") {
        if !home.is_empty() {
            return PathBuf::from(home);
        }
    }
    if let Ok(home) = std::env::var("HOME") {
        if !home.is_empty() {
            return PathBuf::from(home).join(".skunk");
        }
    }
    std::env::temp_dir().join("skunk-home")
}

/// Writes `contents` to `path` unless the file already has identical contents.
/// The write goes through a temporary file plus rename so concurrent readers
/// never observe a partially written file.
pub(crate) fn write_if_changed(path: &Path, contents: &str) -> Result<(), String> {
    if let Ok(existing) = fs::read_to_string(path) {
        if existing == contents {
            return Ok(());
        }
    }

    // Native compiler tests run in parallel and all materialize the same
    // embedded runtime. A process ID alone is therefore not enough to make the
    // staging path unique: one thread can rename the file while another still
    // expects it to exist. Keep the temporary file beside the destination so
    // the final rename is atomic, but give every write its own sequence number.
    let sequence = MATERIALIZED_FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let mut temp_name = path
        .file_name()
        .ok_or_else(|| format!("cannot materialize invalid path `{}`", path.display()))?
        .to_os_string();
    temp_name.push(format!(".tmp-{}-{}", std::process::id(), sequence));
    let temp_path = path.with_file_name(temp_name);
    fs::write(&temp_path, contents)
        .map_err(|err| format!("failed to write `{}`: {}", temp_path.display(), err))?;
    match fs::rename(&temp_path, path) {
        Ok(()) => Ok(()),
        Err(err) => {
            let _ = fs::remove_file(&temp_path);
            // On platforms where rename does not replace an existing file,
            // another writer may have completed the identical materialization.
            if fs::read_to_string(path)
                .map(|existing| existing == contents)
                .unwrap_or(false)
            {
                Ok(())
            } else {
                Err(format!(
                    "failed to move `{}` into place: {}",
                    path.display(),
                    err
                ))
            }
        }
    }
}

/// Materializes the embedded runtime C sources into the Skunk home directory,
/// keyed by compiler version, and returns their paths. The first path is the
/// core runtime; the second is the macOS window runtime when applicable.
fn materialize_runtime_sources() -> Result<(PathBuf, Option<PathBuf>), String> {
    let dir = skunk_home().join("runtime").join(env!("CARGO_PKG_VERSION"));
    fs::create_dir_all(&dir)
        .map_err(|err| format!("failed to create `{}`: {}", dir.display(), err))?;
    let runtime_c = dir.join("skunk_runtime.c");
    write_if_changed(&runtime_c, RUNTIME_C_SOURCE)?;
    #[cfg(target_os = "macos")]
    {
        let runtime_window = dir.join("skunk_window_runtime.m");
        write_if_changed(&runtime_window, RUNTIME_WINDOW_SOURCE)?;
        Ok((runtime_c, Some(runtime_window)))
    }
    #[cfg(not(target_os = "macos"))]
    Ok((runtime_c, None))
}

/// Compiles a checked Skunk program into LLVM IR and a native executable
/// using default build options.
pub fn compile_to_executable(
    program: &crate::pipeline::CheckedProgram,
    source_path: &Path,
    output_path: &Path,
) -> Result<CompiledArtifact, String> {
    compile_to_executable_with_options(program, source_path, output_path, &BuildOptions::default())
}

/// Compiles a checked Skunk program into LLVM IR and a native executable.
pub fn compile_to_executable_with_options(
    program: &crate::pipeline::CheckedProgram,
    source_path: &Path,
    output_path: &Path,
    options: &BuildOptions,
) -> Result<CompiledArtifact, String> {
    let mut logger = crate::pipeline::CompilerLogger::disabled();
    compile_to_executable_with_options_and_logger(
        program,
        source_path,
        output_path,
        options,
        &mut logger,
    )
}

/// Compiles a checked program while reporting MIR and LLVM IR to `logger`.
pub fn compile_to_executable_with_options_and_logger(
    program: &crate::pipeline::CheckedProgram,
    source_path: &Path,
    output_path: &Path,
    options: &BuildOptions,
    logger: &mut crate::pipeline::CompilerLogger,
) -> Result<CompiledArtifact, String> {
    let llvm_ir = compile_to_llvm_ir_with_logger(program, logger)?;
    let llvm_ir_path = output_path.with_extension("ll");
    fs::write(&llvm_ir_path, llvm_ir).map_err(|err| {
        format!(
            "failed to write LLVM IR to {}: {}",
            llvm_ir_path.display(),
            err
        )
    })?;

    let (runtime_c_path, runtime_window_path) = materialize_runtime_sources()?;
    let mut command = if cfg!(target_os = "macos") {
        let mut command = Command::new("xcrun");
        command.args(["--sdk", "macosx", "clang"]);
        command
    } else {
        Command::new("clang")
    };
    command.arg(&llvm_ir_path).arg(&runtime_c_path);
    if cfg!(target_os = "macos") {
        if let Some(runtime_window_path) = &runtime_window_path {
            command.arg(runtime_window_path);
        }
        command.arg("-framework").arg("Cocoa");
        for framework in &options.frameworks {
            command.arg("-framework").arg(framework);
        }
    } else {
        // libm is linked by default on macOS but not on Linux.
        command.arg("-lm");
    }
    for library in &options.libraries {
        command.arg(format!("-l{}", library));
    }
    let status = command
        .arg(if options.optimize { "-O2" } else { "-O0" })
        .arg("-o")
        .arg(output_path)
        .status()
        .map_err(|err| {
            format!(
                "failed to invoke clang while compiling {}: {}. \
                 Skunk needs clang installed (macOS: `xcode-select --install`, \
                 Linux: install the `clang` package)",
                source_path.display(),
                err
            )
        })?;

    if !status.success() {
        return Err(format!(
            "clang failed while compiling {} to {}",
            source_path.display(),
            output_path.display()
        ));
    }

    Ok(CompiledArtifact {
        llvm_ir_path,
        binary_path: output_path.to_path_buf(),
    })
}

/// Lowers a checked Skunk program into textual LLVM IR without invoking the
/// system linker.
pub fn compile_to_llvm_ir(program: &crate::pipeline::CheckedProgram) -> Result<String, String> {
    let mut logger = crate::pipeline::CompilerLogger::disabled();
    compile_to_llvm_ir_with_logger(program, &mut logger)
}

/// Lowers a checked Skunk program to LLVM IR while reporting MIR and LLVM IR.
pub fn compile_to_llvm_ir_with_logger(
    program: &crate::pipeline::CheckedProgram,
    logger: &mut crate::pipeline::CompilerLogger,
) -> Result<String, String> {
    let semantic_model = &program.semantics;
    let mir = crate::pipeline::lower_to_mir_with_logger(program, logger)?;
    let (structs, enums, traits) = collect_typed_layouts(&mir, &program.semantics)?;
    let signatures = collect_typed_signatures(&mir, &program.semantics)?;
    let typed_implementations = collect_typed_implementations(&mir, &program.semantics)?;
    let mir_codegen_context = mir_codegen::CodegenContext::new(&mir, semantic_model)?;
    let mut trait_vtables = HashMap::<String, String>::new();
    let mut trait_vtable_globals = Vec::<String>::new();
    let mut extern_declares = Vec::<String>::new();

    // Symbols that the backend already declares unconditionally; extern
    // declarations may not redeclare them.
    const RESERVED_RUNTIME_SYMBOLS: &[&str] = &[
        "printf",
        "malloc",
        "memcpy",
        "memset",
        "skunk_system_allocator",
        "skunk_arena_init",
        "skunk_arena_allocator",
        "skunk_arena_reset",
        "skunk_arena_deinit",
        "skunk_alloc_create",
        "skunk_alloc_buffer",
        "skunk_alloc_destroy",
        "skunk_alloc_free",
        "skunk_panic_index_out_of_bounds",
        "skunk_panic_slice_range_out_of_bounds",
        "skunk_window_create",
        "skunk_window_is_open",
        "skunk_window_poll",
        "skunk_window_clear",
        "skunk_window_draw_rect",
        "skunk_window_present",
        "skunk_window_delta_time",
        "skunk_window_close",
        "skunk_window_deinit",
        "skunk_keyboard_is_down",
    ];

    let mut emitted_externs = std::collections::HashSet::new();
    for declaration in &mir.declarations {
        let DeclarationKind::ExternFunction { definition, .. } = declaration.kind else {
            continue;
        };
        let name = definition_name(semantic_model, definition)?;
        if RESERVED_RUNTIME_SYMBOLS.contains(&name) {
            return Err(format!(
                "extern function `{name}` redeclares a reserved runtime symbol"
            ));
        }
        if !emitted_externs.insert(name) {
            continue;
        }
        let signature = signatures
            .get(name)
            .ok_or_else(|| format!("missing typed signature for extern function `{name}`"))?;
        extern_declares.push(format!(
            "declare {} @{}({})",
            signature.return_type.ir(),
            signature.symbol_name,
            signature
                .parameters
                .iter()
                .map(LlvmType::ir)
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }

    if !signatures.contains_key("main") {
        return Err("LLVM backend currently requires `function main(): ... {}`".to_string());
    }

    for (trait_name, target_name) in typed_implementations {
        add_trait_vtable(
            &trait_name,
            &target_name,
            &traits,
            &signatures,
            &mut trait_vtables,
            &mut trait_vtable_globals,
        )?;
    }

    let mut globals = Vec::<GlobalString>::new();
    let mut extra_type_decls = Vec::<String>::new();
    let mut function_irs = Vec::<String>::new();
    let mut extra_function_irs = Vec::<String>::new();
    let program_global_decls = mir
        .declarations
        .iter()
        .filter_map(|declaration| match declaration.kind {
            DeclarationKind::Global { definition, .. } => {
                mir_codegen_context.globals.get(&definition)
            }
            _ => None,
        })
        .map(|global| {
            format!(
                "@{} = internal global {} zeroinitializer, align {}",
                global.symbol_name,
                global.llvm_type.ir(),
                llvm_type_align(&global.llvm_type, &structs, &enums)
            )
        })
        .collect::<Vec<_>>();
    let mut global_initializers = Vec::<(String, mir_codegen::GlobalCodegenInfo)>::new();
    let mut lambda_counter = 0usize;

    for function in &mir.functions {
        let (symbol_name, llvm_return_type, param_defs, initialized_global) =
            match function.origin {
                mir::FunctionOrigin::Definition { .. } => {
                    let signature_key = mir_codegen_context
                        .function_key(function)?
                        .ok_or_else(|| "defined MIR function has no signature key".to_string())?;
                    let signature = signatures.get(&signature_key).ok_or_else(|| {
                        format!("missing typed signature for function `{signature_key}`")
                    })?;
                    let has_receiver = mir_function_has_receiver(function, semantic_model);
                    let param_defs =
                        if has_receiver {
                            std::iter::once("ptr %arg0".to_string())
                                .chain(
                                    signature.parameters.iter().enumerate().map(|(index, ty)| {
                                        format!("{} %arg{}", ty.ir(), index + 1)
                                    }),
                                )
                                .collect()
                        } else {
                            signature
                                .parameters
                                .iter()
                                .enumerate()
                                .map(|(index, ty)| format!("{} %arg{}", ty.ir(), index))
                                .collect()
                        };
                    (
                        signature.symbol_name.clone(),
                        signature.return_type.clone(),
                        param_defs,
                        None,
                    )
                }
                mir::FunctionOrigin::GlobalInitializer { global } => {
                    let global_info = mir_codegen_context
                        .globals
                        .get(&global)
                        .cloned()
                        .ok_or_else(|| {
                            format!("missing codegen layout for global {}", global.index())
                        })?;
                    (
                        format!("skunk_init_{}", global_info.symbol_name),
                        global_info.llvm_type.clone(),
                        Vec::new(),
                        Some(global_info),
                    )
                }
                mir::FunctionOrigin::Closure => continue,
            };
        if initialized_global.is_some() && llvm_return_type == LlvmType::Void {
            return Err(format!(
                "LLVM global/function `{symbol_name}` cannot use void as a stored result"
            ));
        };
        let dependencies = FunctionCompilerDependencies {
            signatures: &signatures,
            structs: &structs,
            enums: &enums,
            traits: &traits,
            trait_vtables: &trait_vtables,
            globals: &mut globals,
            extra_type_decls: &mut extra_type_decls,
            extra_function_irs: &mut extra_function_irs,
            lambda_counter: &mut lambda_counter,
        };
        let compiler = FunctionCompiler::new(&symbol_name, llvm_return_type.clone(), dependencies);
        let body_lines = compiler.compile_mir(function, &mir_codegen_context)?;
        let mut function_ir = String::new();
        let _ = writeln!(
            function_ir,
            "define {} @{}({}) {{",
            llvm_return_type.ir(),
            symbol_name,
            param_defs.join(", ")
        );
        let _ = writeln!(function_ir, "entry:");
        for line in body_lines {
            let _ = writeln!(function_ir, "{}", line);
        }
        let _ = writeln!(function_ir, "}}");
        function_irs.push(function_ir);
        if let Some(global) = initialized_global {
            global_initializers.push((symbol_name, global));
        }
    }

    let main_signature = signatures
        .get("main")
        .ok_or_else(|| "LLVM backend requires a `main` function".to_string())?;
    if !main_signature.parameters.is_empty() {
        return Err("LLVM backend requires `main` to take no parameters".to_string());
    }

    let skunk_main_body = match main_signature.return_type {
        LlvmType::I8 => {
            "  %result = call i8 @skunk_main()\n  %exit_code = sext i8 %result to i32\n  ret i32 %exit_code\n"
        }
        LlvmType::I16 => {
            "  %result = call i16 @skunk_main()\n  %exit_code = sext i16 %result to i32\n  ret i32 %exit_code\n"
        }
        LlvmType::I32 => "  %result = call i32 @skunk_main()\n  ret i32 %result\n",
        LlvmType::I64 => "  %result = call i64 @skunk_main()\n  %exit_code = trunc i64 %result to i32\n  ret i32 %exit_code\n",
        LlvmType::F32 => {
            "  %result = call float @skunk_main()\n  %exit_code = fptosi float %result to i32\n  ret i32 %exit_code\n"
        }
        LlvmType::F64 => {
            "  %result = call double @skunk_main()\n  %exit_code = fptosi double %result to i32\n  ret i32 %exit_code\n"
        }
        LlvmType::Char16 => {
            "  %result = call i16 @skunk_main()\n  %exit_code = zext i16 %result to i32\n  ret i32 %exit_code\n"
        }
        LlvmType::I1 => "  %result = call i1 @skunk_main()\n  %exit_code = zext i1 %result to i32\n  ret i32 %exit_code\n",
        LlvmType::Void => "  call void @skunk_main()\n  ret i32 0\n",
        LlvmType::PtrI8 => {
            return Err("LLVM backend does not support `main` returning string yet".to_string())
        }
        LlvmType::Allocator
        | LlvmType::Arena
        | LlvmType::Window
        | LlvmType::TraitObject(_)
        | LlvmType::TraitIntersection(_)
        | LlvmType::Union(_)
        | LlvmType::Reference { .. }
        | LlvmType::Pointer { .. } => {
            return Err("LLVM backend does not support `main` returning pointer-like values yet".to_string())
        }
        LlvmType::Function { .. } => {
            return Err("LLVM backend does not support `main` returning functions yet".to_string())
        }
        LlvmType::Slice { .. } => {
            return Err("LLVM backend does not support `main` returning slices yet".to_string())
        }
        LlvmType::Array { .. } => {
            return Err("LLVM backend does not support `main` returning arrays yet".to_string())
        }
        LlvmType::Struct(_) | LlvmType::Enum(_) => {
            return Err(
                "LLVM backend does not support `main` returning structs or enums yet".to_string()
            )
        }
    };
    let mut c_main_body = String::new();
    for (index, (initializer, global)) in global_initializers.iter().enumerate() {
        let value = format!("%global_init_{index}");
        let _ = writeln!(
            c_main_body,
            "  {value} = call {} @{}()",
            global.llvm_type.ir(),
            initializer
        );
        let _ = writeln!(
            c_main_body,
            "  store {} {value}, ptr @{}",
            global.llvm_type.ir(),
            global.symbol_name
        );
    }
    c_main_body.push_str(skunk_main_body);

    let mut ir = String::new();
    let _ = writeln!(ir, "declare i32 @printf(ptr, ...)");
    let _ = writeln!(ir, "declare ptr @malloc(i64)");
    let _ = writeln!(ir, "declare ptr @memcpy(ptr, ptr, i64)");
    let _ = writeln!(ir, "declare ptr @memset(ptr, i32, i64)");
    let _ = writeln!(ir, "declare ptr @skunk_system_allocator()");
    let _ = writeln!(ir, "declare ptr @skunk_arena_init(ptr)");
    let _ = writeln!(ir, "declare ptr @skunk_arena_allocator(ptr)");
    let _ = writeln!(ir, "declare void @skunk_arena_reset(ptr)");
    let _ = writeln!(ir, "declare void @skunk_arena_deinit(ptr)");
    let _ = writeln!(ir, "declare ptr @skunk_alloc_create(ptr, i64)");
    let _ = writeln!(ir, "declare ptr @skunk_alloc_buffer(ptr, i64, i32)");
    let _ = writeln!(ir, "declare void @skunk_alloc_destroy(ptr, ptr)");
    let _ = writeln!(ir, "declare void @skunk_alloc_free(ptr, ptr)");
    let _ = writeln!(
        ir,
        "declare void @skunk_panic_index_out_of_bounds(i64, i64)"
    );
    let _ = writeln!(
        ir,
        "declare void @skunk_panic_slice_range_out_of_bounds(i64, i64, i64)"
    );
    let _ = writeln!(ir, "declare ptr @skunk_window_create(i32, i32, ptr)");
    let _ = writeln!(ir, "declare i1 @skunk_window_is_open(ptr)");
    let _ = writeln!(ir, "declare void @skunk_window_poll(ptr)");
    let _ = writeln!(ir, "declare void @skunk_window_clear(ptr, i32)");
    let _ = writeln!(
        ir,
        "declare void @skunk_window_draw_rect(ptr, double, double, double, double, i32)"
    );
    let _ = writeln!(ir, "declare void @skunk_window_present(ptr)");
    let _ = writeln!(ir, "declare double @skunk_window_delta_time(ptr)");
    let _ = writeln!(ir, "declare void @skunk_window_close(ptr)");
    let _ = writeln!(ir, "declare void @skunk_window_deinit(ptr)");
    let _ = writeln!(ir, "declare i1 @skunk_keyboard_is_down(ptr, i16)");
    for declare in &extern_declares {
        let _ = writeln!(ir, "{}", declare);
    }
    let _ = writeln!(ir);
    for layout in traits.values() {
        let _ = writeln!(
            ir,
            "%trait.{} = type {{ ptr, ptr }}",
            sanitize_name(&layout.name)
        );
        let vtable_fields = if layout.methods.is_empty() {
            String::new()
        } else {
            std::iter::repeat_n("ptr", layout.methods.len())
                .collect::<Vec<_>>()
                .join(", ")
        };
        let _ = writeln!(
            ir,
            "%vtable.{} = type {{ {} }}",
            sanitize_name(&layout.name),
            vtable_fields
        );
    }
    if !traits.is_empty() {
        let _ = writeln!(ir);
    }
    for layout in structs.values() {
        let field_types = layout
            .fields
            .iter()
            .map(|(_, field_type)| field_type.ir())
            .collect::<Vec<_>>()
            .join(", ");
        let _ = writeln!(
            ir,
            "%struct.{} = type {{ {} }}",
            sanitize_name(&layout.name),
            field_types
        );
    }
    if !structs.is_empty() {
        let _ = writeln!(ir);
    }
    for layout in enums.values() {
        let mut field_types = vec!["i32".to_string()];
        for variant in &layout.variants {
            for payload_type in &variant.payload_types {
                field_types.push(payload_type.ir());
            }
        }
        let _ = writeln!(
            ir,
            "%enum.{} = type {{ {} }}",
            sanitize_name(&layout.name),
            field_types.join(", ")
        );
    }
    if !enums.is_empty() {
        let _ = writeln!(ir);
    }
    for global in &trait_vtable_globals {
        let _ = writeln!(ir, "{}", global);
    }
    if !trait_vtable_globals.is_empty() {
        let _ = writeln!(ir);
    }
    for global in &program_global_decls {
        let _ = writeln!(ir, "{}", global);
    }
    if !program_global_decls.is_empty() {
        let _ = writeln!(ir);
    }
    for decl in &extra_type_decls {
        let _ = writeln!(ir, "{}", decl);
    }
    if !extra_type_decls.is_empty() {
        let _ = writeln!(ir);
    }
    for global in &globals {
        let _ = writeln!(ir, "{}", global.ir_decl());
    }
    if !globals.is_empty() {
        let _ = writeln!(ir);
    }
    for function_ir in function_irs {
        let _ = writeln!(ir, "{}", function_ir);
    }
    for function_ir in extra_function_irs {
        let _ = writeln!(ir, "{}", function_ir);
    }
    let _ = writeln!(ir, "define i32 @main() {{");
    let _ = write!(ir, "{}", c_main_body);
    let _ = writeln!(ir, "}}");

    logger.text("LLVM lowering", "LLVM IR", &ir);
    Ok(ir)
}
