//! LLVM backend and native build facade.
//!
//! This module defines the shared LLVM value/layout model, collects program
//! layouts, and assembles whole-module IR. Function-body lowering is separated
//! into core lowering, access/call lowering, and coercion/runtime operations.

use crate::ast::{self, Literal, Node, Operator, Type, UnaryOperator};
use crate::ids::{DefId, TypeId};
use crate::intrinsics::IntrinsicType;
use crate::resolver::DefinitionKind;
use crate::semantic_types::TypeKind as SemanticTypeKind;
use crate::semantics::SemanticModel;
use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

mod access;
mod coercion;
mod lowering;

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

fn is_pointer_like_llvm_type(llvm_type: &LlvmType) -> bool {
    matches!(
        llvm_type,
        LlvmType::Pointer { .. } | LlvmType::Reference { .. }
    )
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
struct FunctionPlan {
    signature_key: String,
    symbol_name: String,
    parameters: Vec<(String, Type)>,
    body: Vec<Node>,
    is_method: bool,
}

#[derive(Clone, Debug)]
struct StructLayout {
    name: String,
    fields: Vec<(String, LlvmType)>,
}

#[derive(Clone, Debug)]
struct EnumVariantLayout {
    name: String,
    tag: usize,
    payload_types: Vec<LlvmType>,
    field_indices: Vec<usize>,
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

#[derive(Clone)]
struct DeferredExpression {
    expression: Node,
    locals: HashMap<String, LocalVar>,
    unsafe_depth: usize,
}

#[derive(Clone, Debug)]
struct TraitLayout {
    name: String,
    methods: Vec<TraitMethodLayout>,
}

#[derive(Clone, Debug)]
struct ClosureEnv {
    type_name: String,
    captures: Vec<(String, LlvmType)>,
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

/// Maps a checked Skunk type to the value representation used by the LLVM
/// backend, consulting collected nominal and trait layouts when necessary.
fn llvm_type(
    sk_type: &Type,
    structs: &HashMap<String, StructLayout>,
    enums: &HashMap<String, EnumLayout>,
    traits: &HashMap<String, TraitLayout>,
) -> Result<LlvmType, String> {
    match sk_type {
        Type::Const { inner } | Type::BindingConst { inner } => {
            llvm_type(inner, structs, enums, traits)
        }
        Type::Byte => Ok(LlvmType::I8),
        Type::Short => Ok(LlvmType::I16),
        Type::Int => Ok(LlvmType::I32),
        Type::Long => Ok(LlvmType::I64),
        Type::Float => Ok(LlvmType::F32),
        Type::Double => Ok(LlvmType::F64),
        Type::Boolean => Ok(LlvmType::I1),
        Type::String => Ok(LlvmType::PtrI8),
        Type::Char => Ok(LlvmType::Char16),
        Type::Allocator => Ok(LlvmType::Allocator),
        Type::Arena => Ok(LlvmType::Arena),
        Type::Array {
            elem_type,
            dimensions,
        } => {
            let mut llvm_elem = llvm_type(elem_type, structs, enums, traits)?;
            for dimension in dimensions.iter().rev() {
                llvm_elem = LlvmType::Array {
                    elem_type: Box::new(llvm_elem),
                    len: array_len_from_dimension(dimension)?,
                };
            }
            Ok(llvm_elem)
        }
        Type::Slice { elem_type } => Ok(LlvmType::Slice {
            elem_type: Box::new(llvm_type(elem_type, structs, enums, traits)?),
        }),
        Type::Reference {
            target_type,
            mutable,
        } => Ok(LlvmType::Reference {
            target_type: Box::new(llvm_type(target_type, structs, enums, traits)?),
            mutable: *mutable,
        }),
        Type::Pointer { target_type } => Ok(LlvmType::Pointer {
            target_type: Box::new(llvm_type(target_type, structs, enums, traits)?),
        }),
        Type::Function {
            parameters,
            return_type,
        } => Ok(LlvmType::Function {
            parameters: parameters
                .iter()
                .map(|param| llvm_type(param, structs, enums, traits))
                .collect::<Result<Vec<_>, _>>()?,
            return_type: Box::new(llvm_type(return_type, structs, enums, traits)?),
        }),
        Type::Union(members) => Ok(LlvmType::Union(
            members
                .iter()
                .map(|member| llvm_type(member, structs, enums, traits))
                .collect::<Result<Vec<_>, _>>()?,
        )),
        Type::Intersection(members) => {
            let names = members
                .iter()
                .map(|member| match member {
                    Type::Custom(name) if traits.contains_key(name) => Ok(name.clone()),
                    other => Err(format!(
                        "LLVM backend requires trait intersection members, found `{}`",
                        ast::type_to_string(other)
                    )),
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(LlvmType::TraitIntersection(names))
        }
        Type::Custom(name) => {
            if name == "Color" {
                Ok(LlvmType::I32)
            } else if name == "Window" {
                Ok(LlvmType::Window)
            } else if structs.contains_key(name) {
                Ok(LlvmType::Struct(name.clone()))
            } else if enums.contains_key(name) {
                Ok(LlvmType::Enum(name.clone()))
            } else if traits.contains_key(name) {
                Ok(LlvmType::TraitObject(name.clone()))
            } else {
                Err(format!("unknown nominal type `{}` in LLVM backend", name))
            }
        }
        Type::Void => Ok(LlvmType::Void),
        other => Err(format!(
            "LLVM backend does not support type `{}` yet",
            ast::type_to_string(other)
        )),
    }
}

fn array_len_from_dimension(dimension: &Node) -> Result<usize, String> {
    let value = match dimension {
        Node::Literal(Literal::Integer(value)) => *value,
        Node::Literal(Literal::Long(value)) => *value,
        other => {
            return Err(format!(
                "LLVM backend requires array dimensions to be integer literals, found `{:?}`",
                other
            ))
        }
    };
    usize::try_from(value).map_err(|_| {
        format!(
            "LLVM backend requires non-negative array dimensions, found `{}`",
            value
        )
    })
}

/// Collects the field order and concrete LLVM field types for every struct.
fn collect_struct_layouts(statements: &[Node]) -> Result<HashMap<String, StructLayout>, String> {
    let mut raw_fields = HashMap::<String, Vec<(String, Type)>>::new();
    let enum_placeholders = collect_enum_placeholders(statements);
    for statement in statements {
        if let Node::StructDeclaration { name, fields, .. } = statement {
            raw_fields.insert(name.clone(), fields.clone());
        }
    }

    let mut layouts = HashMap::<String, StructLayout>::new();
    for (name, fields) in &raw_fields {
        let llvm_fields = fields
            .iter()
            .map(|(field_name, field_type)| {
                Ok((
                    field_name.clone(),
                    llvm_type(
                        field_type,
                        &layouts_with_raw(raw_fields.keys(), &layouts),
                        &enum_placeholders,
                        &HashMap::new(),
                    )?,
                ))
            })
            .collect::<Result<Vec<_>, String>>()?;
        layouts.insert(
            name.clone(),
            StructLayout {
                name: name.clone(),
                fields: llvm_fields,
            },
        );
    }
    Ok(layouts)
}

/// Assigns enum tags and flattened payload slots used by construction and
/// pattern matching.
fn collect_enum_layouts(
    statements: &[Node],
    structs: &HashMap<String, StructLayout>,
) -> Result<HashMap<String, EnumLayout>, String> {
    let mut layouts = HashMap::<String, EnumLayout>::new();
    let enum_placeholders = collect_enum_placeholders(statements);

    for statement in statements {
        let Node::EnumDeclaration { name, variants, .. } = statement else {
            continue;
        };

        let mut variant_layouts = Vec::new();
        let mut next_field_index = 1usize;
        for (tag, variant) in variants.iter().enumerate() {
            let payload_types = variant
                .payload_types
                .iter()
                .map(|payload_type| {
                    llvm_type(payload_type, structs, &enum_placeholders, &HashMap::new())
                })
                .collect::<Result<Vec<_>, String>>()?;
            let field_indices = (0..payload_types.len())
                .map(|_| {
                    let current = next_field_index;
                    next_field_index += 1;
                    current
                })
                .collect::<Vec<_>>();
            variant_layouts.push(EnumVariantLayout {
                name: variant.name.clone(),
                tag,
                payload_types,
                field_indices,
            });
        }

        layouts.insert(
            name.clone(),
            EnumLayout {
                name: name.clone(),
                variants: variant_layouts,
            },
        );
    }

    Ok(layouts)
}

fn collect_enum_placeholders(statements: &[Node]) -> HashMap<String, EnumLayout> {
    statements
        .iter()
        .filter_map(|statement| match statement {
            Node::EnumDeclaration { name, .. } => Some((
                name.clone(),
                EnumLayout {
                    name: name.clone(),
                    variants: Vec::new(),
                },
            )),
            _ => None,
        })
        .collect()
}

/// Builds inherited trait method tables in their final vtable slot order.
fn collect_trait_layouts(
    statements: &[Node],
    structs: &HashMap<String, StructLayout>,
    enums: &HashMap<String, EnumLayout>,
) -> Result<HashMap<String, TraitLayout>, String> {
    let trait_decls = statements
        .iter()
        .filter_map(|statement| match statement {
            Node::TraitDeclaration {
                name,
                supertraits,
                methods,
                ..
            } => Some((name.clone(), (supertraits.clone(), methods.clone()))),
            _ => None,
        })
        .collect::<HashMap<_, _>>();
    fn build_trait_layout(
        trait_name: &str,
        trait_decls: &HashMap<String, (Vec<String>, Vec<ast::TraitMethodSignature>)>,
        structs: &HashMap<String, StructLayout>,
        enums: &HashMap<String, EnumLayout>,
        layouts: &mut HashMap<String, TraitLayout>,
        visiting: &mut Vec<String>,
    ) -> Result<TraitLayout, String> {
        if let Some(layout) = layouts.get(trait_name) {
            return Ok(layout.clone());
        }
        if visiting.iter().any(|name| name == trait_name) {
            visiting.push(trait_name.to_string());
            return Err(format!(
                "cyclic supertrait relationship detected: {}",
                visiting.join(" -> ")
            ));
        }
        let (supertraits, methods) = trait_decls
            .get(trait_name)
            .cloned()
            .ok_or_else(|| format!("unknown trait `{}` in LLVM backend", trait_name))?;
        visiting.push(trait_name.to_string());
        let mut method_layouts = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for supertrait in supertraits {
            let layout =
                build_trait_layout(&supertrait, trait_decls, structs, enums, layouts, visiting)?;
            for method in layout.methods {
                if seen.insert(method.name.clone()) {
                    method_layouts.push(method);
                }
            }
        }
        for method in methods {
            method
                .parameters
                .first()
                .ok_or_else(|| format!("trait method `{}` is missing self", method.name))?;
            if !seen.insert(method.name.clone()) {
                return Err(format!(
                    "trait `{}` declares duplicate inherited method `{}`",
                    trait_name, method.name
                ));
            }
            method_layouts.push(TraitMethodLayout {
                name: method.name.clone(),
                return_type: llvm_type(&method.return_type, structs, enums, layouts)?,
                parameters: method
                    .parameters
                    .iter()
                    .skip(1)
                    .map(|(_, ty)| llvm_type(ty, structs, enums, layouts))
                    .collect::<Result<Vec<_>, String>>()?,
            });
        }
        visiting.pop();
        let layout = TraitLayout {
            name: trait_name.to_string(),
            methods: method_layouts,
        };
        layouts.insert(trait_name.to_string(), layout.clone());
        Ok(layout)
    }

    let mut layouts = HashMap::new();
    for trait_name in trait_decls.keys() {
        build_trait_layout(
            trait_name,
            &trait_decls,
            structs,
            enums,
            &mut layouts,
            &mut Vec::new(),
        )?;
    }
    Ok(layouts)
}

/// Builds native layouts from validated HIR. This is the production layout
/// path; the legacy collectors above remain for focused backend unit tests.
fn collect_typed_layouts(
    module: &crate::hir::Module,
    model: &SemanticModel,
) -> Result<BackendLayouts, String> {
    let nominal_kinds = typed_nominal_kinds(module);

    let mut structs = HashMap::new();
    let mut enums = HashMap::new();
    for item in &module.items {
        let Some(definition) = item.definition else {
            continue;
        };
        let name = definition_name(model, definition)?.to_string();
        match &item.kind {
            crate::hir::ItemKind::Struct(declaration) => {
                let fields = declaration
                    .fields
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
            crate::hir::ItemKind::Enum(declaration) => {
                let mut next_field_index = 1usize;
                let variants = declaration
                    .variants
                    .iter()
                    .enumerate()
                    .map(|(tag, variant)| {
                        let payload_types = variant
                            .payload
                            .iter()
                            .map(|ty| llvm_type_id(*ty, model, &nominal_kinds))
                            .collect::<Result<Vec<_>, String>>()?;
                        let field_indices = (0..payload_types.len())
                            .map(|_| {
                                let index = next_field_index;
                                next_field_index += 1;
                                index
                            })
                            .collect();
                        Ok(EnumVariantLayout {
                            name: variant.name.clone(),
                            tag,
                            payload_types,
                            field_indices,
                        })
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                enums.insert(name.clone(), EnumLayout { name, variants });
            }
            _ => {}
        }
    }

    let trait_declarations = module
        .items
        .iter()
        .filter_map(|item| match (&item.kind, item.definition) {
            (crate::hir::ItemKind::Trait(declaration), Some(definition)) => {
                Some((definition, declaration))
            }
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
    module: &crate::hir::Module,
    model: &SemanticModel,
) -> Result<HashMap<String, FunctionSignature>, String> {
    let nominal_kinds = typed_nominal_kinds(module);
    let mut signatures = HashMap::new();
    for item in &module.items {
        match (&item.kind, item.definition) {
            (crate::hir::ItemKind::Function(function), Some(definition)) => {
                let name = definition_name(model, definition)?.to_string();
                signatures.insert(
                    name.clone(),
                    typed_function_signature(
                        format!("skunk_{name}"),
                        function,
                        model,
                        &nominal_kinds,
                        false,
                    )?,
                );
            }
            (crate::hir::ItemKind::ExternFunction(signature), Some(definition)) => {
                let name = definition_name(model, definition)?.to_string();
                signatures.insert(
                    name.clone(),
                    FunctionSignature {
                        symbol_name: name,
                        return_type: llvm_type_id(signature.result, model, &nominal_kinds)?,
                        parameters: signature
                            .parameters
                            .iter()
                            .map(|ty| llvm_type_id(*ty, model, &nominal_kinds))
                            .collect::<Result<Vec<_>, String>>()?,
                    },
                );
            }
            (crate::hir::ItemKind::Struct(declaration), Some(owner)) => {
                collect_typed_method_signatures(
                    owner,
                    &declaration.methods,
                    model,
                    &nominal_kinds,
                    &mut signatures,
                )?;
            }
            (crate::hir::ItemKind::Enum(declaration), Some(owner)) => {
                collect_typed_method_signatures(
                    owner,
                    &declaration.methods,
                    model,
                    &nominal_kinds,
                    &mut signatures,
                )?;
            }
            _ => {}
        }
    }
    Ok(signatures)
}

fn collect_typed_implementations(
    module: &crate::hir::Module,
    model: &SemanticModel,
) -> Result<Vec<(String, String)>, String> {
    let mut implementations = Vec::new();
    for item in &module.items {
        let crate::hir::ItemKind::Implementation { traits, target } = &item.kind else {
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

fn collect_typed_method_signatures(
    owner: DefId,
    methods: &[crate::hir::Function],
    model: &SemanticModel,
    nominal_kinds: &HashMap<DefId, DefinitionKind>,
    signatures: &mut HashMap<String, FunctionSignature>,
) -> Result<(), String> {
    let owner_name = definition_name(model, owner)?;
    for method in methods {
        let definition = method
            .definition
            .ok_or_else(|| format!("method on `{owner_name}` has no definition id"))?;
        let method_name = definition_name(model, definition)?;
        let key = format!("{owner_name}::{method_name}");
        let symbol = format!(
            "skunk_{}_{}",
            sanitize_name(owner_name),
            sanitize_name(method_name)
        );
        let has_receiver = method
            .parameters
            .first()
            .and_then(|parameter| model.resolutions.locals.get(parameter.local.index()))
            .is_some_and(|local| local.name == "self");
        signatures.insert(
            key,
            typed_function_signature(symbol, method, model, nominal_kinds, has_receiver)?,
        );
    }
    Ok(())
}

fn typed_function_signature(
    symbol_name: String,
    function: &crate::hir::Function,
    model: &SemanticModel,
    nominal_kinds: &HashMap<DefId, DefinitionKind>,
    skip_receiver: bool,
) -> Result<FunctionSignature, String> {
    Ok(FunctionSignature {
        symbol_name,
        return_type: llvm_type_id(function.result, model, nominal_kinds)?,
        parameters: function
            .parameters
            .iter()
            .skip(usize::from(skip_receiver))
            .map(|parameter| llvm_type_id(parameter.ty, model, nominal_kinds))
            .collect::<Result<Vec<_>, String>>()?,
    })
}

fn typed_nominal_kinds(module: &crate::hir::Module) -> HashMap<DefId, DefinitionKind> {
    module
        .items
        .iter()
        .filter_map(|item| {
            let definition = item.definition?;
            let kind = match item.kind {
                crate::hir::ItemKind::Struct(_) => DefinitionKind::Struct,
                crate::hir::ItemKind::Enum(_) => DefinitionKind::Enum,
                crate::hir::ItemKind::Trait(_) => DefinitionKind::Trait,
                crate::hir::ItemKind::Shape(_) => DefinitionKind::Shape,
                _ => return None,
            };
            Some((definition, kind))
        })
        .collect()
}

fn build_typed_trait_layout(
    definition: DefId,
    declarations: &HashMap<DefId, &crate::hir::Trait>,
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
        .ok_or_else(|| format!("missing HIR declaration for trait `{name}`"))?;
    visiting.push(definition);
    let mut methods = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for supertrait in &declaration.supertraits {
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
    for method in &declaration.methods {
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

fn layouts_with_raw<'a>(
    raw_names: impl Iterator<Item = &'a String>,
    layouts: &HashMap<String, StructLayout>,
) -> HashMap<String, StructLayout> {
    let mut merged = layouts.clone();
    for name in raw_names {
        merged.entry(name.clone()).or_insert_with(|| StructLayout {
            name: name.clone(),
            fields: Vec::new(),
        });
    }
    merged
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
    closure_env: Option<ClosureEnv>,
    scopes: Vec<HashMap<String, LocalVar>>,
    deferred_scopes: Vec<Vec<DeferredExpression>>,
    lines: Vec<String>,
    temp_counter: usize,
    label_counter: usize,
    terminated: bool,
    unsafe_depth: usize,
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

fn sanitize_name(name: &str) -> String {
    name.chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
        .collect()
}

pub struct CompiledArtifact {
    pub llvm_ir_path: PathBuf,
    pub binary_path: PathBuf,
}

/// Input accepted by LLVM code generation.
///
/// `CheckedProgram` is the production path. The `Node` implementation keeps
/// focused backend unit tests useful during the HIR-to-LLVM migration.
pub trait CodegenInput {
    fn legacy_codegen_program(&self) -> &Node;

    fn checked_hir(&self) -> Option<(&crate::hir::Module, &crate::semantics::SemanticModel)> {
        None
    }
}

impl CodegenInput for Node {
    fn legacy_codegen_program(&self) -> &Node {
        self
    }
}

impl CodegenInput for crate::pipeline::CheckedProgram {
    fn legacy_codegen_program(&self) -> &Node {
        self.legacy_codegen()
    }

    fn checked_hir(&self) -> Option<(&crate::hir::Module, &crate::semantics::SemanticModel)> {
        Some((&self.hir, &self.semantics))
    }
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
const RUNTIME_C_SOURCE: &str = include_str!("../runtime/skunk_runtime.c");
#[cfg(target_os = "macos")]
const RUNTIME_WINDOW_SOURCE: &str = include_str!("../runtime/skunk_window_runtime.m");

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
    program: &impl CodegenInput,
    source_path: &Path,
    output_path: &Path,
) -> Result<CompiledArtifact, String> {
    compile_to_executable_with_options(program, source_path, output_path, &BuildOptions::default())
}

/// Compiles a checked Skunk program into LLVM IR and a native executable.
pub fn compile_to_executable_with_options(
    program: &impl CodegenInput,
    source_path: &Path,
    output_path: &Path,
    options: &BuildOptions,
) -> Result<CompiledArtifact, String> {
    let llvm_ir = compile_to_llvm_ir(program)?;
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
pub fn compile_to_llvm_ir(program: &impl CodegenInput) -> Result<String, String> {
    let typed_backend = if let Some((hir, semantics)) = program.checked_hir() {
        crate::hir_validation::validate(hir, semantics).map_err(|diagnostics| {
            diagnostics
                .into_iter()
                .map(|diagnostic| diagnostic.to_string())
                .collect::<Vec<_>>()
                .join("\n")
        })?;
        Some((
            collect_typed_layouts(hir, semantics)?,
            collect_typed_signatures(hir, semantics)?,
            collect_typed_implementations(hir, semantics)?,
        ))
    } else {
        None
    };
    let program = program.legacy_codegen_program();
    let statements = match program {
        Node::Program { statements } => statements,
        other => {
            return Err(format!(
                "expected a program root for LLVM compilation, found `{:?}`",
                other
            ))
        }
    };

    let (typed_layouts, typed_signatures, typed_implementations) = match typed_backend {
        Some((layouts, signatures, implementations)) => {
            (Some(layouts), Some(signatures), Some(implementations))
        }
        None => (None, None, None),
    };
    let (structs, enums, traits) = if let Some(layouts) = typed_layouts {
        layouts
    } else {
        let structs = collect_struct_layouts(statements)?;
        let enums = collect_enum_layouts(statements, &structs)?;
        let traits = collect_trait_layouts(statements, &structs, &enums)?;
        (structs, enums, traits)
    };
    let mut signatures = HashMap::<String, FunctionSignature>::new();
    let mut functions = Vec::<FunctionPlan>::new();
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

    for statement in statements {
        match statement {
            Node::FunctionDeclaration {
                name,
                parameters,
                return_type,
                body,
                ..
            } => {
                let params = parameters
                    .iter()
                    .map(|(_, ty)| llvm_type(ty, &structs, &enums, &traits))
                    .collect::<Result<Vec<_>, _>>()?;
                let llvm_return_type = llvm_type(return_type, &structs, &enums, &traits)?;
                signatures.insert(
                    name.clone(),
                    FunctionSignature {
                        symbol_name: format!("skunk_{}", name),
                        return_type: llvm_return_type,
                        parameters: params,
                    },
                );
                functions.push(FunctionPlan {
                    signature_key: name.clone(),
                    symbol_name: format!("skunk_{}", name),
                    parameters: parameters.clone(),
                    body: body.clone(),
                    is_method: false,
                });
            }
            Node::ExternFunctionDeclaration {
                name,
                parameters,
                return_type,
            } => {
                if RESERVED_RUNTIME_SYMBOLS.contains(&name.as_str()) {
                    return Err(format!(
                        "extern function `{}` redeclares a reserved runtime symbol",
                        name
                    ));
                }
                if let Some(existing) = signatures.get(name) {
                    // Identical redeclarations (e.g. the same binding imported
                    // through two modules) are tolerated; conflicts are not.
                    let params = parameters
                        .iter()
                        .map(|(_, ty)| llvm_type(ty, &structs, &enums, &traits))
                        .collect::<Result<Vec<_>, _>>()?;
                    let return_ty = llvm_type(return_type, &structs, &enums, &traits)?;
                    if existing.parameters != params || existing.return_type != return_ty {
                        return Err(format!(
                            "conflicting extern declarations for `{}`",
                            name
                        ));
                    }
                    continue;
                }
                let params = parameters
                    .iter()
                    .map(|(_, ty)| llvm_type(ty, &structs, &enums, &traits))
                    .collect::<Result<Vec<_>, _>>()?;
                let llvm_return_type = llvm_type(return_type, &structs, &enums, &traits)?;
                extern_declares.push(format!(
                    "declare {} @{}({})",
                    llvm_return_type.ir(),
                    name,
                    params
                        .iter()
                        .map(|param| param.ir())
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
                signatures.insert(
                    name.clone(),
                    FunctionSignature {
                        symbol_name: name.clone(),
                        return_type: llvm_return_type,
                        parameters: params,
                    },
                );
            }
            Node::EOI => {}
            Node::TraitDeclaration { .. }
            | Node::ShapeDeclaration { .. }
            | Node::ImplDeclaration { .. } => {}
            Node::StructDeclaration {
                name,
                functions: nominal_functions,
                ..
            }
            | Node::EnumDeclaration {
                name,
                functions: nominal_functions,
                ..
            } => {
                for function in nominal_functions {
                    if let Node::FunctionDeclaration {
                        name: method_name,
                        parameters,
                        return_type,
                        body,
                        ..
                    } = function
                    {
                        let mut parameter_types = Vec::new();
                        let has_receiver = parameters
                            .first()
                            .is_some_and(|(_, param_type)| ast::is_self_type(param_type));
                        let mut compile_params = Vec::new();
                        for (index, (param_name, param_type)) in parameters.iter().enumerate() {
                            if has_receiver && index == 0 {
                                continue;
                            }
                            parameter_types.push(llvm_type(param_type, &structs, &enums, &traits)?);
                            compile_params.push((param_name.clone(), param_type.clone()));
                        }
                        let llvm_return_type = llvm_type(return_type, &structs, &enums, &traits)?;
                        let key = format!("{}::{}", name, method_name);
                        let symbol_name =
                            format!("skunk_{}_{}", sanitize_name(name), sanitize_name(method_name));
                        signatures.insert(
                            key.clone(),
                            FunctionSignature {
                                symbol_name: symbol_name.clone(),
                                return_type: llvm_return_type.clone(),
                                parameters: parameter_types,
                            },
                        );
                        functions.push(FunctionPlan {
                            signature_key: key,
                            symbol_name,
                            parameters: if has_receiver {
                                let mut method_params =
                                    vec![("self".to_string(), Type::Custom(name.clone()))];
                                method_params.extend(compile_params);
                                method_params
                            } else {
                                compile_params
                            },
                            body: body.clone(),
                            is_method: has_receiver,
                        });
                    }
                }
            }
            other => {
                return Err(format!(
                    "LLVM backend currently expects top-level function, struct, and enum declarations only, found `{:?}`",
                    other
                ))
            }
        }
    }

    if let Some(typed_signatures) = typed_signatures {
        signatures = typed_signatures;
    }

    if !signatures.contains_key("main") {
        return Err("LLVM backend currently requires `function main(): ... {}`".to_string());
    }

    let implementations = if let Some(implementations) = typed_implementations {
        implementations
    } else {
        let mut implementations = Vec::new();
        for statement in statements {
            let Node::ImplDeclaration {
                generic_params,
                trait_types,
                target_type,
                ..
            } = statement
            else {
                continue;
            };
            if !generic_params.is_empty() {
                continue;
            }
            let Type::Custom(target_name) = target_type else {
                return Err(format!(
                    "runtime trait values currently require concrete nominal impl targets, found `{}`",
                    ast::type_to_string(target_type)
                ));
            };
            for trait_type in trait_types {
                let Type::Custom(trait_name) = trait_type else {
                    return Err(format!(
                        "LLVM backend requires a concrete trait implementation, found `{}`",
                        ast::type_to_string(trait_type)
                    ));
                };
                implementations.push((trait_name.clone(), target_name.clone()));
            }
        }
        implementations
    };
    for (trait_name, target_name) in implementations {
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
    let mut lambda_counter = 0usize;

    for function in &functions {
        let signature = signatures.get(&function.signature_key).ok_or_else(|| {
            format!(
                "missing typed signature for function `{}`",
                function.signature_key
            )
        })?;
        let llvm_return_type = signature.return_type.clone();
        let param_defs = if function.is_method {
            let mut defs = vec![format!("ptr %arg0")];
            for (index, ty) in signature.parameters.iter().enumerate() {
                defs.push(format!("{} %arg{}", ty.ir(), index + 1));
            }
            defs
        } else {
            signature
                .parameters
                .iter()
                .enumerate()
                .map(|(index, ty)| format!("{} %arg{}", ty.ir(), index))
                .collect()
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
        let compiler = FunctionCompiler::new(
            &function.symbol_name,
            llvm_return_type.clone(),
            dependencies,
            None,
        );
        let body_lines = compiler.compile(&function.parameters, &function.body)?;
        let mut function_ir = String::new();
        let _ = writeln!(
            function_ir,
            "define {} @{}({}) {{",
            llvm_return_type.ir(),
            function.symbol_name,
            param_defs.join(", ")
        );
        let _ = writeln!(function_ir, "entry:");
        for line in body_lines {
            let _ = writeln!(function_ir, "{}", line);
        }
        let _ = writeln!(function_ir, "}}");
        function_irs.push(function_ir);
    }

    let main_signature = signatures.get("main").expect("validated above");
    if !main_signature.parameters.is_empty() {
        return Err("LLVM backend requires `main` to take no parameters".to_string());
    }

    let c_main_body = match main_signature.return_type {
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

    Ok(ir)
}
