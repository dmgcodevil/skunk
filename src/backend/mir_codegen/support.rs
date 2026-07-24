//! Semantic identities and layout indexes shared by MIR instruction selection.

use super::*;

pub(in crate::backend) struct CodegenContext<'a> {
    pub(super) model: &'a SemanticModel,
    pub(super) nominal_kinds: HashMap<DefId, DefinitionKind>,
    pub(super) direct_signature_keys: HashMap<DefId, String>,
    pub(super) field_indices: HashMap<FieldId, usize>,
    pub(super) variants: HashMap<VariantId, VariantCodegenInfo>,
    pub(in crate::backend) globals: HashMap<DefId, GlobalCodegenInfo>,
    pub(super) closure_functions: HashMap<NodeId, ir::Function>,
}

#[derive(Clone)]
pub(in crate::backend) struct GlobalCodegenInfo {
    pub(in crate::backend) symbol_name: String,
    pub(in crate::backend) llvm_type: LlvmType,
}

pub(super) struct VariantCodegenInfo {
    pub(super) enum_name: String,
    pub(super) tag: usize,
    pub(super) payload_types: Vec<LlvmType>,
    pub(super) field_indices: Vec<usize>,
}

impl<'a> CodegenContext<'a> {
    pub(in crate::backend) fn new(
        module: &ir::Module,
        model: &'a SemanticModel,
    ) -> Result<Self, String> {
        let nominal_kinds = typed_nominal_kinds(module);
        let mut direct_signature_keys = HashMap::new();
        let mut field_indices = HashMap::new();
        let mut variants = HashMap::new();
        let mut globals = HashMap::new();
        let closure_functions = module
            .functions
            .iter()
            .filter(|function| function.origin == ir::FunctionOrigin::Closure)
            .map(|function| (function.source, function.clone()))
            .collect();

        for declaration in &module.declarations {
            match &declaration.kind {
                DeclarationKind::ExternFunction { definition, .. } => {
                    direct_signature_keys.insert(
                        *definition,
                        definition_name(model, *definition)?.to_string(),
                    );
                }
                DeclarationKind::Struct { fields, .. } => {
                    for (index, field) in fields.iter().enumerate() {
                        field_indices.insert(field.id, index);
                    }
                }
                DeclarationKind::Enum {
                    definition,
                    variants: declared_variants,
                    ..
                } => {
                    let enum_name = definition_name(model, *definition)?.to_string();
                    let mut next_field_index = 1usize;
                    for (tag, variant) in declared_variants.iter().enumerate() {
                        let payload_types = variant
                            .payload
                            .iter()
                            .map(|ty| llvm_type_id(*ty, model, &nominal_kinds))
                            .collect::<Result<Vec<_>, _>>()?;
                        let field_indices = (0..payload_types.len())
                            .map(|_| {
                                let index = next_field_index;
                                next_field_index += 1;
                                index
                            })
                            .collect();
                        variants.insert(
                            variant.id,
                            VariantCodegenInfo {
                                enum_name: enum_name.clone(),
                                tag,
                                payload_types,
                                field_indices,
                            },
                        );
                    }
                }
                DeclarationKind::Global { definition, ty, .. } => {
                    let name = definition_name(model, *definition)?;
                    globals.insert(
                        *definition,
                        GlobalCodegenInfo {
                            symbol_name: format!("skunk_global_{}", sanitize_name(name)),
                            llvm_type: llvm_type_id(*ty, model, &nominal_kinds)?,
                        },
                    );
                }
                _ => {}
            }
        }
        for function in &module.functions {
            if let (Some(definition), Some(key)) =
                (function.definition(), signature_key(function, model)?)
            {
                direct_signature_keys.insert(definition, key);
            }
        }

        Ok(Self {
            model,
            nominal_kinds,
            direct_signature_keys,
            field_indices,
            variants,
            globals,
            closure_functions,
        })
    }

    pub(in crate::backend) fn function_key(
        &self,
        function: &ir::Function,
    ) -> Result<Option<String>, String> {
        signature_key(function, self.model)
    }

    pub(super) fn llvm_type(&self, ty: TypeId) -> Result<LlvmType, String> {
        llvm_type_id(ty, self.model, &self.nominal_kinds)
    }
}

fn signature_key(function: &ir::Function, model: &SemanticModel) -> Result<Option<String>, String> {
    let Some(definition) = function.definition() else {
        return Ok(None);
    };
    let name = definition_name(model, definition)?;
    Ok(Some(match function.owner() {
        Some(owner) => format!("{}::{name}", definition_name(model, owner)?),
        None => name.to_string(),
    }))
}
