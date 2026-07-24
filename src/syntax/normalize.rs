//! Source-level normalization that is independent of semantic types.
//!
//! Behavior blocks remain explicit in the parser output, then become nominal
//! methods plus implementation records before name resolution. Later phases
//! therefore consume one declaration shape.

use super::ast::*;
use crate::diagnostic::Diagnostic;
use crate::ids::NodeId;
use std::collections::{HashMap, HashSet};

pub fn normalize(mut module: Module) -> Result<Module, Vec<Diagnostic>> {
    let mut diagnostics = Vec::new();
    let mut next_id = module.id.local_index().saturating_add(1);
    let mut nominals = HashMap::<String, NominalInfo>::new();

    for entry in &module.entries {
        match &entry.kind {
            TopLevelKind::Struct(declaration) => {
                nominals.insert(
                    declaration.name.clone(),
                    NominalInfo {
                        is_struct: true,
                        generic_names: declaration
                            .generic_parameters
                            .iter()
                            .map(|parameter| parameter.name.clone())
                            .collect(),
                        variant_names: HashSet::new(),
                    },
                );
            }
            TopLevelKind::Enum(declaration) => {
                nominals.insert(
                    declaration.name.clone(),
                    NominalInfo {
                        is_struct: false,
                        generic_names: declaration
                            .generic_parameters
                            .iter()
                            .map(|parameter| parameter.name.clone())
                            .collect(),
                        variant_names: declaration
                            .variants
                            .iter()
                            .map(|variant| variant.name.clone())
                            .collect(),
                    },
                );
            }
            _ => {}
        }
    }

    // Infer the natural binder in `conform Trait[T] for Box[T]` when `Box`
    // declares exactly those parameters.
    for entry in &mut module.entries {
        let TopLevelKind::Conformance(declaration) = &mut entry.kind else {
            continue;
        };
        if !declaration.generic_parameters.is_empty() {
            continue;
        }
        let Some((target_name, arguments)) = named_type(&declaration.target) else {
            continue;
        };
        let Some(info) = nominals.get(target_name) else {
            continue;
        };
        let argument_names = simple_type_names(arguments);
        if argument_names.as_deref()
            == Some(
                &info
                    .generic_names
                    .iter()
                    .map(String::as_str)
                    .collect::<Vec<_>>(),
            )
        {
            declaration.generic_parameters = info
                .generic_names
                .iter()
                .map(|name| {
                    let id = NodeId::in_file(entry.span.file, next_id);
                    next_id = next_id.saturating_add(1);
                    GenericParameter {
                        id,
                        span: entry.span,
                        name: name.clone(),
                        capabilities: Vec::new(),
                        lower_bound: None,
                        upper_bound: None,
                    }
                })
                .collect();
        }
    }

    let mut methods = HashMap::<String, Vec<FunctionDecl>>::new();
    let mut seen_methods = HashMap::<String, HashSet<String>>::new();

    for entry in &module.entries {
        let (kind, generic_parameters, target, functions, require_receiver) = match &entry.kind {
            TopLevelKind::Attach(declaration) => (
                "attach",
                &declaration.generic_parameters,
                &declaration.target,
                &declaration.methods,
                false,
            ),
            TopLevelKind::Conformance(declaration) => (
                "conform",
                &declaration.generic_parameters,
                &declaration.target,
                &declaration.methods,
                true,
            ),
            _ => continue,
        };
        let Some((target_name, target_arguments)) = named_type(target) else {
            diagnostics.push(
                Diagnostic::error(format!("`{kind}` target must be a nominal type"))
                    .with_code("E1100")
                    .at(entry.span),
            );
            continue;
        };
        let Some(info) = nominals.get(target_name) else {
            diagnostics.push(
                Diagnostic::error(format!(
                    "`{kind}` target `{target_name}` must name an existing nominal type"
                ))
                .with_code("E1101")
                .at(entry.span),
            );
            continue;
        };
        if require_receiver && !info.is_struct {
            diagnostics.push(
                Diagnostic::error(format!(
                    "`{kind}` target `{target_name}` must name an existing struct type"
                ))
                .with_code("E1102")
                .at(entry.span),
            );
        }
        validate_behavior_generics(
            kind,
            entry,
            generic_parameters,
            target_name,
            target_arguments,
            info,
            &mut diagnostics,
        );

        let seen = seen_methods.entry(target_name.to_string()).or_default();
        let merged = methods.entry(target_name.to_string()).or_default();
        for function in functions {
            if require_receiver
                && !function.parameters.first().is_some_and(|parameter| {
                    matches!(parameter.kind, ParameterKind::Receiver { .. })
                })
            {
                diagnostics.push(
                    Diagnostic::error(format!(
                        "`{kind}` method `{}` on `{target_name}` must declare `self` as its first parameter",
                        function.name
                    ))
                    .with_code("E1103")
                    .at(function.body.span),
                );
                continue;
            }
            if !seen.insert(function.name.clone()) {
                diagnostics.push(
                    Diagnostic::error(format!(
                        "duplicate attached function `{}` on `{target_name}`",
                        function.name
                    ))
                    .with_code("E1104")
                    .at(function.body.span),
                );
                continue;
            }
            let is_static = function
                .parameters
                .first()
                .is_none_or(|parameter| !matches!(parameter.kind, ParameterKind::Receiver { .. }));
            if !require_receiver && is_static && info.variant_names.contains(&function.name) {
                diagnostics.push(
                    Diagnostic::error(format!(
                        "static attached function `{}` on enum `{target_name}` conflicts with a variant constructor",
                        function.name
                    ))
                    .with_code("E1105")
                    .at(function.body.span),
                );
                continue;
            }
            merged.push(function.clone());
        }
    }

    if !diagnostics.is_empty() {
        return Err(diagnostics);
    }

    let mut output = Vec::with_capacity(module.entries.len());
    for mut entry in module.entries {
        match &mut entry.kind {
            TopLevelKind::Struct(declaration) => {
                declaration
                    .methods
                    .extend(methods.remove(&declaration.name).unwrap_or_default());
                output.push(entry);
            }
            TopLevelKind::Enum(declaration) => {
                declaration
                    .methods
                    .extend(methods.remove(&declaration.name).unwrap_or_default());
                output.push(entry);
            }
            TopLevelKind::Attach(_) => {}
            TopLevelKind::Conformance(declaration) => {
                entry.kind = TopLevelKind::Implementation(ImplementationDecl {
                    generic_parameters: std::mem::take(&mut declaration.generic_parameters),
                    traits: std::mem::take(&mut declaration.traits),
                    target: declaration.target.clone(),
                });
                output.push(entry);
            }
            _ => output.push(entry),
        }
    }
    module.entries = output;
    Ok(module)
}

#[derive(Debug)]
struct NominalInfo {
    is_struct: bool,
    generic_names: Vec<String>,
    variant_names: HashSet<String>,
}

fn named_type(ty: &TypeSyntax) -> Option<(&str, &[TypeSyntax])> {
    let TypeSyntaxKind::Named { path, arguments } = &ty.kind else {
        return None;
    };
    Some((path.segments.first()?.as_str(), arguments))
}

fn simple_type_names(types: &[TypeSyntax]) -> Option<Vec<&str>> {
    types
        .iter()
        .map(|argument| match &argument.kind {
            TypeSyntaxKind::Named { path, arguments } if arguments.is_empty() => {
                path.segments.first().map(String::as_str)
            }
            _ => None,
        })
        .collect()
}

fn validate_behavior_generics(
    kind: &str,
    entry: &TopLevel,
    parameters: &[GenericParameter],
    target_name: &str,
    target_arguments: &[TypeSyntax],
    nominal: &NominalInfo,
    diagnostics: &mut Vec<Diagnostic>,
) {
    let parameter_names = parameters
        .iter()
        .map(|parameter| parameter.name.as_str())
        .collect::<Vec<_>>();
    let expected = nominal
        .generic_names
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>();
    if expected.is_empty() {
        if !parameters.is_empty() || !target_arguments.is_empty() {
            diagnostics.push(
                Diagnostic::error(format!(
                    "`{kind}` target `{target_name}` is not generic, so it cannot declare generic parameters"
                ))
                .with_code("E1106")
                .at(entry.span),
            );
        }
    } else if parameter_names != expected
        || simple_type_names(target_arguments).as_deref() != Some(expected.as_slice())
    {
        diagnostics.push(
            Diagnostic::error(format!(
                "`{kind}` target `{target_name}` must declare generic parameters as `[{}]`",
                nominal.generic_names.join(", ")
            ))
            .with_code("E1107")
            .at(entry.span),
        );
    }
}
