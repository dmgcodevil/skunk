//! Front end for generic specialization.
//!
//! This module owns source-template indexes and deterministic output assembly.
//! The implementation is split into semantic validation, AST transformation,
//! and concrete specialization so each part can be read independently.

use crate::ast::{self, Literal, Metadata, Node, Operator, Type, UnaryOperator};
use std::collections::{HashMap, HashSet};
use std::ops::Deref;

mod semantics;
mod specialize;
mod transform;

#[cfg(test)]
mod tests;

#[derive(Clone)]
struct FunctionTemplate {
    name: String,
    generic_params: Vec<String>,
    generic_bounds: HashMap<String, Vec<String>>,
    subtype_bounds: HashMap<String, ast::SubtypeBounds>,
    parameters: Vec<(String, Type)>,
    return_type: Type,
    body: Vec<Node>,
}

#[derive(Clone)]
struct StructTemplate {
    name: String,
    generic_params: Vec<String>,
    generic_bounds: HashMap<String, Vec<String>>,
    subtype_bounds: HashMap<String, ast::SubtypeBounds>,
    fields: Vec<(String, Type)>,
    functions: Vec<Node>,
}

#[derive(Clone)]
struct EnumTemplate {
    name: String,
    generic_params: Vec<String>,
    generic_bounds: HashMap<String, Vec<String>>,
    subtype_bounds: HashMap<String, ast::SubtypeBounds>,
    variants: Vec<ast::EnumVariant>,
    functions: Vec<Node>,
}

#[derive(Clone)]
struct TraitTemplate {
    name: String,
    generic_params: Vec<String>,
    generic_bounds: HashMap<String, Vec<String>>,
    subtype_bounds: HashMap<String, ast::SubtypeBounds>,
    supertraits: Vec<String>,
    methods: Vec<ast::TraitMethodSignature>,
}

#[derive(Clone)]
struct ShapeTemplate {
    name: String,
    methods: Vec<ast::TraitMethodSignature>,
}

#[derive(Clone)]
struct ImplTemplate {
    generic_params: Vec<String>,
    generic_bounds: HashMap<String, Vec<String>>,
    subtype_bounds: HashMap<String, ast::SubtypeBounds>,
    trait_types: Vec<Type>,
    target_type: Type,
}

#[derive(Clone)]
struct FunctionSignature {
    parameters: Vec<Type>,
    return_type: Type,
}

#[derive(Clone)]
struct TypeAliasTemplate {
    name: String,
    generic_params: Vec<String>,
    generic_bounds: HashMap<String, Vec<String>>,
    subtype_bounds: HashMap<String, ast::SubtypeBounds>,
    target_type: Type,
}

fn attached_function_signature(
    functions: &[Node],
    function_name: &str,
    requires_receiver: bool,
) -> Option<(Vec<(String, Type)>, Type)> {
    functions.iter().find_map(|function| match function {
        Node::FunctionDeclaration {
            name,
            parameters,
            return_type,
            ..
        } if name == function_name
            && parameters
                .first()
                .is_some_and(|(_, sk_type)| ast::is_self_type(sk_type))
                == requires_receiver =>
        {
            Some((parameters.clone(), return_type.clone()))
        }
        _ => None,
    })
}

#[derive(Clone, Copy)]
enum ConstraintKind {
    Trait,
    Shape,
}

#[derive(Default, Clone)]
struct Env {
    scopes: Vec<HashMap<String, Type>>,
}

impl Env {
    fn new() -> Self {
        Self {
            scopes: vec![HashMap::new()],
        }
    }

    fn push(&mut self) {
        self.scopes.push(HashMap::new());
    }

    fn pop(&mut self) {
        self.scopes.pop();
    }

    fn insert(&mut self, name: String, sk_type: Type) {
        self.scopes
            .last_mut()
            .expect("scope exists")
            .insert(name, sk_type);
    }

    fn get(&self, name: &str) -> Option<Type> {
        for scope in self.scopes.iter().rev() {
            if let Some(sk_type) = scope.get(name) {
                return Some(sk_type.clone());
            }
        }
        None
    }
}

/// Memoizes generated declarations while preserving their discovery order for
/// deterministic output. Keeping both collections together prevents a newly
/// generated node from being cached without also being scheduled for emission.
#[derive(Default)]
struct GeneratedNodes {
    nodes: HashMap<String, Node>,
    order: Vec<String>,
}

impl GeneratedNodes {
    fn contains(&self, name: &str) -> bool {
        self.nodes.contains_key(name)
    }

    fn insert(&mut self, name: String, node: Node) {
        if self.nodes.insert(name.clone(), node).is_none() {
            self.order.push(name);
        }
    }

    fn values(&self) -> impl Iterator<Item = &Node> {
        self.order.iter().filter_map(|name| self.nodes.get(name))
    }
}

/// Expands generics and synthesizes any derived program structure needed before
/// type checking and code generation.
pub fn prepare_program(node: &Node) -> Result<Node, String> {
    let Node::Program { statements } = node else {
        return Ok(node.clone());
    };
    let mut monomorphizer = Monomorphizer::new(statements)?;
    monomorphizer.prepare()
}

/// Stateful preparation pass that turns generic source declarations into the
/// concrete program consumed by the type checker and LLVM backend.
struct Monomorphizer {
    /// Generic function declarations indexed by their source name. These are
    /// templates only; a concrete function node is emitted when a call supplies
    /// enough type information to specialize one.
    generic_functions: HashMap<String, FunctionTemplate>,
    /// Non-generic function signatures and extern signatures indexed by name.
    /// Their original declarations are kept separately in `root_statements`.
    concrete_functions: HashMap<String, FunctionTemplate>,
    /// Generic struct declarations indexed by their unspecialized source name.
    generic_structs: HashMap<String, StructTemplate>,
    /// Non-generic struct declarations indexed for field and attached-function
    /// lookup while the root declarations are transformed.
    concrete_structs: HashMap<String, StructTemplate>,
    /// Generic enum declarations indexed by their unspecialized source name.
    generic_enums: HashMap<String, EnumTemplate>,
    /// Non-generic enum declarations indexed for variant and attached-function
    /// lookup while the root declarations are transformed.
    concrete_enums: HashMap<String, EnumTemplate>,
    /// Trait templates indexed by name. This starts with source traits and is
    /// extended with concrete specializations of generic traits as needed.
    traits: HashMap<String, TraitTemplate>,
    /// Structural shape declarations used to validate capability bounds. Shapes
    /// guide preparation but do not become runtime declarations.
    shapes: HashMap<String, ShapeTemplate>,
    /// Transparent type-alias templates. Aliases are expanded during this pass
    /// and do not appear as declarations in the prepared program.
    type_aliases: HashMap<String, TypeAliasTemplate>,
    /// Every source `impl` template, both concrete and generic. Generic entries
    /// are matched and specialized when a concrete target type is requested.
    impls: Vec<ImplTemplate>,
    /// Known trait conformance by concrete target type string. It is populated
    /// while validating non-generic impls and reused as a fast path for bound
    /// and subtype checks; generic impl candidates are matched from `impls`.
    implemented_traits: HashMap<String, HashSet<String>>,
    /// Non-generic source trait declarations that can be copied directly to the
    /// beginning of the prepared program.
    root_traits: Vec<Node>,
    /// Source impl declarations with no impl parameters and only plain named
    /// trait references, so they can be copied directly to prepared output.
    root_concrete_impls: Vec<Node>,
    /// Deduplication keys of the form `trait=>target` for concrete impl nodes.
    /// This includes direct impls and generated impls implied by supertraits.
    generated_impl_keys: HashSet<String>,
    /// Concrete impl declarations synthesized from generic impl templates,
    /// specialized trait references, or implied supertrait conformances.
    generated_impls: Vec<Node>,
    /// Specialized generic functions, memoized by mangled concrete symbol and
    /// retained in dependency-discovery order for final emission.
    generated_functions: GeneratedNodes,
    /// Specialized generic structs in deterministic discovery order.
    generated_structs: GeneratedNodes,
    /// Specialized generic enums in deterministic discovery order.
    generated_enums: GeneratedNodes,
    /// Specialized generic traits in deterministic discovery order. Generated
    /// trait templates are also registered in `traits` for later lookup.
    generated_traits: GeneratedNodes,
    /// Function specialization symbols currently being constructed. Re-entering
    /// one means a recursive reference can reuse its symbol instead of trying to
    /// generate the same function indefinitely.
    function_stack: HashSet<String>,
    /// Struct specialization symbols currently being constructed, used to stop
    /// recursive nominal types from repeatedly generating the same struct.
    struct_stack: HashSet<String>,
    /// Enum specialization symbols currently being constructed, used to stop
    /// recursive nominal types from repeatedly generating the same enum.
    enum_stack: HashSet<String>,
    /// Trait specialization symbols currently being constructed. For example,
    /// a method or bound that refers back to the same concrete generic trait can
    /// reuse the in-progress symbol instead of recursing forever.
    trait_stack: HashSet<String>,
    /// Source statements scheduled for normal transformation and emission.
    /// Declarations retained only as templates, or emitted separately through
    /// `root_traits` and `root_concrete_impls`, are excluded from this list.
    root_statements: Vec<Node>,
}

impl Monomorphizer {
    /// Indexes source declarations as reusable templates and records the
    /// declarations that must be transformed directly into prepared output.
    fn new(statements: &[Node]) -> Result<Self, String> {
        let mut generic_functions = HashMap::new();
        let mut concrete_functions = HashMap::new();
        let mut generic_structs = HashMap::new();
        let mut concrete_structs = HashMap::new();
        let mut generic_enums = HashMap::new();
        let mut concrete_enums = HashMap::new();
        let mut traits = HashMap::new();
        let mut shapes = HashMap::new();
        let mut type_aliases = HashMap::new();
        let mut impls = Vec::new();
        let mut root_traits = Vec::new();
        let mut root_concrete_impls = Vec::new();
        let mut root_statements = Vec::new();

        for statement in statements {
            let statement = match statement {
                Node::Export { declaration } => declaration.as_ref(),
                other => other,
            };
            match statement {
                Node::GenericFunctionDeclaration {
                    name,
                    generic_params,
                    generic_bounds,
                    subtype_bounds,
                    parameters,
                    return_type,
                    body,
                    ..
                } => {
                    generic_functions.insert(
                        name.clone(),
                        FunctionTemplate {
                            name: name.clone(),
                            generic_params: generic_params.clone(),
                            generic_bounds: generic_bounds.clone(),
                            subtype_bounds: subtype_bounds.clone(),
                            parameters: parameters.clone(),
                            return_type: return_type.clone(),
                            body: body.clone(),
                        },
                    );
                }
                Node::FunctionDeclaration {
                    name,
                    parameters,
                    return_type,
                    body,
                    lambda: false,
                } => {
                    concrete_functions.insert(
                        name.clone(),
                        FunctionTemplate {
                            name: name.clone(),
                            generic_params: Vec::new(),
                            generic_bounds: HashMap::new(),
                            subtype_bounds: HashMap::new(),
                            parameters: parameters.clone(),
                            return_type: return_type.clone(),
                            body: body.clone(),
                        },
                    );
                    root_statements.push(statement.clone());
                }
                Node::GenericStructDeclaration {
                    name,
                    generic_params,
                    generic_bounds,
                    subtype_bounds,
                    fields,
                    functions,
                } => {
                    generic_structs.insert(
                        name.clone(),
                        StructTemplate {
                            name: name.clone(),
                            generic_params: generic_params.clone(),
                            generic_bounds: generic_bounds.clone(),
                            subtype_bounds: subtype_bounds.clone(),
                            fields: fields.clone(),
                            functions: functions.clone(),
                        },
                    );
                }
                Node::StructDeclaration {
                    name,
                    fields,
                    functions,
                } => {
                    concrete_structs.insert(
                        name.clone(),
                        StructTemplate {
                            name: name.clone(),
                            generic_params: Vec::new(),
                            generic_bounds: HashMap::new(),
                            subtype_bounds: HashMap::new(),
                            fields: fields.clone(),
                            functions: functions.clone(),
                        },
                    );
                    root_statements.push(statement.clone());
                }
                Node::GenericEnumDeclaration {
                    name,
                    generic_params,
                    generic_bounds,
                    subtype_bounds,
                    variants,
                    functions,
                } => {
                    generic_enums.insert(
                        name.clone(),
                        EnumTemplate {
                            name: name.clone(),
                            generic_params: generic_params.clone(),
                            generic_bounds: generic_bounds.clone(),
                            subtype_bounds: subtype_bounds.clone(),
                            variants: variants.clone(),
                            functions: functions.clone(),
                        },
                    );
                }
                Node::EnumDeclaration {
                    name,
                    variants,
                    functions,
                } => {
                    concrete_enums.insert(
                        name.clone(),
                        EnumTemplate {
                            name: name.clone(),
                            generic_params: Vec::new(),
                            generic_bounds: HashMap::new(),
                            subtype_bounds: HashMap::new(),
                            variants: variants.clone(),
                            functions: functions.clone(),
                        },
                    );
                    root_statements.push(statement.clone());
                }
                Node::TraitDeclaration {
                    name,
                    generic_params,
                    generic_bounds,
                    subtype_bounds,
                    supertraits,
                    methods,
                } => {
                    traits.insert(
                        name.clone(),
                        TraitTemplate {
                            name: name.clone(),
                            generic_params: generic_params.clone(),
                            generic_bounds: generic_bounds.clone(),
                            subtype_bounds: subtype_bounds.clone(),
                            supertraits: supertraits.clone(),
                            methods: methods.clone(),
                        },
                    );
                    if generic_params.is_empty() {
                        root_traits.push(statement.clone());
                    }
                }
                Node::ShapeDeclaration { name, methods } => {
                    shapes.insert(
                        name.clone(),
                        ShapeTemplate {
                            name: name.clone(),
                            methods: methods.clone(),
                        },
                    );
                }
                Node::TypeAliasDeclaration {
                    name,
                    generic_params,
                    generic_bounds,
                    subtype_bounds,
                    target_type,
                } => {
                    if type_aliases.contains_key(name) {
                        return Err(format!("duplicate type alias `{}`", name));
                    }
                    type_aliases.insert(
                        name.clone(),
                        TypeAliasTemplate {
                            name: name.clone(),
                            generic_params: generic_params.clone(),
                            generic_bounds: generic_bounds.clone(),
                            subtype_bounds: subtype_bounds.clone(),
                            target_type: target_type.clone(),
                        },
                    );
                }
                Node::ImplDeclaration {
                    generic_params,
                    generic_bounds,
                    subtype_bounds,
                    trait_types,
                    target_type,
                } => {
                    impls.push(ImplTemplate {
                        generic_params: generic_params.clone(),
                        generic_bounds: generic_bounds.clone(),
                        subtype_bounds: subtype_bounds.clone(),
                        trait_types: trait_types.clone(),
                        target_type: target_type.clone(),
                    });
                    if generic_params.is_empty()
                        && trait_types
                            .iter()
                            .all(|trait_type| matches!(trait_type, Type::Custom(_)))
                    {
                        root_concrete_impls.push(statement.clone());
                    }
                }
                Node::ExternFunctionDeclaration {
                    name,
                    parameters,
                    return_type,
                } => {
                    // Register the extern signature so calls resolve like calls
                    // to any concrete function; the empty body is never used
                    // because `prepare` passes the declaration through as-is.
                    concrete_functions.insert(
                        name.clone(),
                        FunctionTemplate {
                            name: name.clone(),
                            generic_params: Vec::new(),
                            generic_bounds: HashMap::new(),
                            subtype_bounds: HashMap::new(),
                            parameters: parameters.clone(),
                            return_type: return_type.clone(),
                            body: Vec::new(),
                        },
                    );
                    root_statements.push(statement.clone());
                }
                // Test declarations only take part in `skunk test`, where they
                // are rewritten into plain functions before this pass runs.
                // In a normal build they are simply dropped.
                Node::TestDeclaration { .. } => {}
                Node::Module { .. } | Node::Import { .. } => {}
                Node::EOI => {}
                other => root_statements.push(other.clone()),
            }
        }

        for name in type_aliases.keys() {
            if concrete_structs.contains_key(name)
                || generic_structs.contains_key(name)
                || concrete_enums.contains_key(name)
                || generic_enums.contains_key(name)
                || traits.contains_key(name)
                || shapes.contains_key(name)
            {
                return Err(format!("duplicate type declaration `{}`", name));
            }
        }
        validate_type_alias_cycles(&type_aliases)?;

        Ok(Self {
            generic_functions,
            concrete_functions,
            generic_structs,
            concrete_structs,
            generic_enums,
            concrete_enums,
            traits,
            shapes,
            type_aliases,
            impls,
            implemented_traits: HashMap::new(),
            root_traits,
            root_concrete_impls,
            generated_impl_keys: HashSet::new(),
            generated_impls: Vec::new(),
            generated_functions: GeneratedNodes::default(),
            generated_structs: GeneratedNodes::default(),
            generated_enums: GeneratedNodes::default(),
            generated_traits: GeneratedNodes::default(),
            function_stack: HashSet::new(),
            struct_stack: HashSet::new(),
            enum_stack: HashSet::new(),
            trait_stack: HashSet::new(),
            root_statements,
        })
    }

    /// Validates declaration relationships, transforms root statements, then
    /// emits every discovered specialization in deterministic order.
    fn prepare(&mut self) -> Result<Node, String> {
        self.validate_traits()?;
        self.validate_impls()?;
        let mut output = Vec::<Node>::new();
        output.extend(self.root_traits.clone());
        output.extend(self.root_concrete_impls.clone());
        for statement in self.root_statements.clone() {
            match statement {
                Node::FunctionDeclaration {
                    name,
                    parameters,
                    return_type,
                    body,
                    lambda: false,
                } => {
                    output.push(self.transform_named_function(
                        &name,
                        &parameters,
                        &return_type,
                        &body,
                        &HashMap::new(),
                        None,
                    )?);
                }
                Node::StructDeclaration {
                    name,
                    fields,
                    functions: _,
                } => {
                    self.ensure_runtime_impls_for_type(
                        &Type::Custom(name.clone()),
                        &Type::Custom(name.clone()),
                    )?;
                    let functions = self
                        .concrete_structs
                        .get(&name)
                        .ok_or_else(|| format!("unknown concrete struct `{}`", name))?
                        .functions
                        .clone();
                    output.push(self.transform_struct_decl(
                        &name,
                        &fields,
                        &functions,
                        &HashMap::new(),
                        None,
                    )?);
                }
                Node::EnumDeclaration {
                    name,
                    variants,
                    functions: _,
                } => {
                    self.ensure_runtime_impls_for_type(
                        &Type::Custom(name.clone()),
                        &Type::Custom(name.clone()),
                    )?;
                    let functions = self
                        .concrete_enums
                        .get(&name)
                        .ok_or_else(|| format!("unknown concrete enum `{}`", name))?
                        .functions
                        .clone();
                    output.push(self.transform_enum_decl(
                        &name,
                        &variants,
                        &functions,
                        &HashMap::new(),
                        None,
                    )?);
                }
                Node::EOI => {}
                extern_decl @ Node::ExternFunctionDeclaration { .. } => {
                    output.push(extern_decl);
                }
                other => {
                    let mut env = Env::new();
                    let (statement, _) = self.transform_statement(
                        &other,
                        &mut env,
                        &Type::Void,
                        &HashMap::new(),
                        None,
                    )?;
                    output.push(statement);
                }
            }
        }

        output.extend(self.generated_structs.values().cloned());
        output.extend(self.generated_enums.values().cloned());
        output.extend(self.generated_traits.values().cloned());
        output.extend(self.generated_impls.clone());
        output.extend(self.generated_functions.values().cloned());
        output.push(Node::EOI);
        Ok(Node::Program { statements: output })
    }
}

fn contains_unresolved_generic(
    sk_type: &Type,
    generic_params: &[String],
    substitutions: &HashMap<String, Type>,
) -> bool {
    match sk_type {
        Type::Custom(name) => {
            generic_params.iter().any(|param| param == name) && !substitutions.contains_key(name)
        }
        Type::Const { inner } | Type::BindingConst { inner } => {
            contains_unresolved_generic(inner, generic_params, substitutions)
        }
        Type::Array { elem_type, .. } | Type::Slice { elem_type } => {
            contains_unresolved_generic(elem_type, generic_params, substitutions)
        }
        Type::Reference { target_type, .. } | Type::Pointer { target_type } => {
            contains_unresolved_generic(target_type, generic_params, substitutions)
        }
        Type::GenericInstance { type_arguments, .. }
        | Type::Union(type_arguments)
        | Type::Intersection(type_arguments) => type_arguments
            .iter()
            .any(|argument| contains_unresolved_generic(argument, generic_params, substitutions)),
        Type::Function {
            parameters,
            return_type,
        } => {
            parameters.iter().any(|parameter| {
                contains_unresolved_generic(parameter, generic_params, substitutions)
            }) || contains_unresolved_generic(return_type, generic_params, substitutions)
        }
        _ => false,
    }
}

fn array_item_type(sk_type: &Type) -> Option<Type> {
    match sk_type {
        Type::Array {
            elem_type,
            dimensions,
        } => {
            if dimensions.len() > 1 {
                Some(Type::Array {
                    elem_type: elem_type.clone(),
                    dimensions: dimensions[1..].to_vec(),
                })
            } else {
                Some(ast::strip_const_view(elem_type.as_ref()))
            }
        }
        Type::Slice { elem_type } => Some(ast::strip_const_view(elem_type.as_ref())),
        _ => None,
    }
}

fn indexed_array_type(sk_type: &Type, coordinate_count: usize) -> Option<Type> {
    let mut current = sk_type.clone();
    for _ in 0..coordinate_count {
        current = array_item_type(&current)?;
    }
    Some(current)
}

fn slice_result_type(sk_type: &Type) -> Result<Type, String> {
    match sk_type {
        Type::Array {
            elem_type,
            dimensions,
        } => {
            let sliced_elem_type = if dimensions.len() > 1 {
                Type::Array {
                    elem_type: elem_type.clone(),
                    dimensions: dimensions[1..].to_vec(),
                }
            } else {
                elem_type.as_ref().clone()
            };
            Ok(Type::Slice {
                elem_type: Box::new(sliced_elem_type),
            })
        }
        Type::Slice { elem_type } => Ok(Type::Slice {
            elem_type: elem_type.clone(),
        }),
        other => Err(format!(
            "slice access requires an array or slice, found `{}`",
            ast::type_to_string(other)
        )),
    }
}

fn resolve_binary_result_type(
    operator: &Operator,
    left: &Type,
    right: &Type,
) -> Result<Type, String> {
    match operator {
        Operator::Add => {
            if let Some(promoted) = ast::promoted_numeric_type(left, right) {
                Ok(promoted)
            } else if (*left == Type::String && ast::is_scalar_type(right))
                || (ast::is_scalar_type(left) && *right == Type::String)
            {
                Ok(Type::String)
            } else {
                Err(format!(
                    "unexpected types for +: {:?} and {:?}",
                    left, right
                ))
            }
        }
        Operator::Subtract | Operator::Multiply | Operator::Divide => {
            ast::promoted_numeric_type(left, right).ok_or_else(|| {
                format!(
                    "unexpected numeric operand types: {:?} and {:?}",
                    left, right
                )
            })
        }
        Operator::Mod => {
            if ast::is_integral_type(left) && ast::is_integral_type(right) {
                Ok(ast::promoted_numeric_type(left, right).unwrap())
            } else {
                Err(format!(
                    "unexpected types for %: {:?} and {:?}",
                    left, right
                ))
            }
        }
        Operator::Equals | Operator::NotEquals => Ok(Type::Boolean),
        Operator::LessThan
        | Operator::LessThanOrEqual
        | Operator::GreaterThan
        | Operator::GreaterThanOrEqual
        | Operator::And
        | Operator::Or => Ok(Type::Boolean),
        Operator::Power => ast::promoted_numeric_type(left, right)
            .ok_or_else(|| format!("unexpected types for ^: {:?} and {:?}", left, right)),
    }
}

fn apply_call_groups_to_type(
    start_type: &Type,
    argument_groups: &[Vec<Node>],
    monomorphizer: &Monomorphizer,
) -> Result<Type, String> {
    let mut current = start_type.clone();
    for args in argument_groups {
        current = apply_single_call_to_type(&current, args.len())?;
    }
    Ok(current)
}

fn apply_call_groups_to_function_signature(
    signature_type: &Type,
    argument_groups: &[Vec<Type>],
    metadata: &Metadata,
) -> Result<Type, String> {
    let mut current = signature_type.clone();
    for args in argument_groups {
        current = match current {
            Type::Function {
                parameters,
                return_type,
            } => {
                if parameters.len() != args.len() {
                    return Err(format!(
                        "incorrect number of args at {}:{}; expected {}, actual {}",
                        metadata.span.line,
                        metadata.span.start,
                        parameters.len(),
                        args.len()
                    ));
                }
                *return_type
            }
            other => {
                return Err(format!(
                    "cannot call value of type `{}`",
                    ast::type_to_string(&other)
                ))
            }
        };
    }
    Ok(current)
}

fn apply_single_call_to_type(signature_type: &Type, arg_len: usize) -> Result<Type, String> {
    match signature_type {
        Type::Function {
            parameters,
            return_type,
        } => {
            if parameters.len() != arg_len {
                return Err(format!(
                    "incorrect number of args; expected {}, actual {}",
                    parameters.len(),
                    arg_len
                ));
            }
            Ok(return_type.as_ref().clone())
        }
        other => Err(format!(
            "cannot call value of type `{}`",
            ast::type_to_string(other)
        )),
    }
}

fn validate_type_alias_cycles(aliases: &HashMap<String, TypeAliasTemplate>) -> Result<(), String> {
    fn visit(
        name: &str,
        aliases: &HashMap<String, TypeAliasTemplate>,
        visited: &mut HashSet<String>,
        stack: &mut Vec<String>,
    ) -> Result<(), String> {
        if let Some(index) = stack.iter().position(|entry| entry == name) {
            let mut cycle = stack[index..].to_vec();
            cycle.push(name.to_string());
            return Err(format!(
                "cyclic type alias detected: {}",
                cycle.join(" -> ")
            ));
        }
        if visited.contains(name) {
            return Ok(());
        }
        stack.push(name.to_string());
        let alias = aliases.get(name).expect("alias dependency exists");
        let mut dependencies = HashSet::new();
        let generic_params = alias
            .generic_params
            .iter()
            .map(String::as_str)
            .collect::<HashSet<_>>();
        collect_alias_dependencies(
            &alias.target_type,
            aliases,
            &generic_params,
            &mut dependencies,
        );
        let mut dependencies = dependencies.into_iter().collect::<Vec<_>>();
        dependencies.sort();
        for dependency in dependencies {
            visit(&dependency, aliases, visited, stack)?;
        }
        stack.pop();
        visited.insert(name.to_string());
        Ok(())
    }

    let mut visited = HashSet::new();
    let mut names = aliases.keys().cloned().collect::<Vec<_>>();
    names.sort();
    for name in names {
        visit(&name, aliases, &mut visited, &mut Vec::new())?;
    }
    Ok(())
}

fn collect_alias_dependencies(
    sk_type: &Type,
    aliases: &HashMap<String, TypeAliasTemplate>,
    generic_params: &HashSet<&str>,
    output: &mut HashSet<String>,
) {
    match sk_type {
        Type::Custom(name) => {
            if !generic_params.contains(name.as_str()) && aliases.contains_key(name) {
                output.insert(name.clone());
            }
        }
        Type::GenericInstance {
            base,
            type_arguments,
        } => {
            if !generic_params.contains(base.as_str()) && aliases.contains_key(base) {
                output.insert(base.clone());
            }
            for argument in type_arguments {
                collect_alias_dependencies(argument, aliases, generic_params, output);
            }
        }
        Type::Const { inner } | Type::BindingConst { inner } => {
            collect_alias_dependencies(inner, aliases, generic_params, output)
        }
        Type::Reference { target_type, .. } | Type::Pointer { target_type } => {
            collect_alias_dependencies(target_type, aliases, generic_params, output)
        }
        Type::Array { elem_type, .. } | Type::Slice { elem_type } => {
            collect_alias_dependencies(elem_type, aliases, generic_params, output)
        }
        Type::Function {
            parameters,
            return_type,
        } => {
            for parameter in parameters {
                collect_alias_dependencies(parameter, aliases, generic_params, output);
            }
            collect_alias_dependencies(return_type, aliases, generic_params, output);
        }
        Type::Union(members) | Type::Intersection(members) => {
            for member in members {
                collect_alias_dependencies(member, aliases, generic_params, output);
            }
        }
        _ => {}
    }
}

fn type_mangle(sk_type: &Type) -> String {
    match sk_type {
        Type::Void => "void".to_string(),
        Type::Byte => "byte".to_string(),
        Type::Short => "short".to_string(),
        Type::Int => "int".to_string(),
        Type::Long => "long".to_string(),
        Type::Float => "float".to_string(),
        Type::Double => "double".to_string(),
        Type::String => "string".to_string(),
        Type::Boolean => "boolean".to_string(),
        Type::Char => "char".to_string(),
        Type::Const { inner } => format!("const_{}", type_mangle(inner)),
        Type::BindingConst { inner } => type_mangle(inner),
        Type::Reference {
            target_type,
            mutable,
        } => {
            if *mutable {
                format!("mut_ref_{}", type_mangle(target_type))
            } else {
                format!("ref_{}", type_mangle(target_type))
            }
        }
        Type::Allocator => "Allocator".to_string(),
        Type::Arena => "Arena".to_string(),
        Type::Union(members) => format!(
            "union_{}",
            members
                .iter()
                .map(type_mangle)
                .collect::<Vec<_>>()
                .join("_or_")
        ),
        Type::Intersection(members) => format!(
            "intersection_{}",
            members
                .iter()
                .map(type_mangle)
                .collect::<Vec<_>>()
                .join("_and_")
        ),
        Type::Custom(name) => sanitize_mangle(name),
        Type::GenericInstance {
            base,
            type_arguments,
        } => format!(
            "{}__{}",
            sanitize_mangle(base),
            type_arguments
                .iter()
                .map(type_mangle)
                .collect::<Vec<_>>()
                .join("__")
        ),
        Type::Array {
            elem_type,
            dimensions,
        } => format!("arr{}_{}", dimensions.len(), type_mangle(elem_type)),
        Type::Pointer { target_type } => format!("ptr_{}", type_mangle(target_type)),
        Type::Slice { elem_type } => format!("slice_{}", type_mangle(elem_type)),
        Type::Function {
            parameters,
            return_type,
        } => format!(
            "fn_{}_to_{}",
            parameters
                .iter()
                .map(type_mangle)
                .collect::<Vec<_>>()
                .join("_"),
            type_mangle(return_type)
        ),
        Type::MutSelf => "mut_self".to_string(),
        Type::SkSelf => "self".to_string(),
    }
}

fn sanitize_mangle(input: &str) -> String {
    input
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}

fn specialized_struct_name(base: &str, args: &[Type]) -> String {
    format!(
        "{}__{}",
        base,
        args.iter().map(type_mangle).collect::<Vec<_>>().join("__")
    )
}

fn specialized_function_name(
    base: &str,
    substitutions: &HashMap<String, Type>,
    generic_params: &[String],
) -> String {
    let args = generic_params
        .iter()
        .map(|param| {
            substitutions
                .get(param)
                .expect("generic substitution exists")
        })
        .map(type_mangle)
        .collect::<Vec<_>>();
    format!("{}__{}", base, args.join("__"))
}
