use crate::ast::{self, Literal, Metadata, Node, Operator, Type, UnaryOperator};
use std::collections::{HashMap, HashSet};
use std::ops::Deref;

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
}

#[derive(Clone)]
struct TraitTemplate {
    name: String,
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
    trait_names: Vec<String>,
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

/// Expands generics and synthesizes any derived program structure needed before
/// type checking and code generation.
pub fn prepare_program(node: &Node) -> Result<Node, String> {
    let Node::Program { statements } = node else {
        return Ok(node.clone());
    };
    let mut monomorphizer = Monomorphizer::new(statements)?;
    monomorphizer.prepare()
}

struct Monomorphizer {
    generic_functions: HashMap<String, FunctionTemplate>,
    concrete_functions: HashMap<String, FunctionTemplate>,
    generic_structs: HashMap<String, StructTemplate>,
    concrete_structs: HashMap<String, StructTemplate>,
    generic_enums: HashMap<String, EnumTemplate>,
    concrete_enums: HashMap<String, EnumTemplate>,
    traits: HashMap<String, TraitTemplate>,
    shapes: HashMap<String, ShapeTemplate>,
    type_aliases: HashMap<String, TypeAliasTemplate>,
    impls: Vec<ImplTemplate>,
    implemented_traits: HashMap<String, HashSet<String>>,
    root_traits: Vec<Node>,
    root_concrete_impls: Vec<Node>,
    generated_impl_keys: HashSet<String>,
    generated_impls: Vec<Node>,
    generated_functions: HashMap<String, Node>,
    generated_function_order: Vec<String>,
    generated_structs: HashMap<String, Node>,
    generated_struct_order: Vec<String>,
    generated_enums: HashMap<String, Node>,
    generated_enum_order: Vec<String>,
    function_stack: HashSet<String>,
    struct_stack: HashSet<String>,
    enum_stack: HashSet<String>,
    root_statements: Vec<Node>,
}

impl Monomorphizer {
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
                } => {
                    generic_enums.insert(
                        name.clone(),
                        EnumTemplate {
                            name: name.clone(),
                            generic_params: generic_params.clone(),
                            generic_bounds: generic_bounds.clone(),
                            subtype_bounds: subtype_bounds.clone(),
                            variants: variants.clone(),
                        },
                    );
                }
                Node::EnumDeclaration { name, variants } => {
                    concrete_enums.insert(
                        name.clone(),
                        EnumTemplate {
                            name: name.clone(),
                            generic_params: Vec::new(),
                            generic_bounds: HashMap::new(),
                            subtype_bounds: HashMap::new(),
                            variants: variants.clone(),
                        },
                    );
                    root_statements.push(statement.clone());
                }
                Node::TraitDeclaration {
                    name,
                    supertraits,
                    methods,
                } => {
                    traits.insert(
                        name.clone(),
                        TraitTemplate {
                            name: name.clone(),
                            supertraits: supertraits.clone(),
                            methods: methods.clone(),
                        },
                    );
                    root_traits.push(statement.clone());
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
                    trait_names,
                    target_type,
                } => {
                    impls.push(ImplTemplate {
                        generic_params: generic_params.clone(),
                        generic_bounds: generic_bounds.clone(),
                        subtype_bounds: subtype_bounds.clone(),
                        trait_names: trait_names.clone(),
                        target_type: target_type.clone(),
                    });
                    if generic_params.is_empty() {
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
            generated_functions: HashMap::new(),
            generated_function_order: Vec::new(),
            generated_structs: HashMap::new(),
            generated_struct_order: Vec::new(),
            generated_enums: HashMap::new(),
            generated_enum_order: Vec::new(),
            function_stack: HashSet::new(),
            struct_stack: HashSet::new(),
            enum_stack: HashSet::new(),
            root_statements,
        })
    }

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
                Node::EnumDeclaration { name, variants } => {
                    self.ensure_runtime_impls_for_type(
                        &Type::Custom(name.clone()),
                        &Type::Custom(name.clone()),
                    )?;
                    output.push(self.transform_enum_decl(&name, &variants, &HashMap::new())?);
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

        for name in self.generated_struct_order.clone() {
            if let Some(node) = self.generated_structs.get(&name) {
                output.push(node.clone());
            }
        }
        for name in self.generated_enum_order.clone() {
            if let Some(node) = self.generated_enums.get(&name) {
                output.push(node.clone());
            }
        }
        output.extend(self.generated_impls.clone());
        for name in self.generated_function_order.clone() {
            if let Some(node) = self.generated_functions.get(&name) {
                output.push(node.clone());
            }
        }
        output.push(Node::EOI);
        Ok(Node::Program { statements: output })
    }

    fn expand_type(&mut self, sk_type: &Type) -> Result<Type, String> {
        self.expand_type_inner(sk_type, &mut Vec::new())
    }

    fn expand_type_inner(
        &mut self,
        sk_type: &Type,
        stack: &mut Vec<String>,
    ) -> Result<Type, String> {
        match sk_type {
            Type::Custom(name) if self.type_aliases.contains_key(name) => {
                let template = self.type_aliases.get(name).cloned().unwrap();
                if !template.generic_params.is_empty() {
                    return Err(format!(
                        "generic type alias `{}` expects {} type arguments",
                        name,
                        template.generic_params.len()
                    ));
                }
                if stack.iter().any(|entry| entry == name) {
                    stack.push(name.clone());
                    return Err(format!(
                        "cyclic type alias detected: {}",
                        stack.join(" -> ")
                    ));
                }
                stack.push(name.clone());
                let result = self.expand_type_inner(&template.target_type, stack);
                stack.pop();
                result
            }
            Type::GenericInstance {
                base,
                type_arguments,
            } if self.type_aliases.contains_key(base) => {
                let template = self.type_aliases.get(base).cloned().unwrap();
                if template.generic_params.len() != type_arguments.len() {
                    return Err(format!(
                        "generic type alias `{}` expects {} type arguments, got {}",
                        base,
                        template.generic_params.len(),
                        type_arguments.len()
                    ));
                }
                let expanded_arguments = type_arguments
                    .iter()
                    .map(|argument| self.expand_type_inner(argument, stack))
                    .collect::<Result<Vec<_>, _>>()?;
                let substitutions = template
                    .generic_params
                    .iter()
                    .cloned()
                    .zip(expanded_arguments)
                    .collect::<HashMap<_, _>>();
                self.check_generic_bounds(
                    &template.generic_bounds,
                    &template.subtype_bounds,
                    &substitutions,
                    &format!("type alias `{}`", template.name),
                )?;
                if stack.iter().any(|entry| entry == base) {
                    stack.push(base.clone());
                    return Err(format!(
                        "cyclic type alias detected: {}",
                        stack.join(" -> ")
                    ));
                }
                let target = self.apply_substitutions(&template.target_type, &substitutions);
                stack.push(base.clone());
                let result = self.expand_type_inner(&target, stack);
                stack.pop();
                result
            }
            Type::Const { inner } => Ok(Type::Const {
                inner: Box::new(self.expand_type_inner(inner, stack)?),
            }),
            Type::BindingConst { inner } => Ok(Type::BindingConst {
                inner: Box::new(self.expand_type_inner(inner, stack)?),
            }),
            Type::Reference {
                target_type,
                mutable,
            } => Ok(Type::Reference {
                target_type: Box::new(self.expand_type_inner(target_type, stack)?),
                mutable: *mutable,
            }),
            Type::Pointer { target_type } => Ok(Type::Pointer {
                target_type: Box::new(self.expand_type_inner(target_type, stack)?),
            }),
            Type::Array {
                elem_type,
                dimensions,
            } => Ok(Type::Array {
                elem_type: Box::new(self.expand_type_inner(elem_type, stack)?),
                dimensions: dimensions.clone(),
            }),
            Type::Slice { elem_type } => Ok(Type::Slice {
                elem_type: Box::new(self.expand_type_inner(elem_type, stack)?),
            }),
            Type::GenericInstance {
                base,
                type_arguments,
            } => Ok(Type::GenericInstance {
                base: base.clone(),
                type_arguments: type_arguments
                    .iter()
                    .map(|argument| self.expand_type_inner(argument, stack))
                    .collect::<Result<Vec<_>, _>>()?,
            }),
            Type::Function {
                parameters,
                return_type,
            } => Ok(Type::Function {
                parameters: parameters
                    .iter()
                    .map(|parameter| self.expand_type_inner(parameter, stack))
                    .collect::<Result<Vec<_>, _>>()?,
                return_type: Box::new(self.expand_type_inner(return_type, stack)?),
            }),
            Type::Union(members) => {
                let mut flattened = Vec::new();
                for member in members {
                    match self.expand_type_inner(member, stack)? {
                        Type::Union(nested) => flattened.extend(nested),
                        Type::Void => {
                            return Err("union types cannot contain `void`".to_string())
                        }
                        member => flattened.push(member),
                    }
                }
                flattened.sort_by_key(ast::type_to_string);
                flattened.dedup();
                if flattened.len() < 2 {
                    return Err("a union type requires at least two distinct members".to_string());
                }
                Ok(Type::Union(flattened))
            }
            Type::Intersection(members) => {
                let mut flattened = Vec::new();
                for member in members {
                    match self.expand_type_inner(member, stack)? {
                        Type::Intersection(nested) => flattened.extend(nested),
                        member => flattened.push(member),
                    }
                }
                flattened.sort_by_key(ast::type_to_string);
                flattened.dedup();
                if flattened.len() < 2 {
                    return Err(
                        "an intersection type requires at least two distinct traits".to_string()
                    );
                }
                for member in &flattened {
                    let Type::Custom(name) = member else {
                        return Err(format!(
                            "intersection member `{}` is not a trait",
                            ast::type_to_string(member)
                        ));
                    };
                    if !self.traits.contains_key(name) {
                        return Err(format!("intersection member `{}` is not a trait", name));
                    }
                }
                Ok(Type::Intersection(flattened))
            }
            other => Ok(other.clone()),
        }
    }

    fn validate_traits(&self) -> Result<(), String> {
        for trait_name in self.traits.keys() {
            let mut visiting = Vec::new();
            let _ = self.collect_trait_methods(trait_name, &mut visiting)?;
        }
        Ok(())
    }

    fn collect_trait_methods(
        &self,
        trait_name: &str,
        visiting: &mut Vec<String>,
    ) -> Result<Vec<ast::TraitMethodSignature>, String> {
        if visiting.iter().any(|name| name == trait_name) {
            visiting.push(trait_name.to_string());
            return Err(format!(
                "cyclic supertrait relationship detected: {}",
                visiting.join(" -> ")
            ));
        }
        let trait_template = self
            .traits
            .get(trait_name)
            .ok_or_else(|| format!("unknown trait `{}`", trait_name))?;
        visiting.push(trait_name.to_string());
        let mut methods = Vec::new();
        let mut seen = HashSet::new();
        for supertrait in &trait_template.supertraits {
            for method in self.collect_trait_methods(supertrait, visiting)? {
                if seen.insert(method.name.clone()) {
                    methods.push(method);
                }
            }
        }
        for method in &trait_template.methods {
            if !seen.insert(method.name.clone()) {
                return Err(format!(
                    "trait `{}` declares duplicate inherited method `{}`",
                    trait_name, method.name
                ));
            }
            methods.push(method.clone());
        }
        visiting.pop();
        Ok(methods)
    }

    fn collect_trait_ancestors(
        &self,
        trait_name: &str,
        visiting: &mut Vec<String>,
    ) -> Result<Vec<String>, String> {
        if visiting.iter().any(|name| name == trait_name) {
            visiting.push(trait_name.to_string());
            return Err(format!(
                "cyclic supertrait relationship detected: {}",
                visiting.join(" -> ")
            ));
        }
        let trait_template = self
            .traits
            .get(trait_name)
            .ok_or_else(|| format!("unknown trait `{}`", trait_name))?;
        visiting.push(trait_name.to_string());
        let mut ancestors = Vec::new();
        let mut seen = HashSet::new();
        for supertrait in &trait_template.supertraits {
            if seen.insert(supertrait.clone()) {
                ancestors.push(supertrait.clone());
            }
            for ancestor in self.collect_trait_ancestors(supertrait, visiting)? {
                if seen.insert(ancestor.clone()) {
                    ancestors.push(ancestor);
                }
            }
        }
        visiting.pop();
        Ok(ancestors)
    }

    fn trait_extends(&self, child: &str, ancestor: &str) -> Result<bool, String> {
        if child == ancestor {
            return Ok(true);
        }
        Ok(self
            .collect_trait_ancestors(child, &mut Vec::new())?
            .iter()
            .any(|name| name == ancestor))
    }

    fn implied_trait_names(&self, trait_name: &str) -> Result<Vec<String>, String> {
        let mut names = vec![trait_name.to_string()];
        names.extend(self.collect_trait_ancestors(trait_name, &mut Vec::new())?);
        Ok(names)
    }

    fn validate_impls(&mut self) -> Result<(), String> {
        let impls = self.impls.clone();
        for impl_block in impls {
            self.validate_impl_target_type(&impl_block.target_type, &impl_block.generic_params)?;
            let target_key = ast::type_to_string(&impl_block.target_type);
            for trait_name in impl_block.trait_names {
                let trait_template = self
                    .traits
                    .get(&trait_name)
                    .cloned()
                    .ok_or_else(|| format!("unknown trait `{}`", trait_name))?;
                self.validate_trait_implementation(&trait_template, &impl_block.target_type)?;
                if impl_block.generic_params.is_empty() {
                    let implied_traits =
                        self.collect_trait_ancestors(&trait_name, &mut Vec::new())?;
                    let implemented = self
                        .implemented_traits
                        .entry(target_key.clone())
                        .or_default();
                    if !implemented.insert(trait_name.clone()) {
                        return Err(format!(
                            "duplicate impl of trait `{}` for `{}`",
                            trait_name, target_key
                        ));
                    }
                    self.generated_impl_keys
                        .insert(format!("{}=>{}", trait_name, target_key));
                    for implied_trait in implied_traits {
                        implemented.insert(implied_trait.clone());
                        let implied_key = format!("{}=>{}", implied_trait, target_key);
                        if self.generated_impl_keys.insert(implied_key) {
                                self.generated_impls.push(Node::ImplDeclaration {
                                    generic_params: Vec::new(),
                                    generic_bounds: HashMap::new(),
                                    subtype_bounds: HashMap::new(),
                                trait_names: vec![implied_trait],
                                target_type: impl_block.target_type.clone(),
                            });
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn validate_impl_target_type(
        &self,
        sk_type: &Type,
        generic_params: &[String],
    ) -> Result<(), String> {
        let sk_type = ast::unwrap_binding_const(sk_type);
        match sk_type {
            Type::BindingConst { inner } => self.validate_impl_target_type(inner, generic_params),
            Type::Const { inner } => self.validate_impl_target_type(inner, generic_params),
            Type::Void => Err("cannot implement traits for `void`".to_string()),
            Type::Byte
            | Type::Short
            | Type::Int
            | Type::Long
            | Type::Float
            | Type::Double
            | Type::String
            | Type::Boolean
            | Type::Char
            | Type::Allocator
            | Type::Arena => Ok(()),
            Type::Custom(name) if generic_params.iter().any(|param| param == name) => Ok(()),
            Type::Custom(name) => {
                if self.concrete_structs.contains_key(name)
                    || self.concrete_enums.contains_key(name)
                {
                    Ok(())
                } else if self.generic_structs.contains_key(name)
                    || self.generic_enums.contains_key(name)
                {
                    Err(format!(
                        "impl targets must be concrete types; generic type `{}` needs concrete type arguments",
                        name
                    ))
                } else {
                    Err(format!("unknown impl target type `{}`", name))
                }
            }
            Type::Array { elem_type, .. } => self.validate_impl_target_type(elem_type, generic_params),
            Type::Reference { target_type, .. } => {
                self.validate_impl_target_type(target_type, generic_params)
            }
            Type::Pointer { target_type } => {
                self.validate_impl_target_type(target_type, generic_params)
            }
            Type::Slice { elem_type } => self.validate_impl_target_type(elem_type, generic_params),
            Type::Function {
                parameters,
                return_type,
            } => {
                for parameter in parameters {
                    self.validate_impl_target_type(parameter, generic_params)?;
                }
                self.validate_impl_target_type(return_type, generic_params)
            }
            Type::GenericInstance {
                base,
                type_arguments,
            } => {
                if !(self.generic_structs.contains_key(base) || self.generic_enums.contains_key(base))
                {
                    return Err(format!("unknown generic nominal type `{}`", base));
                }
                for type_argument in type_arguments {
                    self.validate_impl_target_type(type_argument, generic_params)?;
                }
                Ok(())
            }
            Type::Union(_) | Type::Intersection(_) => {
                Err("impl targets cannot be union or intersection types".to_string())
            }
            Type::SkSelf | Type::MutSelf => Err("`self` is not a valid impl target".to_string()),
        }
    }

    fn validate_trait_implementation(
        &mut self,
        trait_template: &TraitTemplate,
        target_type: &Type,
    ) -> Result<(), String> {
        for method in self.collect_trait_methods(&trait_template.name, &mut Vec::new())? {
            let Some((_, expected_receiver_type)) = method.parameters.first() else {
                return Err(format!(
                    "trait `{}` method `{}` must declare `self` as its first parameter",
                    trait_template.name, method.name
                ));
            };
            if !ast::is_self_type(expected_receiver_type) {
                return Err(format!(
                    "trait `{}` method `{}` must declare `self` as its first parameter",
                    trait_template.name, method.name
                ));
            }
            let (actual_receiver_type, actual_parameters, actual_return_type) =
                match self.lookup_method_signature(target_type, &method.name) {
                    Ok(signature) => signature,
                    Err(_) => {
                        if method.default_body.is_some() {
                            self.synthesize_trait_default_method(
                                trait_template,
                                target_type,
                                &method,
                            )?;
                            self.lookup_method_signature(target_type, &method.name).map_err(
                                |_| {
                                    format!(
                                        "type `{}` does not implement required trait method `{}.{}`",
                                        ast::type_to_string(target_type),
                                        trait_template.name,
                                        method.name
                                    )
                                },
                            )?
                        } else {
                            return Err(format!(
                                "type `{}` does not implement required trait method `{}.{}`",
                                ast::type_to_string(target_type),
                                trait_template.name,
                                method.name
                            ));
                        }
                    }
                };
            let expected_parameters = method
                .parameters
                .iter()
                .skip(1)
                .map(|(_, sk_type)| sk_type.clone())
                .collect::<Vec<_>>();
            if ast::is_mut_self_type(expected_receiver_type)
                != ast::is_mut_self_type(&actual_receiver_type)
                || actual_parameters != expected_parameters
                || actual_return_type != method.return_type
            {
                return Err(format!(
                    "trait method `{}.{}` expects `({}) -> {}`, but `{}` provides `({}) -> {}`",
                    trait_template.name,
                    method.name,
                    expected_parameters
                        .iter()
                        .map(ast::type_to_string)
                        .collect::<Vec<_>>()
                        .join(", "),
                    ast::type_to_string(&method.return_type),
                    ast::type_to_string(target_type),
                    actual_parameters
                        .iter()
                        .map(ast::type_to_string)
                        .collect::<Vec<_>>()
                        .join(", "),
                    ast::type_to_string(&actual_return_type)
                ));
            }
        }
        Ok(())
    }

    fn synthesize_trait_default_method(
        &mut self,
        trait_template: &TraitTemplate,
        target_type: &Type,
        method: &ast::TraitMethodSignature,
    ) -> Result<(), String> {
        let Some(default_body) = method.default_body.clone() else {
            return Err(format!(
                "trait `{}` method `{}` has no default body to synthesize",
                trait_template.name, method.name
            ));
        };
        let function = Node::FunctionDeclaration {
            name: method.name.clone(),
            parameters: method.parameters.clone(),
            return_type: method.return_type.clone(),
            body: default_body,
            lambda: false,
        };

        let target_type = ast::unwrap_binding_const(target_type);
        match target_type {
            Type::Custom(name) => {
                let template = self.concrete_structs.get_mut(name).ok_or_else(|| {
                    format!(
                        "trait default methods currently require struct targets, found `{}`",
                        ast::type_to_string(target_type)
                    )
                })?;
                if !template.functions.iter().any(|candidate| {
                    matches!(
                        candidate,
                        Node::FunctionDeclaration { name: candidate_name, .. }
                            if candidate_name == &method.name
                    )
                }) {
                    template.functions.push(function);
                }
            }
            Type::GenericInstance { base, .. } => {
                let template = self.generic_structs.get_mut(base).ok_or_else(|| {
                    format!(
                        "trait default methods currently require struct targets, found `{}`",
                        ast::type_to_string(target_type)
                    )
                })?;
                if !template.functions.iter().any(|candidate| {
                    matches!(
                        candidate,
                        Node::FunctionDeclaration { name: candidate_name, .. }
                            if candidate_name == &method.name
                    )
                }) {
                    template.functions.push(function);
                }
            }
            _ => {
                return Err(format!(
                    "trait default methods currently require struct targets, found `{}`",
                    ast::type_to_string(target_type)
                ));
            }
        }
        Ok(())
    }

    fn check_trait_bounds(
        &mut self,
        generic_bounds: &HashMap<String, Vec<String>>,
        substitutions: &HashMap<String, Type>,
        context: &str,
    ) -> Result<(), String> {
        for (param, constraint_names) in generic_bounds {
            let actual_type = substitutions
                .get(param)
                .ok_or_else(|| format!("missing type argument `{}` for {}", param, context))?;
            for constraint_name in constraint_names {
                match self.constraint_kind(constraint_name) {
                    Some(ConstraintKind::Trait) => {
                        let implemented = self.type_implements_trait(actual_type, constraint_name)?;
                        if !implemented {
                            return Err(format!(
                                "{} requires `{}` to implement trait `{}`, but `{}` does not",
                                context,
                                param,
                                constraint_name,
                                ast::type_to_string(actual_type)
                            ));
                        }
                    }
                    Some(ConstraintKind::Shape) => {
                        let satisfied = self.type_satisfies_shape(actual_type, constraint_name)?;
                        if !satisfied {
                            return Err(format!(
                                "{} requires `{}` to satisfy shape `{}`, but `{}` does not",
                                context,
                                param,
                                constraint_name,
                                ast::type_to_string(actual_type)
                            ));
                        }
                    }
                    None => {
                        return Err(format!(
                            "unknown trait or shape `{}` referenced by {}",
                            constraint_name, context
                        ))
                    }
                }
            }
        }
        Ok(())
    }

    fn check_generic_bounds(
        &mut self,
        generic_bounds: &HashMap<String, Vec<String>>,
        subtype_bounds: &HashMap<String, ast::SubtypeBounds>,
        substitutions: &HashMap<String, Type>,
        context: &str,
    ) -> Result<(), String> {
        self.check_trait_bounds(generic_bounds, substitutions, context)?;
        for (param, bounds) in subtype_bounds {
            let actual = substitutions
                .get(param)
                .ok_or_else(|| format!("missing type argument `{}` for {}", param, context))?;
            let actual = self.expand_type(actual)?;
            if let Some(lower) = &bounds.lower {
                let lower = self.apply_substitutions(lower, substitutions);
                let lower = self.expand_type(&lower)?;
                if !self.is_subtype(&lower, &actual)? {
                    return Err(format!(
                        "{} requires `{}` to be a supertype of `{}`, but found `{}`",
                        context,
                        param,
                        ast::type_to_string(&lower),
                        ast::type_to_string(&actual)
                    ));
                }
            }
            if let Some(upper) = &bounds.upper {
                let upper = self.apply_substitutions(upper, substitutions);
                let upper = self.expand_type(&upper)?;
                if !self.is_subtype(&actual, &upper)? {
                    return Err(format!(
                        "{} requires `{}` to be a subtype of `{}`, but found `{}`",
                        context,
                        param,
                        ast::type_to_string(&upper),
                        ast::type_to_string(&actual)
                    ));
                }
            }
        }
        Ok(())
    }

    fn is_subtype(&mut self, subtype: &Type, supertype: &Type) -> Result<bool, String> {
        let subtype = ast::strip_binding_const(subtype);
        let supertype = ast::strip_binding_const(supertype);
        let subtype = ast::unwrap_const_view(&subtype);
        let supertype = ast::unwrap_const_view(&supertype);

        if subtype == supertype {
            return Ok(true);
        }
        match (subtype, supertype) {
            (Type::Intersection(children), Type::Intersection(parents)) => {
                for parent in parents {
                    let mut implied = false;
                    for child in children {
                        if self.is_subtype(child, parent)? {
                            implied = true;
                            break;
                        }
                    }
                    if !implied {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            (Type::Union(members), supertype) => {
                for member in members {
                    if !self.is_subtype(member, supertype)? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            (subtype, Type::Union(members)) => {
                for member in members {
                    if self.is_subtype(subtype, member)? {
                        return Ok(true);
                    }
                }
                Ok(false)
            }
            (Type::Intersection(members), supertype) => {
                for member in members {
                    if self.is_subtype(member, supertype)? {
                        return Ok(true);
                    }
                }
                Ok(false)
            }
            (subtype, Type::Intersection(members)) => {
                for member in members {
                    if !self.is_subtype(subtype, member)? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            (Type::Custom(child), Type::Custom(parent))
                if self.traits.contains_key(child) && self.traits.contains_key(parent) =>
            {
                self.trait_extends(child, parent)
            }
            (subtype, Type::Custom(trait_name)) if self.traits.contains_key(trait_name) => {
                self.type_implements_trait(subtype, trait_name)
            }
            (
                Type::Reference {
                    target_type: child,
                    mutable: false,
                },
                Type::Reference {
                    target_type: parent,
                    mutable: false,
                },
            ) => self.is_subtype(child, parent),
            (
                Type::Reference {
                    target_type: child,
                    mutable: true,
                },
                Type::Reference {
                    target_type: parent,
                    mutable: true,
                },
            ) => Ok(child == parent),
            _ => Ok(false),
        }
    }

    fn constraint_kind(&self, name: &str) -> Option<ConstraintKind> {
        if self.traits.contains_key(name) {
            Some(ConstraintKind::Trait)
        } else if self.shapes.contains_key(name) {
            Some(ConstraintKind::Shape)
        } else {
            None
        }
    }

    fn type_implements_trait(&mut self, actual_type: &Type, trait_name: &str) -> Result<bool, String> {
        let actual_key = ast::type_to_string(actual_type);
        if self
            .implemented_traits
            .get(&actual_key)
            .is_some_and(|traits| traits.contains(trait_name))
        {
            return Ok(true);
        }

        let mut matched = false;
        for impl_block in self.impls.clone() {
            if !impl_block
                .trait_names
                .iter()
                .any(|name| self.trait_extends(name, trait_name).unwrap_or(false))
            {
                continue;
            }
            if impl_block.generic_params.is_empty() {
                continue;
            }
            let mut substitutions = HashMap::new();
            if self
                .unify_generic_type(
                    &impl_block.target_type,
                    actual_type,
                    &impl_block.generic_params,
                    &mut substitutions,
                )
                .is_err()
            {
                continue;
            }
            self.check_generic_bounds(
                &impl_block.generic_bounds,
                &impl_block.subtype_bounds,
                &substitutions,
                &format!(
                    "generic impl of trait `{}` for `{}`",
                    trait_name,
                    ast::type_to_string(&impl_block.target_type)
                ),
            )?;
            if matched {
                return Err(format!(
                    "multiple impls of trait `{}` match `{}`",
                    trait_name, actual_key
                ));
            }
            matched = true;
        }

        Ok(matched)
    }

    fn type_satisfies_shape(&mut self, actual_type: &Type, shape_name: &str) -> Result<bool, String> {
        let shape = self
            .shapes
            .get(shape_name)
            .ok_or_else(|| format!("unknown shape `{}`", shape_name))?
            .clone();
        self.validate_shape_satisfaction(&shape, actual_type)
            .map(|_| true)
    }

    fn validate_shape_satisfaction(
        &mut self,
        shape: &ShapeTemplate,
        target_type: &Type,
    ) -> Result<(), String> {
        for method in &shape.methods {
            let Some((_, expected_receiver_type)) = method.parameters.first() else {
                return Err(format!(
                    "shape `{}` method `{}` must declare `self` as its first parameter",
                    shape.name, method.name
                ));
            };
            if !ast::is_self_type(expected_receiver_type) {
                return Err(format!(
                    "shape `{}` method `{}` must declare `self` as its first parameter",
                    shape.name, method.name
                ));
            }
            let (actual_receiver_type, actual_parameters, actual_return_type) = self
                .lookup_method_signature(target_type, &method.name)
                .map_err(|_| {
                    format!(
                        "type `{}` does not satisfy required shape method `{}.{}`",
                        ast::type_to_string(target_type),
                        shape.name,
                        method.name
                    )
                })?;
            let expected_parameters = method
                .parameters
                .iter()
                .skip(1)
                .map(|(_, sk_type)| sk_type.clone())
                .collect::<Vec<_>>();
            if ast::is_mut_self_type(expected_receiver_type)
                != ast::is_mut_self_type(&actual_receiver_type)
                || actual_parameters != expected_parameters
                || actual_return_type != method.return_type
            {
                return Err(format!(
                    "shape method `{}.{}` expects `({}) -> {}`, but `{}` provides `({}) -> {}`",
                    shape.name,
                    method.name,
                    expected_parameters
                        .iter()
                        .map(ast::type_to_string)
                        .collect::<Vec<_>>()
                        .join(", "),
                    ast::type_to_string(&method.return_type),
                    ast::type_to_string(target_type),
                    actual_parameters
                        .iter()
                        .map(ast::type_to_string)
                        .collect::<Vec<_>>()
                        .join(", "),
                    ast::type_to_string(&actual_return_type)
                ));
            }
        }
        Ok(())
    }

    fn ensure_runtime_impls_for_type(
        &mut self,
        source_type: &Type,
        concrete_type: &Type,
    ) -> Result<(), String> {
        let concrete_key = ast::type_to_string(concrete_type);
        for impl_block in self.impls.clone() {
            let mut substitutions = HashMap::new();
            if impl_block.generic_params.is_empty() {
                if impl_block.target_type != *source_type && impl_block.target_type != *concrete_type
                {
                    continue;
                }
            } else if self
                .unify_generic_type(
                    &impl_block.target_type,
                    source_type,
                    &impl_block.generic_params,
                    &mut substitutions,
                )
                .is_err()
            {
                continue;
            } else {
                self.check_generic_bounds(
                    &impl_block.generic_bounds,
                    &impl_block.subtype_bounds,
                    &substitutions,
                    &format!(
                        "generic impl target `{}`",
                        ast::type_to_string(&impl_block.target_type)
                    ),
                )?;
            }

            for trait_name in impl_block.trait_names {
                for implied_trait in self.implied_trait_names(&trait_name)? {
                    let key = format!("{}=>{}", implied_trait, concrete_key);
                    if !self.generated_impl_keys.insert(key) {
                        continue;
                    }
                    self.generated_impls.push(Node::ImplDeclaration {
                        generic_params: Vec::new(),
                        generic_bounds: HashMap::new(),
                        subtype_bounds: HashMap::new(),
                        trait_names: vec![implied_trait],
                        target_type: concrete_type.clone(),
                    });
                }
            }
        }
        Ok(())
    }

    fn transform_named_function(
        &mut self,
        name: &str,
        parameters: &[(String, Type)],
        return_type: &Type,
        body: &[Node],
        substitutions: &HashMap<String, Type>,
        self_type: Option<Type>,
    ) -> Result<Node, String> {
        let mut env = Env::new();
        let mut output_parameters = Vec::new();
        for (param_name, param_type) in parameters {
            let internal_type = if ast::is_self_type(param_type) {
                let resolved_self_type = self_type
                    .clone()
                    .ok_or_else(|| "self parameter requires a receiver type".to_string())?;
                if ast::is_mut_self_type(param_type) {
                    resolved_self_type
                } else {
                    Type::BindingConst {
                        inner: Box::new(resolved_self_type),
                    }
                }
            } else {
                let substituted = self.apply_substitutions(param_type, substitutions);
                self.expand_type(&substituted)?
            };
            env.insert(param_name.clone(), internal_type.clone());
            output_parameters.push((
                param_name.clone(),
                if ast::is_self_type(param_type) {
                    param_type.clone()
                } else {
                    self.concretize_type(&internal_type)?
                },
            ));
        }

        let substituted_return_type = self.apply_substitutions(return_type, substitutions);
        let internal_return_type = self.expand_type(&substituted_return_type)?;
        let mut output_body = Vec::new();
        for statement in body {
            let (statement, _) = self.transform_statement(
                statement,
                &mut env,
                &internal_return_type,
                substitutions,
                self_type.clone(),
            )?;
            output_body.push(statement);
        }

        Ok(Node::FunctionDeclaration {
            name: name.to_string(),
            parameters: output_parameters,
            return_type: self.concretize_type(&internal_return_type)?,
            body: output_body,
            lambda: false,
        })
    }

    fn transform_struct_decl(
        &mut self,
        name: &str,
        fields: &[(String, Type)],
        functions: &[Node],
        substitutions: &HashMap<String, Type>,
        self_type: Option<Type>,
    ) -> Result<Node, String> {
        let concrete_self_type = self_type.unwrap_or_else(|| Type::Custom(name.to_string()));
        let output_fields = fields
            .iter()
            .map(|(field_name, field_type)| {
                Ok((
                    field_name.clone(),
                    self.concretize_type(&self.apply_substitutions(field_type, substitutions))?,
                ))
            })
            .collect::<Result<Vec<_>, String>>()?;

        let mut output_functions = Vec::new();
        for function in functions {
            match function {
                Node::FunctionDeclaration {
                    name: method_name,
                    parameters,
                    return_type,
                    body,
                    lambda: false,
                } => {
                    output_functions.push(self.transform_named_function(
                        method_name,
                        parameters,
                        return_type,
                        body,
                        substitutions,
                        Some(concrete_self_type.clone()),
                    )?);
                }
                Node::GenericFunctionDeclaration { name, .. } => {
                    return Err(format!("generic methods are not supported yet: `{}`", name));
                }
                other => {
                    return Err(format!(
                        "unsupported struct member during monomorphization: `{:?}`",
                        other
                    ))
                }
            }
        }

        Ok(Node::StructDeclaration {
            name: name.to_string(),
            fields: output_fields,
            functions: output_functions,
        })
    }

    fn transform_enum_decl(
        &mut self,
        name: &str,
        variants: &[ast::EnumVariant],
        substitutions: &HashMap<String, Type>,
    ) -> Result<Node, String> {
        let output_variants = variants
            .iter()
            .map(|variant| {
                Ok(ast::EnumVariant {
                    name: variant.name.clone(),
                    payload_types: variant
                        .payload_types
                        .iter()
                        .map(|payload_type| {
                            let substituted = self.apply_substitutions(payload_type, substitutions);
                            self.concretize_type(&substituted)
                        })
                        .collect::<Result<Vec<_>, String>>()?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;

        Ok(Node::EnumDeclaration {
            name: name.to_string(),
            variants: output_variants,
        })
    }

    fn transform_statement(
        &mut self,
        node: &Node,
        env: &mut Env,
        expected_return_type: &Type,
        substitutions: &HashMap<String, Type>,
        self_type: Option<Type>,
    ) -> Result<(Node, Option<Type>), String> {
        match node {
            Node::VariableDeclaration {
                var_type,
                name,
                value,
                metadata,
            } => {
                let substituted = self.apply_substitutions(var_type, substitutions);
                let internal_type = self.expand_type(&substituted)?;
                let output_type = self.concretize_type(&internal_type)?;
                let is_recursive_lambda = matches!(
                    value.as_deref(),
                    Some(Node::FunctionDeclaration { lambda: true, .. })
                );
                if is_recursive_lambda {
                    env.insert(name.clone(), internal_type.clone());
                }
                let value = if let Some(value) = value {
                    let (value, _) = self.transform_expr(
                        value,
                        env,
                        Some(&internal_type),
                        substitutions,
                        self_type.clone(),
                    )?;
                    Some(Box::new(value))
                } else {
                    None
                };
                if !is_recursive_lambda {
                    env.insert(name.clone(), internal_type.clone());
                }
                Ok((
                    Node::VariableDeclaration {
                        var_type: output_type,
                        name: name.clone(),
                        value,
                        metadata: metadata.clone(),
                    },
                    Some(internal_type),
                ))
            }
            Node::StructDestructure {
                struct_type,
                fields,
                value,
                metadata,
            } => {
                let substituted = self.apply_substitutions(struct_type, substitutions);
                let internal_type = self.expand_type(&substituted)?;
                let output_type = self.concretize_type(&internal_type)?;
                let (value, _) = self.transform_expr(
                    value,
                    env,
                    Some(&internal_type),
                    substitutions,
                    self_type.clone(),
                )?;
                for field in fields {
                    let field_type = self
                        .lookup_struct_field_type(&internal_type, &field.field_name)?
                        .ok_or_else(|| {
                            format!(
                                "unknown field `{}` on `{}`",
                                field.field_name,
                                ast::type_to_string(&internal_type)
                            )
                        })?;
                    env.insert(field.binding.clone(), field_type);
                }
                Ok((
                    Node::StructDestructure {
                        struct_type: output_type,
                        fields: fields.clone(),
                        value: Box::new(value),
                        metadata: metadata.clone(),
                    },
                    Some(Type::Void),
                ))
            }
            Node::Assignment {
                var,
                value,
                metadata,
            } => {
                let (var, var_type) =
                    self.transform_expr(var, env, None, substitutions, self_type.clone())?;
                let (value, _) = self.transform_expr(
                    value,
                    env,
                    Some(&var_type),
                    substitutions,
                    self_type.clone(),
                )?;
                Ok((
                    Node::Assignment {
                        var: Box::new(var),
                        value: Box::new(value),
                        metadata: metadata.clone(),
                    },
                    Some(var_type),
                ))
            }
            Node::Return(value) => {
                let value = if let Some(value) = value {
                    let (value, _) = self.transform_expr(
                        value,
                        env,
                        Some(expected_return_type),
                        substitutions,
                        self_type,
                    )?;
                    Some(Box::new(value))
                } else {
                    None
                };
                Ok((Node::Return(value), Some(expected_return_type.clone())))
            }
            Node::Defer(expression) => {
                let (expression, _) = self.transform_expr(
                    expression,
                    env,
                    None,
                    substitutions,
                    self_type,
                )?;
                Ok((Node::Defer(Box::new(expression)), Some(Type::Void)))
            }
            Node::Print(expr) => {
                let (expr, expr_type) =
                    self.transform_expr(expr, env, None, substitutions, self_type)?;
                Ok((Node::Print(Box::new(expr)), Some(expr_type)))
            }
            Node::Block { statements } => {
                env.push();
                let mut output = Vec::new();
                for statement in statements {
                    let (statement, _) = self.transform_statement(
                        statement,
                        env,
                        expected_return_type,
                        substitutions,
                        self_type.clone(),
                    )?;
                    output.push(statement);
                }
                env.pop();
                Ok((Node::Block { statements: output }, Some(Type::Void)))
            }
            Node::UnsafeBlock { statements } => {
                env.push();
                let mut output = Vec::new();
                for statement in statements {
                    let (statement, _) = self.transform_statement(
                        statement,
                        env,
                        expected_return_type,
                        substitutions,
                        self_type.clone(),
                    )?;
                    output.push(statement);
                }
                env.pop();
                Ok((Node::UnsafeBlock { statements: output }, Some(Type::Void)))
            }
            Node::If {
                condition,
                body,
                else_if_blocks,
                else_block,
            } => {
                let (condition, _) = self.transform_expr(
                    condition,
                    env,
                    Some(&Type::Boolean),
                    substitutions,
                    self_type.clone(),
                )?;
                env.push();
                let mut output_body = Vec::new();
                for statement in body {
                    let (statement, _) = self.transform_statement(
                        statement,
                        env,
                        expected_return_type,
                        substitutions,
                        self_type.clone(),
                    )?;
                    output_body.push(statement);
                }
                env.pop();

                let mut output_else_if_blocks = Vec::new();
                for block in else_if_blocks {
                    let (block, _) = self.transform_statement(
                        block,
                        env,
                        expected_return_type,
                        substitutions,
                        self_type.clone(),
                    )?;
                    output_else_if_blocks.push(block);
                }

                let output_else_block = if let Some(else_block) = else_block {
                    env.push();
                    let mut output = Vec::new();
                    for statement in else_block {
                        let (statement, _) = self.transform_statement(
                            statement,
                            env,
                            expected_return_type,
                            substitutions,
                            self_type.clone(),
                        )?;
                        output.push(statement);
                    }
                    env.pop();
                    Some(output)
                } else {
                    None
                };

                Ok((
                    Node::If {
                        condition: Box::new(condition),
                        body: output_body,
                        else_if_blocks: output_else_if_blocks,
                        else_block: output_else_block,
                    },
                    Some(Type::Void),
                ))
            }
            Node::Match { value, cases } => {
                let (value, value_type) =
                    self.transform_expr(value, env, None, substitutions, self_type.clone())?;
                let mut output_cases = Vec::new();
                for case in cases {
                    env.push();
                    match &case.pattern {
                        ast::MatchPattern::EnumVariant { bindings, .. } => {
                            let payload_types =
                                self.lookup_enum_variant_payload_types(&value_type, &case.pattern)?;
                            if bindings.len() != payload_types.len() {
                                return Err(format!(
                                    "match pattern binding count does not match enum payload arity for `{}`",
                                    ast::type_to_string(&value_type)
                                ));
                            }
                            for (binding, payload_type) in
                                bindings.iter().cloned().zip(payload_types.into_iter())
                            {
                                env.insert(binding, payload_type);
                            }
                        }
                        ast::MatchPattern::Struct { .. } => {
                            for (binding, binding_type) in
                                self.lookup_struct_pattern_bindings(&value_type, &case.pattern)?
                            {
                                env.insert(binding, binding_type);
                            }
                        }
                    }
                    let mut output_body = Vec::new();
                    for statement in &case.body {
                        let (statement, _) = self.transform_statement(
                            statement,
                            env,
                            expected_return_type,
                            substitutions,
                            self_type.clone(),
                        )?;
                        output_body.push(statement);
                    }
                    env.pop();
                    output_cases.push(ast::MatchCase {
                        pattern: self.transform_match_pattern(
                            &case.pattern,
                            substitutions,
                            &value_type,
                        )?,
                        body: output_body,
                    });
                }
                Ok((
                    Node::Match {
                        value: Box::new(value),
                        cases: output_cases,
                    },
                    Some(Type::Void),
                ))
            }
            Node::For {
                init,
                condition,
                update,
                body,
            } => {
                env.push();
                let init = if let Some(init) = init {
                    let (init, _) = self.transform_statement(
                        init,
                        env,
                        expected_return_type,
                        substitutions,
                        self_type.clone(),
                    )?;
                    Some(Box::new(init))
                } else {
                    None
                };
                let condition = if let Some(condition) = condition {
                    let (condition, _) = self.transform_expr(
                        condition,
                        env,
                        Some(&Type::Boolean),
                        substitutions,
                        self_type.clone(),
                    )?;
                    Some(Box::new(condition))
                } else {
                    None
                };
                let update = if let Some(update) = update {
                    let (update, _) = self.transform_statement(
                        update,
                        env,
                        expected_return_type,
                        substitutions,
                        self_type.clone(),
                    )?;
                    Some(Box::new(update))
                } else {
                    None
                };
                let mut output_body = Vec::new();
                for statement in body {
                    let (statement, _) = self.transform_statement(
                        statement,
                        env,
                        expected_return_type,
                        substitutions,
                        self_type.clone(),
                    )?;
                    output_body.push(statement);
                }
                env.pop();
                Ok((
                    Node::For {
                        init,
                        condition,
                        update,
                        body: output_body,
                    },
                    Some(Type::Void),
                ))
            }
            Node::FunctionDeclaration {
                name,
                parameters,
                return_type,
                body,
                lambda: true,
            } => {
                let (node, sk_type) = self.transform_lambda(
                    name,
                    parameters,
                    return_type,
                    body,
                    env,
                    substitutions,
                    self_type,
                )?;
                Ok((node, Some(sk_type)))
            }
            Node::Export { .. } => Err("`export` is only allowed at module scope".to_string()),
            Node::TraitDeclaration { .. } => {
                Err("`trait` is only allowed at module scope".to_string())
            }
            Node::ShapeDeclaration { .. } => {
                Err("`shape` is only allowed at module scope".to_string())
            }
            Node::ImplDeclaration { .. } => {
                Err("`impl` is only allowed at module scope".to_string())
            }
            Node::FunctionCall { .. }
            | Node::Access { .. }
            | Node::ArrayInit { .. }
            | Node::StructInitialization { .. }
            | Node::StaticFunctionCall { .. }
            | Node::Dereference { .. }
            | Node::Literal(_)
            | Node::Identifier(_)
            | Node::BinaryOp { .. }
            | Node::UnaryOp { .. }
            | Node::Input => {
                let (node, sk_type) =
                    self.transform_expr(node, env, None, substitutions, self_type)?;
                Ok((node, Some(sk_type)))
            }
            other => Err(format!(
                "unsupported statement during monomorphization: `{:?}`",
                other
            )),
        }
    }

    fn transform_lambda(
        &mut self,
        name: &str,
        parameters: &[(String, Type)],
        return_type: &Type,
        body: &[Node],
        parent_env: &Env,
        substitutions: &HashMap<String, Type>,
        self_type: Option<Type>,
    ) -> Result<(Node, Type), String> {
        let mut env = parent_env.clone();
        env.push();
        let mut output_parameters = Vec::new();
        let mut function_parameters = Vec::new();
        for (param_name, param_type) in parameters {
            let internal_type = if ast::is_self_type(param_type) {
                let resolved_self_type = self_type
                    .clone()
                    .ok_or_else(|| "self parameter requires a receiver type".to_string())?;
                if ast::is_mut_self_type(param_type) {
                    resolved_self_type
                } else {
                    Type::BindingConst {
                        inner: Box::new(resolved_self_type),
                    }
                }
            } else {
                let substituted = self.apply_substitutions(param_type, substitutions);
                self.expand_type(&substituted)?
            };
            env.insert(param_name.clone(), internal_type.clone());
            function_parameters.push(ast::strip_binding_const(&internal_type));
            output_parameters.push((
                param_name.clone(),
                if ast::is_self_type(param_type) {
                    param_type.clone()
                } else {
                    self.concretize_type(&internal_type)?
                },
            ));
        }
        let substituted_return_type = self.apply_substitutions(return_type, substitutions);
        let internal_return_type = self.expand_type(&substituted_return_type)?;
        let mut output_body = Vec::new();
        for statement in body {
            let (statement, _) = self.transform_statement(
                statement,
                &mut env,
                &internal_return_type,
                substitutions,
                self_type.clone(),
            )?;
            output_body.push(statement);
        }
        Ok((
            Node::FunctionDeclaration {
                name: name.to_string(),
                parameters: output_parameters,
                return_type: self.concretize_type(&internal_return_type)?,
                body: output_body,
                lambda: true,
            },
            Type::Function {
                parameters: function_parameters,
                return_type: Box::new(internal_return_type),
            },
        ))
    }

    fn transform_expr(
        &mut self,
        node: &Node,
        env: &mut Env,
        expected_type: Option<&Type>,
        substitutions: &HashMap<String, Type>,
        self_type: Option<Type>,
    ) -> Result<(Node, Type), String> {
        match node {
            Node::Dereference { .. } => {
                Err("access steps should be nested inside `Node::Access`".to_string())
            }
            Node::Literal(literal) => {
                Ok((node.clone(), self.literal_type(literal, expected_type)?))
            }
            Node::Identifier(name) => {
                if let Some(sk_type) = env.get(name) {
                    return Ok((
                        Node::Identifier(name.clone()),
                        ast::strip_binding_const(&sk_type),
                    ));
                }
                if let Some(function) = self.concrete_functions.get(name).cloned() {
                    let signature = Type::Function {
                        parameters: function
                            .parameters
                            .iter()
                            .map(|(_, sk_type)| sk_type.clone())
                            .collect(),
                        return_type: Box::new(function.return_type.clone()),
                    };
                    return Ok((
                        Node::Identifier(name.clone()),
                        self.expand_type(&signature)?,
                    ));
                }
                if self.generic_functions.contains_key(name) {
                    return Err(format!(
                        "generic function `{}` must be called so its type arguments can be inferred",
                        name
                    ));
                }
                Ok((Node::Identifier(name.clone()), Type::Custom(name.clone())))
            }
            Node::ArrayInit { elements } => {
                let expected_item_type = expected_type.and_then(array_item_type);
                let mut output = Vec::new();
                let mut element_type = None;
                for element in elements {
                    let (element, current_type) = self.transform_expr(
                        element,
                        env,
                        expected_item_type.as_ref(),
                        substitutions,
                        self_type.clone(),
                    )?;
                    if let Some(existing) = &element_type {
                        if current_type != *existing {
                            if let Some(promoted) =
                                ast::promoted_numeric_type(existing, &current_type)
                            {
                                element_type = Some(promoted);
                            } else {
                                return Err(
                                    "array literal contains incompatible element types".to_string()
                                );
                            }
                        }
                    } else {
                        element_type = Some(current_type.clone());
                    }
                    output.push(element);
                }
                let internal_type = if let Some(expected) = expected_type {
                    expected.clone()
                } else {
                    Type::Slice {
                        elem_type: Box::new(element_type.unwrap_or(Type::Int)),
                    }
                };
                Ok((Node::ArrayInit { elements: output }, internal_type))
            }
            Node::StructInitialization { _type, fields } => {
                let substituted = self.apply_substitutions(_type, substitutions);
                let internal_type = self.expand_type(&substituted)?;
                let struct_name = self.ensure_struct_for_type(&internal_type)?;
                let mut output_fields = Vec::new();
                for (field_name, value) in fields {
                    let field_type = self
                        .lookup_struct_field_type(&internal_type, field_name)?
                        .ok_or_else(|| {
                            format!(
                                "unknown field `{}` on `{}`",
                                field_name,
                                ast::type_to_string(&internal_type)
                            )
                        })?;
                    let (value, _) = self.transform_expr(
                        value,
                        env,
                        Some(&field_type),
                        substitutions,
                        self_type.clone(),
                    )?;
                    output_fields.push((field_name.clone(), value));
                }
                Ok((
                    Node::StructInitialization {
                        _type: Type::Custom(struct_name),
                        fields: output_fields,
                    },
                    internal_type,
                ))
            }
            Node::StaticFunctionCall {
                _type,
                name,
                arguments,
                metadata,
            } => {
                let substituted = self.apply_substitutions(_type, substitutions);
                let internal_type = self.expand_type(&substituted)?;
                let output_type = self.concretize_type(&internal_type)?;
                let mut output_args = Vec::new();
                if name == "size_of" || name == "align_of" {
                    if !arguments.is_empty() {
                        return Err(format!(
                            "{} expects no arguments",
                            name
                        ));
                    }
                    return Ok((
                        Node::StaticFunctionCall {
                            _type: output_type,
                            name: name.clone(),
                            arguments: Vec::new(),
                            metadata: metadata.clone(),
                        },
                        Type::Int,
                    ));
                }
                let return_type = match &internal_type {
                    Type::Custom(system_name) if system_name == "System" && name == "allocator" => {
                        Type::Allocator
                    }
                    Type::Custom(window_name) if window_name == "Window" && name == "create" => {
                        if arguments.len() != 3 {
                            return Err("Window::create expects width, height, and title".to_string());
                        }
                        let expected_types = [
                            Type::Int,
                            Type::Int,
                            Type::String,
                        ];
                        for (argument, expected_type) in arguments.iter().zip(expected_types.iter()) {
                            let (argument, _) = self.transform_expr(
                                argument,
                                env,
                                Some(expected_type),
                                substitutions,
                                self_type.clone(),
                            )?;
                            output_args.push(argument);
                        }
                        Type::Custom("Window".to_string())
                    }
                    Type::Custom(color_name) if color_name == "Color" => {
                        match name.as_str() {
                            "black" | "white" | "red" | "green" | "blue" => {
                                if !arguments.is_empty() {
                                    return Err(format!(
                                        "Color::{} expects no arguments",
                                        name
                                    ));
                                }
                                Type::Custom("Color".to_string())
                            }
                            "rgb" | "rgba" => {
                                let expected_len = if name == "rgb" { 3 } else { 4 };
                                if arguments.len() != expected_len {
                                    return Err(format!(
                                        "Color::{} expects {} arguments",
                                        name, expected_len
                                    ));
                                }
                                for argument in arguments {
                                    let (argument, _) = self.transform_expr(
                                        argument,
                                        env,
                                        Some(&Type::Int),
                                        substitutions,
                                        self_type.clone(),
                                    )?;
                                    output_args.push(argument);
                                }
                                Type::Custom("Color".to_string())
                            }
                            _ => {
                                return Err(format!(
                                    "unsupported static call during monomorphization: `Color::{}`",
                                    name
                                ))
                            }
                        }
                    }
                    Type::Custom(keyboard_name) if keyboard_name == "Keyboard" && name == "is_down" => {
                        if arguments.len() != 2 {
                            return Err("Keyboard::is_down expects window and key".to_string());
                        }
                        let expected_types = [
                            Type::Custom("Window".to_string()),
                            Type::Char,
                        ];
                        for (argument, expected_type) in arguments.iter().zip(expected_types.iter()) {
                            let (argument, _) = self.transform_expr(
                                argument,
                                env,
                                Some(expected_type),
                                substitutions,
                                self_type.clone(),
                            )?;
                            output_args.push(argument);
                        }
                        Type::Boolean
                    }
                    Type::Custom(memory_name) if memory_name == "Memory" => {
                        match name.as_str() {
                            "copy" | "set" => {
                                for argument in arguments {
                                    let (argument, _) = self.transform_expr(
                                        argument,
                                        env,
                                        None,
                                        substitutions,
                                        self_type.clone(),
                                    )?;
                                    output_args.push(argument);
                                }
                                Type::Void
                            }
                            _ => {
                                return Err(format!(
                                    "unsupported static call during monomorphization: `{}::{}`",
                                    ast::type_to_string(&internal_type),
                                    name
                                ))
                            }
                        }
                    }
                    Type::Arena if name == "init" => {
                        for argument in arguments {
                            let (argument, _) = self.transform_expr(
                                argument,
                                env,
                                Some(&Type::Allocator),
                                substitutions,
                                self_type.clone(),
                            )?;
                            output_args.push(argument);
                        }
                        Type::Arena
                    }
                    Type::Array { elem_type, .. } if name == "fill" || name == "new" => {
                        for argument in arguments {
                            let (argument, _) = self.transform_expr(
                                argument,
                                env,
                                Some(elem_type.deref()),
                                substitutions,
                                self_type.clone(),
                            )?;
                            output_args.push(argument);
                        }
                        internal_type.clone()
                    }
                    Type::Slice { .. } if name == "alloc" => {
                        for (index, argument) in arguments.iter().enumerate() {
                            let expected_argument_type = if index == 0 {
                                Some(&Type::Allocator)
                            } else {
                                Some(&Type::Int)
                            };
                            let (argument, _) = self.transform_expr(
                                argument,
                                env,
                                expected_argument_type,
                                substitutions,
                                self_type.clone(),
                            )?;
                            output_args.push(argument);
                        }
                        internal_type.clone()
                    }
                    Type::Pointer { target_type } if name == "cast" => {
                        for argument in arguments {
                            let (argument, _) = self.transform_expr(
                                argument,
                                env,
                                None,
                                substitutions,
                                self_type.clone(),
                            )?;
                            output_args.push(argument);
                        }
                        Type::Pointer {
                            target_type: target_type.clone(),
                        }
                    }
                    Type::Pointer { target_type } if name == "offset" => {
                        for argument in arguments {
                            let (argument, _) = self.transform_expr(
                                argument,
                                env,
                                None,
                                substitutions,
                                self_type.clone(),
                            )?;
                            output_args.push(argument);
                        }
                        Type::Pointer {
                            target_type: target_type.clone(),
                        }
                    }
                    Type::Reference { .. } if name == "cast" || name == "offset" => {
                        return Err(format!(
                            "reference type `{}` does not support static method `{}`",
                            ast::type_to_string(&internal_type),
                            name
                        ));
                    }
                    Type::Custom(_) | Type::GenericInstance { .. } if name == "create" => {
                        for argument in arguments {
                            let (argument, _) = self.transform_expr(
                                argument,
                                env,
                                Some(&Type::Allocator),
                                substitutions,
                                self_type.clone(),
                            )?;
                            output_args.push(argument);
                        }
                        Type::Pointer {
                            target_type: Box::new(internal_type.clone()),
                        }
                    }
                    Type::Custom(_) | Type::GenericInstance { .. }
                        if self.lookup_enum_variant(&internal_type, name)?.is_none() =>
                    {
                        let (parameter_types, return_type) =
                            self.lookup_static_function_signature(&internal_type, name)?;
                        if arguments.len() != parameter_types.len() {
                            return Err(format!(
                                "static function `{}::{}` expects {} argument(s), got {}",
                                ast::type_to_string(&internal_type),
                                name,
                                parameter_types.len(),
                                arguments.len()
                            ));
                        }
                        for (argument, parameter_type) in arguments.iter().zip(parameter_types.iter())
                        {
                            let (argument, _) = self.transform_expr(
                                argument,
                                env,
                                Some(parameter_type),
                                substitutions,
                                self_type.clone(),
                            )?;
                            output_args.push(argument);
                        }
                        return_type
                    }
                    Type::Custom(_) | Type::GenericInstance { .. }
                        if self.lookup_enum_variant(&internal_type, name)?.is_some() =>
                    {
                        let payload_types =
                            self.lookup_enum_variant(&internal_type, name)?.unwrap_or_default();
                        if arguments.len() != payload_types.len() {
                            return Err(format!(
                                "enum variant constructor `{}::{}` expects {} argument(s), got {}",
                                ast::type_to_string(&internal_type),
                                name,
                                payload_types.len(),
                                arguments.len()
                            ));
                        }
                        for (argument, payload_type) in arguments.iter().zip(payload_types.iter()) {
                            let (argument, _) = self.transform_expr(
                                argument,
                                env,
                                Some(payload_type),
                                substitutions,
                                self_type.clone(),
                            )?;
                            output_args.push(argument);
                        }
                        internal_type.clone()
                    }
                    _ => {
                        return Err(format!(
                            "unsupported static call during monomorphization: `{}::{}`",
                            ast::type_to_string(&internal_type),
                            name
                        ))
                    }
                };
                Ok((
                    Node::StaticFunctionCall {
                        _type: output_type,
                        name: name.clone(),
                        arguments: output_args,
                        metadata: metadata.clone(),
                    },
                    return_type,
                ))
            }
            Node::FunctionCall {
                name,
                type_arguments,
                arguments,
                metadata,
            } => self.transform_function_call(
                name,
                type_arguments,
                arguments,
                metadata,
                env,
                expected_type,
                substitutions,
                self_type,
            ),
            Node::Access { nodes } => {
                let mut output_nodes = Vec::new();
                let (first, mut current_type) =
                    self.transform_expr(&nodes[0], env, None, substitutions, self_type.clone())?;
                output_nodes.push(first);

                for step in nodes.iter().skip(1) {
                    if matches!(
                        step,
                        Node::MemberAccess { .. }
                            | Node::ArrayAccess { .. }
                            | Node::SliceAccess { .. }
                    ) {
                        loop {
                            match current_type.clone() {
                                Type::Const { inner } => {
                                    current_type = inner.deref().clone();
                                }
                                Type::Reference { target_type, .. } => {
                                    current_type = target_type.deref().clone();
                                }
                                Type::Pointer { target_type } => {
                                    current_type = target_type.deref().clone();
                                }
                                _ => break,
                            }
                        }
                    }

                    match step {
                        Node::ArrayAccess { coordinates } => {
                            let mut output_coords = Vec::new();
                            for coordinate in coordinates {
                                let (coordinate, _) = self.transform_expr(
                                    coordinate,
                                    env,
                                    Some(&Type::Int),
                                    substitutions,
                                    self_type.clone(),
                                )?;
                                output_coords.push(coordinate);
                            }
                            current_type = indexed_array_type(&current_type, output_coords.len())
                                .ok_or_else(|| {
                                format!(
                                    "cannot index into `{}`",
                                    ast::type_to_string(&current_type)
                                )
                            })?;
                            output_nodes.push(Node::ArrayAccess {
                                coordinates: output_coords,
                            });
                        }
                        Node::SliceAccess { start, end } => {
                            let start = if let Some(start) = start {
                                let (start, _) = self.transform_expr(
                                    start,
                                    env,
                                    Some(&Type::Int),
                                    substitutions,
                                    self_type.clone(),
                                )?;
                                Some(Box::new(start))
                            } else {
                                None
                            };
                            let end = if let Some(end) = end {
                                let (end, _) = self.transform_expr(
                                    end,
                                    env,
                                    Some(&Type::Int),
                                    substitutions,
                                    self_type.clone(),
                                )?;
                                Some(Box::new(end))
                            } else {
                                None
                            };
                            current_type = slice_result_type(&current_type)?;
                            output_nodes.push(Node::SliceAccess { start, end });
                        }
                        Node::Dereference { metadata } => {
                            current_type = match current_type {
                                Type::Reference { target_type, .. } => target_type.deref().clone(),
                                Type::Pointer { target_type } => target_type.deref().clone(),
                                other => {
                                    return Err(format!(
                                        "cannot dereference non-pointer type `{}`",
                                        ast::type_to_string(&other)
                                    ))
                                }
                            };
                            output_nodes.push(Node::Dereference {
                                metadata: metadata.clone(),
                            });
                        }
                        Node::MemberAccess { member, metadata } => match member.as_ref() {
                            Node::Identifier(field_name) => {
                                current_type = self
                                    .lookup_struct_field_type(&current_type, field_name)?
                                    .or_else(|| match current_type {
                                        Type::Array { .. } | Type::Slice { .. }
                                            if field_name == "len" =>
                                        {
                                            Some(Type::Int)
                                        }
                                        _ => None,
                                    })
                                    .ok_or_else(|| {
                                        format!(
                                            "unknown field `{}` on `{}`",
                                            field_name,
                                            ast::type_to_string(&current_type)
                                        )
                                    })?;
                                output_nodes.push(Node::MemberAccess {
                                    member: Box::new(Node::Identifier(field_name.clone())),
                                    metadata: metadata.clone(),
                                });
                            }
                            Node::FunctionCall {
                                name: method_name,
                                type_arguments,
                                arguments,
                                metadata: call_metadata,
                            } => {
                                if !type_arguments.is_empty() {
                                    return Err(format!(
                                        "generic method calls are not supported yet: `{}[...]`",
                                        method_name
                                    ));
                                }
                                if matches!(&current_type, Type::Custom(name) if name == "Window") {
                                    let mut output_args = Vec::<Vec<Node>>::new();
                                    let return_type = match method_name.as_str() {
                                        "is_open" => {
                                            if arguments.len() != 1 || !arguments[0].is_empty() {
                                                return Err(format!(
                                                    "error {}:{}: Window.is_open expects no arguments",
                                                    call_metadata.span.line, call_metadata.span.start
                                                ));
                                            }
                                            Type::Boolean
                                        }
                                        "poll" | "present" | "close" | "deinit" => {
                                            if arguments.len() != 1 || !arguments[0].is_empty() {
                                                return Err(format!(
                                                    "error {}:{}: Window.{} expects no arguments",
                                                    call_metadata.span.line, call_metadata.span.start, method_name
                                                ));
                                            }
                                            Type::Void
                                        }
                                        "delta_time" => {
                                            if arguments.len() != 1 || !arguments[0].is_empty() {
                                                return Err(format!(
                                                    "error {}:{}: Window.delta_time expects no arguments",
                                                    call_metadata.span.line, call_metadata.span.start
                                                ));
                                            }
                                            Type::Double
                                        }
                                        "clear" => {
                                            if arguments.len() != 1 || arguments[0].len() != 1 {
                                                return Err(format!(
                                                    "error {}:{}: Window.clear expects one color argument",
                                                    call_metadata.span.line, call_metadata.span.start
                                                ));
                                            }
                                            let (argument, _) = self.transform_expr(
                                                &arguments[0][0],
                                                env,
                                                Some(&Type::Custom("Color".to_string())),
                                                substitutions,
                                                self_type.clone(),
                                            )?;
                                            output_args.push(vec![argument]);
                                            Type::Void
                                        }
                                        "draw_rect" => {
                                            if arguments.len() != 1 || arguments[0].len() != 5 {
                                                return Err(format!(
                                                    "error {}:{}: Window.draw_rect expects x, y, width, height, and color",
                                                    call_metadata.span.line, call_metadata.span.start
                                                ));
                                            }
                                            let mut output_group = Vec::new();
                                            for argument in arguments[0].iter().take(4) {
                                                let (argument, _) = self.transform_expr(
                                                    argument,
                                                    env,
                                                    Some(&Type::Double),
                                                    substitutions,
                                                    self_type.clone(),
                                                )?;
                                                output_group.push(argument);
                                            }
                                            let (color, _) = self.transform_expr(
                                                &arguments[0][4],
                                                env,
                                                Some(&Type::Custom("Color".to_string())),
                                                substitutions,
                                                self_type.clone(),
                                            )?;
                                            output_group.push(color);
                                            output_args.push(output_group);
                                            Type::Void
                                        }
                                        _ => {
                                            return Err(format!(
                                                "error {}:{}: no method named `{}` found for Window",
                                                call_metadata.span.line, call_metadata.span.start, method_name
                                            ))
                                        }
                                    };
                                    current_type = return_type;
                                    output_nodes.push(Node::MemberAccess {
                                        member: Box::new(Node::FunctionCall {
                                            name: method_name.clone(),
                                            type_arguments: Vec::new(),
                                            arguments: if output_args.is_empty() {
                                                arguments.clone()
                                            } else {
                                                output_args
                                            },
                                            metadata: call_metadata.clone(),
                                        }),
                                        metadata: metadata.clone(),
                                    });
                                    continue;
                                }

                                if current_type == Type::Allocator {
                                    let mut output_args = Vec::<Vec<Node>>::new();
                                    let return_type = match method_name.as_str() {
                                        "destroy" => {
                                            if arguments.len() != 1 || arguments[0].len() != 1 {
                                                return Err(format!(
                                                    "error {}:{}: Allocator.destroy expects exactly one pointer argument",
                                                    call_metadata.span.line, call_metadata.span.start
                                                ));
                                            }
                                            let (argument, argument_type) = self.transform_expr(
                                                &arguments[0][0],
                                                env,
                                                None,
                                                substitutions,
                                                self_type.clone(),
                                            )?;
                                            if !matches!(argument_type, Type::Pointer { .. }) {
                                                return Err(format!(
                                                    "error {}:{}: Allocator.destroy expects a pointer argument",
                                                    call_metadata.span.line, call_metadata.span.start
                                                ));
                                            }
                                            output_args.push(vec![argument]);
                                            Type::Void
                                        }
                                        "free" => {
                                            if arguments.len() != 1 || arguments[0].len() != 1 {
                                                return Err(format!(
                                                    "error {}:{}: Allocator.free expects exactly one slice argument",
                                                    call_metadata.span.line, call_metadata.span.start
                                                ));
                                            }
                                            let (argument, argument_type) = self.transform_expr(
                                                &arguments[0][0],
                                                env,
                                                None,
                                                substitutions,
                                                self_type.clone(),
                                            )?;
                                            if !matches!(argument_type, Type::Slice { .. }) {
                                                return Err(format!(
                                                    "error {}:{}: Allocator.free expects a slice argument",
                                                    call_metadata.span.line, call_metadata.span.start
                                                ));
                                            }
                                            output_args.push(vec![argument]);
                                            Type::Void
                                        }
                                        _ => {
                                            return Err(format!(
                                            "error {}:{}: no method named `{}` found for Allocator",
                                            call_metadata.span.line,
                                            call_metadata.span.start,
                                            method_name
                                        ))
                                        }
                                    };
                                    current_type = return_type;
                                    output_nodes.push(Node::MemberAccess {
                                        member: Box::new(Node::FunctionCall {
                                            name: method_name.clone(),
                                            type_arguments: Vec::new(),
                                            arguments: output_args,
                                            metadata: call_metadata.clone(),
                                        }),
                                        metadata: metadata.clone(),
                                    });
                                    continue;
                                }

                                if current_type == Type::Arena {
                                    let no_args = arguments.len() == 1
                                        && arguments.first().is_some_and(|group| group.is_empty());
                                    if !no_args {
                                        return Err(format!(
                                            "error {}:{}: Arena.{} expects no arguments",
                                            call_metadata.span.line,
                                            call_metadata.span.start,
                                            method_name
                                        ));
                                    }

                                    current_type = match method_name.as_str() {
                                        "allocator" => Type::Allocator,
                                        "reset" | "deinit" => Type::Void,
                                        _ => {
                                            return Err(format!(
                                                "error {}:{}: no method named `{}` found for Arena",
                                                call_metadata.span.line,
                                                call_metadata.span.start,
                                                method_name
                                            ))
                                        }
                                    };
                                    output_nodes.push(Node::MemberAccess {
                                        member: Box::new(Node::FunctionCall {
                                            name: method_name.clone(),
                                            type_arguments: Vec::new(),
                                            arguments: arguments.clone(),
                                            metadata: call_metadata.clone(),
                                        }),
                                        metadata: metadata.clone(),
                                    });
                                    continue;
                                }

                                let (_, parameter_types, return_type) = match &current_type {
                                    _ => {
                                        self.lookup_method_signature(&current_type, method_name)?
                                    }
                                };
                                let mut output_args = Vec::<Vec<Node>>::new();
                                let mut group_types = Vec::<Vec<Type>>::new();
                                for (group_index, args) in arguments.iter().enumerate() {
                                    let mut output_group = Vec::new();
                                    let mut type_group = Vec::new();
                                    for (index, argument) in args.iter().enumerate() {
                                        let expected_argument_type = if group_index == 0 {
                                            parameter_types.get(index)
                                        } else {
                                            None
                                        };
                                        let (argument, argument_type) = self.transform_expr(
                                            argument,
                                            env,
                                            expected_argument_type,
                                            substitutions,
                                            self_type.clone(),
                                        )?;
                                        output_group.push(argument);
                                        type_group.push(argument_type);
                                    }
                                    output_args.push(output_group);
                                    group_types.push(type_group);
                                }
                                let method_signature = Type::Function {
                                    parameters: parameter_types,
                                    return_type: Box::new(return_type),
                                };
                                current_type = apply_call_groups_to_function_signature(
                                    &method_signature,
                                    &group_types,
                                    call_metadata,
                                )?;
                                output_nodes.push(Node::MemberAccess {
                                    member: Box::new(Node::FunctionCall {
                                        name: method_name.clone(),
                                        type_arguments: Vec::new(),
                                        arguments: output_args,
                                        metadata: call_metadata.clone(),
                                    }),
                                    metadata: metadata.clone(),
                                });
                            }
                            other => {
                                return Err(format!(
                                    "unsupported member access during monomorphization: `{:?}`",
                                    other
                                ))
                            }
                        },
                        other => {
                            return Err(format!(
                                "unsupported access step during monomorphization: `{:?}`",
                                other
                            ))
                        }
                    }
                }

                Ok((
                    Node::Access {
                        nodes: output_nodes,
                    },
                    current_type,
                ))
            }
            Node::BinaryOp {
                left,
                operator,
                right,
            } => {
                let (left, left_type) = self.transform_expr(
                    left,
                    env,
                    expected_type,
                    substitutions,
                    self_type.clone(),
                )?;
                let (right, right_type) =
                    self.transform_expr(right, env, expected_type, substitutions, self_type)?;
                let result_type = resolve_binary_result_type(operator, &left_type, &right_type)?;
                Ok((
                    Node::BinaryOp {
                        left: Box::new(left),
                        operator: operator.clone(),
                        right: Box::new(right),
                    },
                    result_type,
                ))
            }
            Node::UnaryOp { operator, operand } => {
                let (operand, operand_type) =
                    self.transform_expr(operand, env, expected_type, substitutions, self_type)?;
                let result_type = match operator {
                    UnaryOperator::Plus | UnaryOperator::Minus => operand_type.clone(),
                    UnaryOperator::Negate => Type::Boolean,
                    UnaryOperator::AddressOf => match expected_type.map(ast::unwrap_binding_const) {
                        Some(Type::Pointer { .. }) => Type::Pointer {
                            target_type: Box::new(operand_type.clone()),
                        },
                        _ => Type::Reference {
                            target_type: Box::new(operand_type.clone()),
                            mutable: false,
                        },
                    },
                    UnaryOperator::AddressOfMut => {
                        match expected_type.map(ast::unwrap_binding_const) {
                            Some(Type::Pointer { .. }) => Type::Pointer {
                                target_type: Box::new(operand_type.clone()),
                            },
                            _ => Type::Reference {
                                target_type: Box::new(operand_type.clone()),
                                mutable: true,
                            },
                        }
                    }
                };
                Ok((
                    Node::UnaryOp {
                        operator: operator.clone(),
                        operand: Box::new(operand),
                    },
                    result_type,
                ))
            }
            Node::FunctionDeclaration {
                name,
                parameters,
                return_type,
                body,
                lambda: true,
            } => self.transform_lambda(
                name,
                parameters,
                return_type,
                body,
                env,
                substitutions,
                self_type,
            ),
            Node::Block { .. }
            | Node::UnsafeBlock { .. }
            | Node::If { .. }
            | Node::Match { .. }
            | Node::For { .. }
            | Node::Defer(_)
            | Node::Return(_)
            | Node::Print(_)
            | Node::Input
            | Node::EOI
            | Node::Program { .. }
            | Node::Module { .. }
            | Node::Import { .. }
            | Node::Export { .. }
            | Node::TraitDeclaration { .. }
            | Node::ShapeDeclaration { .. }
            | Node::AttachDeclaration { .. }
            | Node::ConformDeclaration { .. }
            | Node::ImplDeclaration { .. }
            | Node::TypeAliasDeclaration { .. }
            | Node::VariableDeclaration { .. }
            | Node::StructDestructure { .. }
            | Node::StructDeclaration { .. }
            | Node::EnumDeclaration { .. }
            | Node::GenericStructDeclaration { .. }
            | Node::GenericEnumDeclaration { .. }
            | Node::FunctionDeclaration { .. }
            | Node::GenericFunctionDeclaration { .. }
            | Node::ExternFunctionDeclaration { .. }
            | Node::TestDeclaration { .. }
            | Node::EMPTY
            | Node::Assignment { .. } => Err(format!(
                "unsupported expression during monomorphization: `{:?}`",
                node
            )),
            Node::ArrayAccess { .. } | Node::SliceAccess { .. } | Node::MemberAccess { .. } => {
                Err("access steps should be nested inside `Node::Access`".to_string())
            }
        }
    }

    fn transform_match_pattern(
        &mut self,
        pattern: &ast::MatchPattern,
        substitutions: &HashMap<String, Type>,
        matched_type: &Type,
    ) -> Result<ast::MatchPattern, String> {
        match pattern {
            ast::MatchPattern::EnumVariant {
                enum_type,
                variant,
                bindings,
            } => Ok(ast::MatchPattern::EnumVariant {
                enum_type: if let Some(enum_type) = enum_type {
                    Some(self.concretize_type(&self.apply_substitutions(enum_type, substitutions))?)
                } else {
                    match matched_type {
                        Type::GenericInstance { .. } => Some(self.concretize_type(matched_type)?),
                        _ => None,
                    }
                },
                variant: variant.clone(),
                bindings: bindings.clone(),
            }),
            ast::MatchPattern::Struct {
                struct_type,
                fields,
            } => Ok(ast::MatchPattern::Struct {
                struct_type: self
                    .concretize_type(&self.apply_substitutions(struct_type, substitutions))?,
                fields: fields.clone(),
            }),
        }
    }

    fn lookup_enum_variant(
        &mut self,
        enum_type: &Type,
        variant_name: &str,
    ) -> Result<Option<Vec<Type>>, String> {
        match enum_type {
            Type::Custom(name) => Ok(self.concrete_enums.get(name).and_then(|template| {
                template
                    .variants
                    .iter()
                    .find(|variant| variant.name == variant_name)
                    .map(|variant| variant.payload_types.clone())
            })),
            Type::GenericInstance {
                base,
                type_arguments,
            } => {
                let template = match self.generic_enums.get(base) {
                    Some(template) => template,
                    None => return Ok(None),
                };
                let substitutions = template
                    .generic_params
                    .iter()
                    .cloned()
                    .zip(type_arguments.iter().cloned())
                    .collect::<HashMap<_, _>>();
                Ok(template
                    .variants
                    .iter()
                    .find(|variant| variant.name == variant_name)
                    .map(|variant| {
                        variant
                            .payload_types
                            .iter()
                            .map(|payload_type| {
                                self.apply_substitutions(payload_type, &substitutions)
                            })
                            .collect::<Vec<_>>()
                    }))
            }
            _ => Ok(None),
        }
    }

    fn lookup_enum_variant_payload_types(
        &mut self,
        enum_type: &Type,
        pattern: &ast::MatchPattern,
    ) -> Result<Vec<Type>, String> {
        match pattern {
            ast::MatchPattern::EnumVariant { variant, .. } => self
                .lookup_enum_variant(enum_type, variant)
                .map(|payload| payload.unwrap_or_default()),
            ast::MatchPattern::Struct { .. } => Ok(Vec::new()),
        }
    }

    fn lookup_struct_pattern_bindings(
        &mut self,
        struct_type: &Type,
        pattern: &ast::MatchPattern,
    ) -> Result<Vec<(String, Type)>, String> {
        match pattern {
            ast::MatchPattern::Struct { fields, .. } => {
                let mut bindings = Vec::new();
                for field in fields {
                    let field_type = self
                        .lookup_struct_field_type(struct_type, &field.field_name)?
                        .ok_or_else(|| {
                            format!(
                                "unknown field `{}` on `{}`",
                                field.field_name,
                                ast::type_to_string(struct_type)
                            )
                        })?;
                    bindings.push((field.binding.clone(), field_type));
                }
                Ok(bindings)
            }
            _ => Ok(Vec::new()),
        }
    }

    fn transform_function_call(
        &mut self,
        name: &str,
        explicit_type_arguments: &[Type],
        arguments: &[Vec<Node>],
        metadata: &Metadata,
        env: &mut Env,
        expected_type: Option<&Type>,
        substitutions: &HashMap<String, Type>,
        self_type: Option<Type>,
    ) -> Result<(Node, Type), String> {
        let mut output_args = Vec::new();
        let mut argument_types = Vec::new();
        for args in arguments {
            let transformed = args
                .iter()
                .map(|arg| self.transform_expr(arg, env, None, substitutions, self_type.clone()))
                .collect::<Result<Vec<_>, String>>()?;
            output_args.push(transformed.iter().map(|(node, _)| node.clone()).collect());
            argument_types.push(
                transformed
                    .into_iter()
                    .map(|(_, sk_type)| sk_type)
                    .collect::<Vec<_>>(),
            );
        }

        if let Some(template) = self.generic_functions.get(name).cloned() {
            let substitutions = if explicit_type_arguments.is_empty() {
                self.infer_generic_function_arguments(&template, &argument_types, expected_type)?
            } else {
                if template.generic_params.len() != explicit_type_arguments.len() {
                    return Err(format!(
                        "generic function `{}` expects {} type arguments, got {}",
                        template.name,
                        template.generic_params.len(),
                        explicit_type_arguments.len()
                    ));
                }
                template
                    .generic_params
                    .iter()
                    .cloned()
                    .zip(
                        explicit_type_arguments
                            .iter()
                            .map(|arg| self.concretize_type(&self.apply_substitutions(arg, substitutions)))
                            .collect::<Result<Vec<_>, String>>()?,
                    )
                    .collect::<HashMap<_, _>>()
            };
            self.check_generic_bounds(
                &template.generic_bounds,
                &template.subtype_bounds,
                &substitutions,
                &format!("generic function `{}`", template.name),
            )?;
            let specialized_name = self.ensure_specialized_function(&template, &substitutions)?;
            let substituted_return_type =
                self.apply_substitutions(&template.return_type, &substitutions);
            let base_return_type = self.expand_type(&substituted_return_type)?;
            let result_type = apply_call_groups_to_function_signature(
                &base_return_type,
                &argument_types[1..],
                metadata,
            )?;
            return Ok((
                Node::FunctionCall {
                    name: specialized_name,
                    type_arguments: Vec::new(),
                    arguments: output_args,
                    metadata: metadata.clone(),
                },
                result_type,
            ));
        }

        if !explicit_type_arguments.is_empty() {
            return Err(format!(
                "function `{}` does not accept explicit type arguments",
                name
            ));
        }

        let signature = if let Some(local_type) = env.get(name) {
            ast::strip_binding_const(&local_type)
        } else if let Some(function) = self.concrete_functions.get(name).cloned() {
            let signature = Type::Function {
                parameters: function
                    .parameters
                    .iter()
                    .map(|(_, sk_type)| sk_type.clone())
                    .collect(),
                return_type: Box::new(function.return_type.clone()),
            };
            self.expand_type(&signature)?
        } else {
            return Err(format!("unknown function `{}`", name));
        };

        let result_type =
            apply_call_groups_to_function_signature(&signature, &argument_types, metadata)?;
        Ok((
            Node::FunctionCall {
                name: name.to_string(),
                type_arguments: Vec::new(),
                arguments: output_args,
                metadata: metadata.clone(),
            },
            result_type,
        ))
    }

    fn infer_generic_function_arguments(
        &mut self,
        template: &FunctionTemplate,
        argument_types: &[Vec<Type>],
        expected_type: Option<&Type>,
    ) -> Result<HashMap<String, Type>, String> {
        let mut substitutions = HashMap::<String, Type>::new();
        let parameter_types = &template.parameters;
        let first_args = argument_types.first().cloned().unwrap_or_default();
        if first_args.len() != parameter_types.len() {
            return Err(format!(
                "incorrect number of args to function {}. expected={}, actual={}",
                template.name,
                parameter_types.len(),
                first_args.len()
            ));
        }
        for ((_, parameter_type), argument_type) in parameter_types.iter().zip(first_args.iter()) {
            if let Some((param, candidate)) = self.direct_generic_candidate(
                parameter_type,
                argument_type,
                &template.generic_params,
            ) {
                self.merge_inferred_lower(&param, &candidate, &mut substitutions)?;
            } else {
                self.unify_generic_type(
                    parameter_type,
                    argument_type,
                    &template.generic_params,
                    &mut substitutions,
                )?;
            }
        }
        if let Some(expected_type) = expected_type {
            if let Some((param, candidate)) = self.direct_generic_candidate(
                &template.return_type,
                expected_type,
                &template.generic_params,
            ) {
                substitutions.entry(param).or_insert(candidate);
            } else {
                self.unify_generic_type(
                    &template.return_type,
                    expected_type,
                    &template.generic_params,
                    &mut substitutions,
                )?;
            }
        }

        for _ in 0..=template.generic_params.len() {
            let mut changed = false;
            for param in &template.generic_params {
                let Some(bounds) = template.subtype_bounds.get(param) else {
                    continue;
                };
                let Some(lower) = &bounds.lower else {
                    continue;
                };
                let lower = self.apply_substitutions(lower, &substitutions);
                if contains_unresolved_generic(
                    &lower,
                    &template.generic_params,
                    &substitutions,
                ) {
                    continue;
                }
                let previous = substitutions.get(param).cloned();
                self.merge_inferred_lower(param, &lower, &mut substitutions)?;
                changed |= substitutions.get(param) != previous.as_ref();
            }
            if !changed {
                break;
            }
        }
        for generic_param in &template.generic_params {
            if !substitutions.contains_key(generic_param) {
                return Err(format!(
                    "could not infer type argument `{}` for generic function `{}`",
                    generic_param, template.name
                ));
            }
        }
        Ok(substitutions)
    }

    fn direct_generic_candidate(
        &self,
        pattern: &Type,
        actual: &Type,
        generic_params: &[String],
    ) -> Option<(String, Type)> {
        match pattern {
            Type::BindingConst { inner } => self.direct_generic_candidate(
                inner,
                ast::unwrap_binding_const(actual),
                generic_params,
            ),
            Type::Const { inner } => self.direct_generic_candidate(
                inner,
                ast::unwrap_const_view(actual),
                generic_params,
            ),
            Type::Custom(name) if generic_params.iter().any(|param| param == name) => {
                Some((name.clone(), actual.clone()))
            }
            _ => None,
        }
    }

    fn merge_inferred_lower(
        &mut self,
        param: &str,
        candidate: &Type,
        substitutions: &mut HashMap<String, Type>,
    ) -> Result<(), String> {
        let candidate = self.expand_type(candidate)?;
        let Some(existing) = substitutions.get(param).cloned() else {
            substitutions.insert(param.to_string(), candidate);
            return Ok(());
        };
        let existing = self.expand_type(&existing)?;
        let widened = if self.is_subtype(&existing, &candidate)? {
            candidate
        } else if self.is_subtype(&candidate, &existing)? {
            existing
        } else {
            self.expand_type(&Type::Union(vec![existing, candidate]))?
        };
        substitutions.insert(param.to_string(), widened);
        Ok(())
    }

    fn unify_generic_type(
        &self,
        pattern: &Type,
        actual: &Type,
        generic_params: &[String],
        substitutions: &mut HashMap<String, Type>,
    ) -> Result<(), String> {
        match pattern {
            Type::BindingConst { inner } => self.unify_generic_type(
                inner,
                ast::unwrap_binding_const(actual),
                generic_params,
                substitutions,
            ),
            Type::Const { inner } => self.unify_generic_type(
                inner,
                ast::unwrap_const_view(actual),
                generic_params,
                substitutions,
            ),
            Type::Custom(name) if generic_params.iter().any(|param| param == name) => {
                if let Some(existing) = substitutions.get(name) {
                    if existing != actual {
                        return Err(format!(
                            "conflicting inferred types for `{}`: `{}` and `{}`",
                            name,
                            ast::type_to_string(existing),
                            ast::type_to_string(actual)
                        ));
                    }
                } else {
                    substitutions.insert(name.clone(), actual.clone());
                }
                Ok(())
            }
            Type::Array {
                elem_type,
                dimensions,
            } => {
                let Type::Array {
                    elem_type: actual_elem_type,
                    dimensions: actual_dimensions,
                } = actual
                else {
                    return Err(format!(
                        "expected `{}`, found `{}`",
                        ast::type_to_string(pattern),
                        ast::type_to_string(actual)
                    ));
                };
                if dimensions.len() != actual_dimensions.len() {
                    return Err("array dimension mismatch during generic inference".to_string());
                }
                self.unify_generic_type(elem_type, actual_elem_type, generic_params, substitutions)
            }
            Type::Slice { elem_type } => {
                let Type::Slice {
                    elem_type: actual_elem_type,
                } = actual
                else {
                    return Err(format!(
                        "expected `{}`, found `{}`",
                        ast::type_to_string(pattern),
                        ast::type_to_string(actual)
                    ));
                };
                self.unify_generic_type(elem_type, actual_elem_type, generic_params, substitutions)
            }
            Type::Reference {
                target_type,
                mutable,
            } => {
                let Type::Reference {
                    target_type: actual_target_type,
                    mutable: actual_mutable,
                } = actual
                else {
                    return Err(format!(
                        "expected `{}`, found `{}`",
                        ast::type_to_string(pattern),
                        ast::type_to_string(actual)
                    ));
                };
                if mutable != actual_mutable {
                    return Err(format!(
                        "expected `{}`, found `{}`",
                        ast::type_to_string(pattern),
                        ast::type_to_string(actual)
                    ));
                }
                self.unify_generic_type(
                    target_type,
                    actual_target_type,
                    generic_params,
                    substitutions,
                )
            }
            Type::Pointer { target_type } => {
                let Type::Pointer {
                    target_type: actual_target_type,
                } = actual
                else {
                    return Err(format!(
                        "expected `{}`, found `{}`",
                        ast::type_to_string(pattern),
                        ast::type_to_string(actual)
                    ));
                };
                self.unify_generic_type(
                    target_type,
                    actual_target_type,
                    generic_params,
                    substitutions,
                )
            }
            Type::Function {
                parameters,
                return_type,
            } => {
                let Type::Function {
                    parameters: actual_parameters,
                    return_type: actual_return_type,
                } = actual
                else {
                    return Err(format!(
                        "expected `{}`, found `{}`",
                        ast::type_to_string(pattern),
                        ast::type_to_string(actual)
                    ));
                };
                if parameters.len() != actual_parameters.len() {
                    return Err("function arity mismatch during generic inference".to_string());
                }
                for (parameter, actual_parameter) in parameters.iter().zip(actual_parameters.iter())
                {
                    self.unify_generic_type(
                        parameter,
                        actual_parameter,
                        generic_params,
                        substitutions,
                    )?;
                }
                self.unify_generic_type(
                    return_type,
                    actual_return_type,
                    generic_params,
                    substitutions,
                )
            }
            Type::GenericInstance {
                base,
                type_arguments,
            } => {
                let Type::GenericInstance {
                    base: actual_base,
                    type_arguments: actual_type_arguments,
                } = actual
                else {
                    return Err(format!(
                        "expected `{}`, found `{}`",
                        ast::type_to_string(pattern),
                        ast::type_to_string(actual)
                    ));
                };
                if base != actual_base || type_arguments.len() != actual_type_arguments.len() {
                    return Err(format!(
                        "expected `{}`, found `{}`",
                        ast::type_to_string(pattern),
                        ast::type_to_string(actual)
                    ));
                }
                for (type_argument, actual_type_argument) in
                    type_arguments.iter().zip(actual_type_arguments.iter())
                {
                    self.unify_generic_type(
                        type_argument,
                        actual_type_argument,
                        generic_params,
                        substitutions,
                    )?;
                }
                Ok(())
            }
            _ => {
                if pattern == actual {
                    Ok(())
                } else {
                    Err(format!(
                        "expected `{}`, found `{}`",
                        ast::type_to_string(pattern),
                        ast::type_to_string(actual)
                    ))
                }
            }
        }
    }

    fn ensure_specialized_function(
        &mut self,
        template: &FunctionTemplate,
        substitutions: &HashMap<String, Type>,
    ) -> Result<String, String> {
        self.check_generic_bounds(
            &template.generic_bounds,
            &template.subtype_bounds,
            substitutions,
            &format!("generic function `{}`", template.name),
        )?;
        let symbol_name =
            specialized_function_name(&template.name, substitutions, &template.generic_params);
        if self.generated_functions.contains_key(&symbol_name) {
            return Ok(symbol_name);
        }
        if !self.function_stack.insert(symbol_name.clone()) {
            return Ok(symbol_name);
        }

        let node = self.transform_named_function(
            &symbol_name,
            &template.parameters,
            &template.return_type,
            &template.body,
            substitutions,
            None,
        )?;
        self.generated_functions.insert(symbol_name.clone(), node);
        self.generated_function_order.push(symbol_name.clone());
        self.function_stack.remove(&symbol_name);
        Ok(symbol_name)
    }

    fn ensure_struct_for_type(&mut self, sk_type: &Type) -> Result<String, String> {
        match sk_type {
            Type::Custom(name) => Ok(name.clone()),
            Type::GenericInstance {
                base,
                type_arguments,
            } => {
                let template = self
                    .generic_structs
                    .get(base)
                    .cloned()
                    .ok_or_else(|| format!("unknown generic struct `{}`", base))?;
                if template.generic_params.len() != type_arguments.len() {
                    return Err(format!(
                        "generic struct `{}` expects {} type arguments, got {}",
                        base,
                        template.generic_params.len(),
                        type_arguments.len()
                    ));
                }
                let symbol_name = specialized_struct_name(base, type_arguments);
                if self.generated_structs.contains_key(&symbol_name) {
                    return Ok(symbol_name);
                }
                if !self.struct_stack.insert(symbol_name.clone()) {
                    return Ok(symbol_name);
                }
                let substitutions = template
                    .generic_params
                    .iter()
                    .cloned()
                    .zip(type_arguments.iter().cloned())
                    .collect::<HashMap<_, _>>();
                self.check_generic_bounds(
                    &template.generic_bounds,
                    &template.subtype_bounds,
                    &substitutions,
                    &format!("generic struct `{}`", base),
                )?;
                let node = self.transform_struct_decl(
                    &symbol_name,
                    &template.fields,
                    &template.functions,
                    &substitutions,
                    Some(sk_type.clone()),
                )?;
                self.generated_structs.insert(symbol_name.clone(), node);
                self.generated_struct_order.push(symbol_name.clone());
                self.ensure_runtime_impls_for_type(sk_type, &Type::Custom(symbol_name.clone()))?;
                self.struct_stack.remove(&symbol_name);
                Ok(symbol_name)
            }
            other => Err(format!(
                "expected a struct type, found `{}`",
                ast::type_to_string(other)
            )),
        }
    }

    fn ensure_enum_for_type(&mut self, sk_type: &Type) -> Result<String, String> {
        match sk_type {
            Type::Custom(name) => Ok(name.clone()),
            Type::GenericInstance {
                base,
                type_arguments,
            } => {
                let template = self
                    .generic_enums
                    .get(base)
                    .cloned()
                    .ok_or_else(|| format!("unknown generic enum `{}`", base))?;
                if template.generic_params.len() != type_arguments.len() {
                    return Err(format!(
                        "generic enum `{}` expects {} type arguments, got {}",
                        base,
                        template.generic_params.len(),
                        type_arguments.len()
                    ));
                }
                let symbol_name = specialized_struct_name(base, type_arguments);
                if self.generated_enums.contains_key(&symbol_name) {
                    return Ok(symbol_name);
                }
                if !self.enum_stack.insert(symbol_name.clone()) {
                    return Ok(symbol_name);
                }
                let substitutions = template
                    .generic_params
                    .iter()
                    .cloned()
                    .zip(type_arguments.iter().cloned())
                    .collect::<HashMap<_, _>>();
                self.check_generic_bounds(
                    &template.generic_bounds,
                    &template.subtype_bounds,
                    &substitutions,
                    &format!("generic enum `{}`", base),
                )?;
                let node =
                    self.transform_enum_decl(&symbol_name, &template.variants, &substitutions)?;
                self.generated_enums.insert(symbol_name.clone(), node);
                self.generated_enum_order.push(symbol_name.clone());
                self.ensure_runtime_impls_for_type(sk_type, &Type::Custom(symbol_name.clone()))?;
                self.enum_stack.remove(&symbol_name);
                Ok(symbol_name)
            }
            other => Err(format!(
                "expected an enum type, found `{}`",
                ast::type_to_string(other)
            )),
        }
    }

    fn ensure_nominal_type(&mut self, sk_type: &Type) -> Result<String, String> {
        match sk_type {
            Type::Custom(name) => Ok(name.clone()),
            Type::GenericInstance { base, .. } if self.generic_structs.contains_key(base) => {
                self.ensure_struct_for_type(sk_type)
            }
            Type::GenericInstance { base, .. } if self.generic_enums.contains_key(base) => {
                self.ensure_enum_for_type(sk_type)
            }
            Type::GenericInstance { base, .. } => {
                Err(format!("unknown generic nominal type `{}`", base))
            }
            other => Err(format!(
                "expected a nominal type, found `{}`",
                ast::type_to_string(other)
            )),
        }
    }

    fn lookup_struct_field_type(
        &mut self,
        sk_type: &Type,
        field_name: &str,
    ) -> Result<Option<Type>, String> {
        match sk_type {
            Type::Custom(name) => {
                if let Some(template) = self.concrete_structs.get(name) {
                    let field_type = template
                        .fields
                        .iter()
                        .find(|(candidate, _)| candidate == field_name)
                        .map(|(_, field_type)| field_type.clone());
                    field_type
                        .map(|field_type| self.expand_type(&field_type))
                        .transpose()
                } else {
                    Ok(None)
                }
            }
            Type::GenericInstance {
                base,
                type_arguments,
            } => {
                let template = self
                    .generic_structs
                    .get(base)
                    .ok_or_else(|| format!("unknown generic struct `{}`", base))?;
                let substitutions = template
                    .generic_params
                    .iter()
                    .cloned()
                    .zip(type_arguments.iter().cloned())
                    .collect::<HashMap<_, _>>();
                let field_type = template
                    .fields
                    .iter()
                    .find(|(candidate, _)| candidate == field_name)
                    .map(|(_, field_type)| self.apply_substitutions(field_type, &substitutions));
                field_type
                    .map(|field_type| self.expand_type(&field_type))
                    .transpose()
            }
            _ => Ok(None),
        }
    }

    fn lookup_method_signature(
        &mut self,
        receiver_type: &Type,
        method_name: &str,
    ) -> Result<(Type, Vec<Type>, Type), String> {
        let result = match receiver_type {
            Type::Custom(name) => {
                if let Some(template) = self.concrete_structs.get(name) {
                    let function = template
                        .functions
                        .iter()
                        .find_map(|function| match function {
                            Node::FunctionDeclaration {
                                name,
                                parameters,
                                return_type,
                                ..
                            } if name == method_name => Some((parameters.clone(), return_type.clone())),
                            _ => None,
                        })
                        .ok_or_else(|| format!("unknown method `{}` on `{}`", method_name, name))?;
                    let receiver = function
                        .0
                        .first()
                        .map(|(_, sk_type)| sk_type.clone())
                        .ok_or_else(|| {
                            format!("method `{}` on `{}` is missing self", method_name, name)
                        })?;
                    Ok((
                        receiver,
                        function
                            .0
                            .into_iter()
                            .filter(|(_, sk_type)| !ast::is_self_type(sk_type))
                            .map(|(_, sk_type)| sk_type)
                            .collect::<Vec<_>>(),
                        function.1,
                    ))
                } else if let Some(trait_template) = self.traits.get(name) {
                    let method = trait_template
                        .methods
                        .iter()
                        .find(|method| method.name == method_name)
                        .ok_or_else(|| format!("unknown method `{}` on trait `{}`", method_name, name))?;
                    let receiver = method
                        .parameters
                        .first()
                        .map(|(_, sk_type)| sk_type.clone())
                        .ok_or_else(|| {
                            format!("method `{}` on trait `{}` is missing self", method_name, name)
                        })?;
                    Ok((
                        receiver,
                        method
                            .parameters
                            .iter()
                            .skip(1)
                            .map(|(_, sk_type)| sk_type.clone())
                            .collect(),
                        method.return_type.clone(),
                    ))
                } else {
                    Err(format!("unknown struct or trait `{}`", name))
                }
            }
            Type::GenericInstance {
                base,
                type_arguments,
            } => {
                let template = self
                    .generic_structs
                    .get(base)
                    .ok_or_else(|| format!("unknown generic struct `{}`", base))?;
                let substitutions = template
                    .generic_params
                    .iter()
                    .cloned()
                    .zip(type_arguments.iter().cloned())
                    .collect::<HashMap<_, _>>();
                let function = template
                    .functions
                    .iter()
                    .find_map(|function| match function {
                        Node::FunctionDeclaration {
                            name,
                            parameters,
                            return_type,
                            ..
                        } if name == method_name => Some((parameters.clone(), return_type.clone())),
                        _ => None,
                    })
                    .ok_or_else(|| format!("unknown method `{}` on `{}`", method_name, base))?;
                let receiver = function
                    .0
                    .first()
                    .map(|(_, sk_type)| self.apply_substitutions(sk_type, &substitutions))
                    .ok_or_else(|| {
                        format!("method `{}` on `{}` is missing self", method_name, base)
                    })?;
                Ok((
                    receiver,
                    function
                        .0
                        .into_iter()
                        .filter(|(_, sk_type)| !ast::is_self_type(sk_type))
                        .map(|(_, sk_type)| self.apply_substitutions(&sk_type, &substitutions))
                        .collect(),
                    self.apply_substitutions(&function.1, &substitutions),
                ))
            }
            Type::Intersection(members) => {
                let mut found = Vec::new();
                for member in members {
                    let Type::Custom(trait_name) = member else {
                        continue;
                    };
                    let methods = self.collect_trait_methods(trait_name, &mut Vec::new())?;
                    if let Some(method) = methods.into_iter().find(|method| method.name == method_name)
                    {
                        found.push((trait_name.clone(), method));
                    }
                }
                if found.is_empty() {
                    return Err(format!(
                        "unknown method `{}` on intersection `{}`",
                        method_name,
                        ast::type_to_string(receiver_type)
                    ));
                }
                if found.len() > 1 {
                    return Err(format!(
                        "ambiguous method `{}` on intersection `{}`",
                        method_name,
                        ast::type_to_string(receiver_type)
                    ));
                }
                let (_, method) = found.pop().unwrap();
                let receiver = method
                    .parameters
                    .first()
                    .map(|(_, sk_type)| sk_type.clone())
                    .ok_or_else(|| format!("method `{}` is missing self", method_name))?;
                Ok((
                    receiver,
                    method
                        .parameters
                        .iter()
                        .skip(1)
                        .map(|(_, sk_type)| sk_type.clone())
                        .collect(),
                    method.return_type,
                ))
            }
            other => Err(format!(
                "method lookup requires a struct receiver, found `{}`",
                ast::type_to_string(other)
            )),
        };
        let (receiver, parameters, return_type) = result?;
        Ok((
            self.expand_type(&receiver)?,
            parameters
                .iter()
                .map(|parameter| self.expand_type(parameter))
                .collect::<Result<Vec<_>, _>>()?,
            self.expand_type(&return_type)?,
        ))
    }

    fn lookup_static_function_signature(
        &mut self,
        target_type: &Type,
        function_name: &str,
    ) -> Result<(Vec<Type>, Type), String> {
        let result = match target_type {
            Type::Custom(name) => {
                let template = self
                    .concrete_structs
                    .get(name)
                    .ok_or_else(|| format!("unknown struct `{}`", name))?;
                let function = template
                    .functions
                    .iter()
                    .find_map(|function| match function {
                        Node::FunctionDeclaration {
                            name,
                            parameters,
                            return_type,
                            ..
                        } if name == function_name
                            && !parameters
                                .first()
                                .is_some_and(|(_, sk_type)| ast::is_self_type(sk_type)) =>
                        {
                            Some((parameters.clone(), return_type.clone()))
                        }
                        _ => None,
                    })
                    .ok_or_else(|| {
                        format!("unknown static function `{}` on `{}`", function_name, name)
                    })?;
                Ok((
                    function
                        .0
                        .into_iter()
                        .map(|(_, sk_type)| sk_type)
                        .collect::<Vec<_>>(),
                    function.1,
                ))
            }
            Type::GenericInstance {
                base,
                type_arguments,
            } => {
                let template = self
                    .generic_structs
                    .get(base)
                    .ok_or_else(|| format!("unknown generic struct `{}`", base))?;
                let substitutions = template
                    .generic_params
                    .iter()
                    .cloned()
                    .zip(type_arguments.iter().cloned())
                    .collect::<HashMap<_, _>>();
                let function = template
                    .functions
                    .iter()
                    .find_map(|function| match function {
                        Node::FunctionDeclaration {
                            name,
                            parameters,
                            return_type,
                            ..
                        } if name == function_name
                            && !parameters
                                .first()
                                .is_some_and(|(_, sk_type)| ast::is_self_type(sk_type)) =>
                        {
                            Some((parameters.clone(), return_type.clone()))
                        }
                        _ => None,
                    })
                    .ok_or_else(|| {
                        format!("unknown static function `{}` on `{}`", function_name, base)
                    })?;
                Ok((
                    function
                        .0
                        .into_iter()
                        .map(|(_, sk_type)| self.apply_substitutions(&sk_type, &substitutions))
                        .collect(),
                    self.apply_substitutions(&function.1, &substitutions),
                ))
            }
            other => Err(format!(
                "static function lookup requires a struct type, found `{}`",
                ast::type_to_string(other)
            )),
        };
        let (parameters, return_type) = result?;
        Ok((
            parameters
                .iter()
                .map(|parameter| self.expand_type(parameter))
                .collect::<Result<Vec<_>, _>>()?,
            self.expand_type(&return_type)?,
        ))
    }

    fn apply_substitutions(&self, sk_type: &Type, substitutions: &HashMap<String, Type>) -> Type {
        match sk_type {
            Type::Const { inner } => Type::Const {
                inner: Box::new(self.apply_substitutions(inner, substitutions)),
            },
            Type::BindingConst { inner } => Type::BindingConst {
                inner: Box::new(self.apply_substitutions(inner, substitutions)),
            },
            Type::Reference {
                target_type,
                mutable,
            } => Type::Reference {
                target_type: Box::new(self.apply_substitutions(target_type, substitutions)),
                mutable: *mutable,
            },
            Type::MutSelf => Type::MutSelf,
            Type::Custom(name) => substitutions
                .get(name)
                .cloned()
                .unwrap_or_else(|| Type::Custom(name.clone())),
            Type::Array {
                elem_type,
                dimensions,
            } => Type::Array {
                elem_type: Box::new(self.apply_substitutions(elem_type, substitutions)),
                dimensions: dimensions.clone(),
            },
            Type::Pointer { target_type } => Type::Pointer {
                target_type: Box::new(self.apply_substitutions(target_type, substitutions)),
            },
            Type::Slice { elem_type } => Type::Slice {
                elem_type: Box::new(self.apply_substitutions(elem_type, substitutions)),
            },
            Type::GenericInstance {
                base,
                type_arguments,
            } => Type::GenericInstance {
                base: base.clone(),
                type_arguments: type_arguments
                    .iter()
                    .map(|sk_type| self.apply_substitutions(sk_type, substitutions))
                    .collect(),
            },
            Type::Union(members) => Type::Union(
                members
                    .iter()
                    .map(|member| self.apply_substitutions(member, substitutions))
                    .collect(),
            ),
            Type::Intersection(members) => Type::Intersection(
                members
                    .iter()
                    .map(|member| self.apply_substitutions(member, substitutions))
                    .collect(),
            ),
            Type::Function {
                parameters,
                return_type,
            } => Type::Function {
                parameters: parameters
                    .iter()
                    .map(|parameter| self.apply_substitutions(parameter, substitutions))
                    .collect(),
                return_type: Box::new(self.apply_substitutions(return_type, substitutions)),
            },
            other => other.clone(),
        }
    }

    fn concretize_type(&mut self, sk_type: &Type) -> Result<Type, String> {
        let expanded = self.expand_type(sk_type)?;
        self.concretize_expanded_type(&expanded)
    }

    fn concretize_expanded_type(&mut self, sk_type: &Type) -> Result<Type, String> {
        match sk_type {
            Type::Const { inner } => Ok(Type::Const {
                inner: Box::new(self.concretize_expanded_type(inner)?),
            }),
            Type::BindingConst { inner } => Ok(Type::BindingConst {
                inner: Box::new(self.concretize_expanded_type(inner)?),
            }),
            Type::MutSelf => Ok(Type::MutSelf),
            Type::Array {
                elem_type,
                dimensions,
            } => Ok(Type::Array {
                elem_type: Box::new(self.concretize_expanded_type(elem_type)?),
                dimensions: dimensions.clone(),
            }),
            Type::Reference {
                target_type,
                mutable,
            } => Ok(Type::Reference {
                target_type: Box::new(self.concretize_expanded_type(target_type)?),
                mutable: *mutable,
            }),
            Type::Pointer { target_type } => Ok(Type::Pointer {
                target_type: Box::new(self.concretize_expanded_type(target_type)?),
            }),
            Type::Slice { elem_type } => Ok(Type::Slice {
                elem_type: Box::new(self.concretize_expanded_type(elem_type)?),
            }),
            Type::Union(members) => Ok(Type::Union(
                members
                    .iter()
                    .map(|member| self.concretize_expanded_type(member))
                    .collect::<Result<Vec<_>, _>>()?,
            )),
            Type::Intersection(members) => Ok(Type::Intersection(
                members
                    .iter()
                    .map(|member| self.concretize_expanded_type(member))
                    .collect::<Result<Vec<_>, _>>()?,
            )),
            Type::Function {
                parameters,
                return_type,
            } => Ok(Type::Function {
                parameters: parameters
                    .iter()
                    .map(|parameter| self.concretize_expanded_type(parameter))
                    .collect::<Result<Vec<_>, _>>()?,
                return_type: Box::new(self.concretize_expanded_type(return_type)?),
            }),
            Type::GenericInstance { .. } => Ok(Type::Custom(self.ensure_nominal_type(sk_type)?)),
            other => Ok(other.clone()),
        }
    }

    fn literal_type(
        &self,
        literal: &Literal,
        expected_type: Option<&Type>,
    ) -> Result<Type, String> {
        match literal {
            Literal::Integer(value) => {
                if let Some(expected_type) = expected_type {
                    if ast::is_integral_type(expected_type) {
                        if *expected_type == Type::Long
                            && !ast::fits_integer_type(*value, &Type::Int)
                        {
                            return Err(format!(
                                "integer literal `{}` is out of range for `int`; use an `L` suffix for `long`",
                                value
                            ));
                        }
                        if ast::fits_integer_type(*value, expected_type) {
                            return Ok(expected_type.clone());
                        }
                        return Err(format!(
                            "integer literal `{}` is out of range for `{}`",
                            value,
                            ast::type_to_string(expected_type)
                        ));
                    }
                }
                if ast::fits_integer_type(*value, &Type::Int) {
                    Ok(Type::Int)
                } else {
                    Err(format!(
                        "integer literal `{}` is out of range for `int`; use an `L` suffix for `long`",
                        value
                    ))
                }
            }
            Literal::Long(_) => Ok(Type::Long),
            Literal::Float(_) => Ok(Type::Float),
            Literal::Double(_) => Ok(Type::Double),
            Literal::StringLiteral(_) => Ok(Type::String),
            Literal::Boolean(_) => Ok(Type::Boolean),
            Literal::Char(_) => Ok(Type::Char),
        }
    }
}

fn contains_unresolved_generic(
    sk_type: &Type,
    generic_params: &[String],
    substitutions: &HashMap<String, Type>,
) -> bool {
    match sk_type {
        Type::Custom(name) => {
            generic_params.iter().any(|param| param == name)
                && !substitutions.contains_key(name)
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

fn validate_type_alias_cycles(
    aliases: &HashMap<String, TypeAliasTemplate>,
) -> Result<(), String> {
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
            members.iter().map(type_mangle).collect::<Vec<_>>().join("_or_")
        ),
        Type::Intersection(members) => format!(
            "intersection_{}",
            members.iter().map(type_mangle).collect::<Vec<_>>().join("_and_")
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast;

    fn prepared_statements(source: &str) -> Vec<Node> {
        let program = ast::parse(source);
        let Node::Program { statements } = prepare_program(&program).unwrap() else {
            panic!("expected prepared program");
        };
        statements
    }

    #[test]
    fn monomorphizes_generic_wrap_program() {
        let statements = prepared_statements(
            r#"
            struct Box[T] {
                value: T;
            }

            function wrap[T](value: T): Box[T] {
                return Box[T] { value: value };
            }

            function main(): void {
                box: Box[int] = wrap(7);
                print(box.value);
            }
            "#,
        );

        assert!(statements.iter().any(|statement| matches!(
            statement,
            Node::StructDeclaration { name, .. } if name == "Box__int"
        )));
        assert!(statements.iter().any(|statement| matches!(
            statement,
            Node::FunctionDeclaration { name, body, .. }
                if name == "wrap__int" && matches!(body.first(), Some(Node::Return(Some(_))))
        )));
    }

    #[test]
    fn monomorphizes_nested_generic_program() {
        let statements = prepared_statements(
            r#"
            struct Box[T] {
                value: T;
            }

            function wrap[T](value: T): Box[T] {
                return Box[T] { value: value };
            }

            function main(): void {
                inner: Box[int] = wrap(7);
                outer: Box[Box[int]] = wrap(inner);
                print(outer.value.value);
            }
            "#,
        );

        assert!(statements.iter().any(|statement| matches!(
            statement,
            Node::StructDeclaration { name, .. } if name == "Box__int"
        )));
        assert!(statements.iter().any(|statement| matches!(
            statement,
            Node::StructDeclaration { name, .. } if name == "Box__Box__int"
        )));
        assert!(statements.iter().any(|statement| matches!(
            statement,
            Node::FunctionDeclaration { name, body, .. }
                if name == "wrap__Box__int" && matches!(body.first(), Some(Node::Return(Some(_))))
        )));
    }

    #[test]
    fn monomorphizes_generic_enum_program() {
        let statements = prepared_statements(
            r#"
            enum Option[T] {
                None;
                Some(T);
            }

            function wrap[T](value: T): Option[T] {
                return Option[T]::Some(value);
            }

            function main(): void {
                value: Option[int] = wrap(7);
                match (value) {
                    case None: {
                        print(0);
                    }
                    case Some(v): {
                        print(v);
                    }
                }
            }
            "#,
        );

        assert!(statements.iter().any(|statement| matches!(
            statement,
            Node::EnumDeclaration { name, variants }
                if name == "Option__int"
                    && variants
                        .iter()
                        .any(|variant| variant.name == "Some" && variant.payload_types == vec![Type::Int])
        )));
        assert!(statements.iter().any(|statement| matches!(
            statement,
            Node::FunctionDeclaration { name, .. } if name == "wrap__int"
        )));
    }

    #[test]
    fn expands_concrete_and_generic_type_aliases() {
        let statements = prepared_statements(
            r#"
            type Value = string | int;
            type Either[T] = T | string;

            function use_value(value: Value): Either[int] {
                return value;
            }

            function main(): void {
                value: Value = 7;
                result: Either[int] = use_value(value);
            }
            "#,
        );
        let use_value = statements
            .iter()
            .find(|statement| matches!(
                statement,
                Node::FunctionDeclaration { name, .. } if name == "use_value"
            ))
            .expect("prepared function");
        let Node::FunctionDeclaration {
            parameters,
            return_type,
            ..
        } = use_value
        else {
            unreachable!();
        };
        let expected = Type::Union(vec![Type::Int, Type::String]);
        assert_eq!(parameters[0].1, expected);
        assert_eq!(return_type, &expected);
        assert!(!statements
            .iter()
            .any(|statement| matches!(statement, Node::TypeAliasDeclaration { .. })));
    }

    #[test]
    fn rejects_recursive_type_aliases() {
        let program = ast::parse(
            r#"
            type A = B;
            type B = A;
            function main(): void {}
            "#,
        );
        let error = prepare_program(&program).unwrap_err();
        assert!(error.contains("cyclic type alias detected: A -> B -> A"));
    }

    #[test]
    fn rejects_type_alias_bound_violations() {
        let program = ast::parse(
            r#"
            trait Writer {
                function write(mut self, value: int): int;
            }

            type WriterValue[T: Writer] = T | string;

            function main(): void {
                value: WriterValue[int] = 1;
            }
            "#,
        );
        let error = prepare_program(&program).unwrap_err();
        assert!(error.contains("type alias `WriterValue` requires `T` to implement trait `Writer`"));
    }

    #[test]
    fn rejects_non_trait_intersection_members() {
        let program = ast::parse(
            r#"
            type Invalid = int & string;

            function main(): void {
                value: Invalid;
            }
            "#,
        );
        let error = prepare_program(&program).unwrap_err();
        assert!(error.contains("intersection member `int` is not a trait"));
    }

    #[test]
    fn accepts_upper_subtype_bound_for_conforming_type() {
        let statements = prepared_statements(
            r#"
            trait Animal {
                function sound(self): int;
            }

            struct Dog {}

            conform Animal for Dog {
                function sound(self): int { return 1; }
            }

            function identity[T <: Animal](value: T): T {
                return value;
            }

            function main(): void {
                dog: Dog = identity(Dog {});
            }
            "#,
        );

        assert!(statements.iter().any(|statement| matches!(
            statement,
            Node::FunctionDeclaration { name, .. } if name == "identity__Dog"
        )));
    }

    #[test]
    fn accepts_trait_intersection_as_upper_bound() {
        let statements = prepared_statements(
            r#"
            trait Animal {}
            trait Serializable {}

            struct Dog {}
            conform Animal for Dog {}
            conform Serializable for Dog {}

            function identity[T <: Animal & Serializable](value: T): T {
                return value;
            }

            function main(): void {
                dog: Dog = identity(Dog {});
            }
            "#,
        );

        assert!(statements.iter().any(|statement| matches!(
            statement,
            Node::FunctionDeclaration { name, .. } if name == "identity__Dog"
        )));
    }

    #[test]
    fn rejects_upper_subtype_bound_violation() {
        let program = ast::parse(
            r#"
            trait Animal {}

            function identity[T <: Animal](value: T): T {
                return value;
            }

            function main(): void {
                value: int = identity(1);
            }
            "#,
        );
        let error = prepare_program(&program).unwrap_err();
        assert!(error.contains(
            "generic function `identity` requires `T` to be a subtype of `Animal`, but found `int`"
        ));
    }

    #[test]
    fn numeric_conversion_is_not_subtyping() {
        let program = ast::parse(
            r#"
            function identity[T <: long](value: T): T {
                return value;
            }

            function main(): void {
                value: int = identity(1);
            }
            "#,
        );
        let error = prepare_program(&program).unwrap_err();
        assert!(error.contains(
            "generic function `identity` requires `T` to be a subtype of `long`, but found `int`"
        ));
    }

    #[test]
    fn checks_subtype_bounds_on_generic_type_aliases() {
        let program = ast::parse(
            r#"
            trait Animal {}
            type Pet[T <: Animal] = T | string;

            function main(): void {
                pet: Pet[int] = 1;
            }
            "#,
        );
        let error = prepare_program(&program).unwrap_err();
        assert!(error.contains(
            "type alias `Pet` requires `T` to be a subtype of `Animal`, but found `int`"
        ));
    }

    #[test]
    fn rejects_lower_subtype_bound_violation_for_explicit_argument() {
        let program = ast::parse(
            r#"
            struct Dog {}
            struct Cat {}

            function keep[T >: Dog](value: T): T {
                return value;
            }

            function main(): void {
                cat: Cat = keep[Cat](Cat {});
            }
            "#,
        );
        let error = prepare_program(&program).unwrap_err();
        assert!(error.contains(
            "generic function `keep` requires `T` to be a supertype of `Dog`, but found `Cat`"
        ));
    }

    #[test]
    fn widens_direct_generic_lower_constraints_to_union() {
        let statements = prepared_statements(
            r#"
            struct Dog {}
            struct Cat {}

            function choose[T](left: T, right: T, first: bool): T {
                if (first) {
                    return left;
                }
                return right;
            }

            function main(): void {
                pet: Dog | Cat = choose(Dog {}, Cat {}, true);
            }
            "#,
        );

        assert!(statements.iter().any(|statement| matches!(
            statement,
            Node::FunctionDeclaration { name, .. }
                if name == "choose__union_Cat_or_Dog"
        )));
    }

    #[test]
    fn inferred_lower_bound_can_reference_another_type_parameter() {
        let statements = prepared_statements(
            r#"
            struct Dog {}
            struct Cat {}

            function choose[A, B >: A](left: A, right: B, first: bool): B {
                if (first) {
                    return left;
                }
                return right;
            }

            function main(): void {
                pet: Dog | Cat = choose(Dog {}, Cat {}, true);
            }
            "#,
        );

        assert!(statements.iter().any(|statement| matches!(
            statement,
            Node::FunctionDeclaration { name, .. }
                if name == "choose__Dog__union_Cat_or_Dog"
        )));
    }
}
