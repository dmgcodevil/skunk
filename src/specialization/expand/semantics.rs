//! Type expansion, trait validation, bounds, and subtype relations.

use super::*;

impl Monomorphizer {
    /// Expands transparent aliases and normalizes compound types before they
    /// participate in inference, validation, or specialization.
    pub(super) fn expand_type(&mut self, sk_type: &Type) -> Result<Type, String> {
        self.expand_type_inner(sk_type, &mut Vec::new())
    }

    pub(super) fn expand_type_inner(
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
                        Type::Void => return Err("union types cannot contain `void`".to_string()),
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

    /// Validates every trait hierarchy and computes inherited method sets to
    /// detect cycles and duplicate inherited declarations early.
    pub(super) fn validate_traits(&self) -> Result<(), String> {
        for trait_name in self.traits.keys() {
            let mut visiting = Vec::new();
            let _ = self.collect_trait_methods(trait_name, &mut visiting)?;
        }
        Ok(())
    }

    pub(super) fn collect_trait_methods(
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

    pub(super) fn collect_trait_ancestors(
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

    pub(super) fn trait_extends(&self, child: &str, ancestor: &str) -> Result<bool, String> {
        if child == ancestor {
            return Ok(true);
        }
        Ok(self
            .collect_trait_ancestors(child, &mut Vec::new())?
            .iter()
            .any(|name| name == ancestor))
    }

    pub(super) fn implied_trait_names(&self, trait_name: &str) -> Result<Vec<String>, String> {
        let mut names = vec![trait_name.to_string()];
        names.extend(self.collect_trait_ancestors(trait_name, &mut Vec::new())?);
        Ok(names)
    }

    /// Validates source conformances and records concrete trait relationships
    /// that later bound and subtype checks can query directly.
    pub(super) fn validate_impls(&mut self) -> Result<(), String> {
        let impls = self.impls.clone();
        for impl_block in impls {
            self.validate_impl_target_type(&impl_block.target_type, &impl_block.generic_params)?;
            let target_key = ast::type_to_string(&impl_block.target_type);
            for trait_type in impl_block.trait_types {
                let (trait_template, trait_substitutions) =
                    self.resolve_trait_reference(&trait_type)?;
                self.validate_trait_implementation(
                    &trait_template,
                    &trait_substitutions,
                    &trait_type,
                    &impl_block.target_type,
                )?;
                if impl_block.generic_params.is_empty() {
                    let concrete_trait_name = self.ensure_trait_for_type(&trait_type)?;
                    let implied_traits =
                        self.collect_trait_ancestors(&concrete_trait_name, &mut Vec::new())?;
                    let implemented = self
                        .implemented_traits
                        .entry(target_key.clone())
                        .or_default();
                    if !implemented.insert(concrete_trait_name.clone()) {
                        return Err(format!(
                            "duplicate impl of trait `{}` for `{}`",
                            concrete_trait_name, target_key
                        ));
                    }
                    let primary_key = format!("{}=>{}", concrete_trait_name, target_key);
                    self.generated_impl_keys.insert(primary_key);
                    if !matches!(&trait_type, Type::Custom(name) if name == &concrete_trait_name) {
                        self.generated_impls.push(Node::ImplDeclaration {
                            generic_params: Vec::new(),
                            generic_bounds: HashMap::new(),
                            subtype_bounds: HashMap::new(),
                            trait_types: vec![Type::Custom(concrete_trait_name.clone())],
                            target_type: impl_block.target_type.clone(),
                        });
                    }
                    for implied_trait in implied_traits {
                        implemented.insert(implied_trait.clone());
                        let implied_key = format!("{}=>{}", implied_trait, target_key);
                        if self.generated_impl_keys.insert(implied_key) {
                            self.generated_impls.push(Node::ImplDeclaration {
                                generic_params: Vec::new(),
                                generic_bounds: HashMap::new(),
                                subtype_bounds: HashMap::new(),
                                trait_types: vec![Type::Custom(implied_trait)],
                                target_type: impl_block.target_type.clone(),
                            });
                        }
                    }
                }
            }
        }
        Ok(())
    }

    pub(super) fn resolve_trait_reference(
        &self,
        trait_type: &Type,
    ) -> Result<(TraitTemplate, HashMap<String, Type>), String> {
        let (name, arguments) = match trait_type {
            Type::Custom(name) => (name, &[][..]),
            Type::GenericInstance {
                base,
                type_arguments,
            } => (base, type_arguments.as_slice()),
            other => {
                return Err(format!(
                    "impl requires a trait type, found `{}`",
                    ast::type_to_string(other)
                ))
            }
        };
        let template = self
            .traits
            .get(name)
            .cloned()
            .ok_or_else(|| format!("unknown trait `{}`", name))?;
        if template.generic_params.len() != arguments.len() {
            return Err(format!(
                "trait `{}` expects {} type argument{}, got {}",
                name,
                template.generic_params.len(),
                if template.generic_params.len() == 1 {
                    ""
                } else {
                    "s"
                },
                arguments.len()
            ));
        }
        let substitutions = template
            .generic_params
            .iter()
            .cloned()
            .zip(arguments.iter().cloned())
            .collect();
        Ok((template, substitutions))
    }

    pub(super) fn validate_impl_target_type(
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
            Type::Array { elem_type, .. } => {
                self.validate_impl_target_type(elem_type, generic_params)
            }
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
                if !(self.generic_structs.contains_key(base)
                    || self.generic_enums.contains_key(base))
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

    /// Checks one conformance against its trait contract, including inherited
    /// requirements, receiver types, and default method synthesis.
    pub(super) fn validate_trait_implementation(
        &mut self,
        trait_template: &TraitTemplate,
        trait_substitutions: &HashMap<String, Type>,
        trait_type: &Type,
        target_type: &Type,
    ) -> Result<(), String> {
        let trait_display = ast::type_to_string(trait_type);
        for mut method in self.collect_trait_methods(&trait_template.name, &mut Vec::new())? {
            method.parameters = method
                .parameters
                .into_iter()
                .map(|(name, sk_type)| {
                    (
                        name,
                        self.apply_substitutions(&sk_type, trait_substitutions),
                    )
                })
                .collect();
            method.return_type = self.apply_substitutions(&method.return_type, trait_substitutions);
            let Some((_, expected_receiver_type)) = method.parameters.first() else {
                return Err(format!(
                    "trait `{}` method `{}` must declare `self` as its first parameter",
                    trait_display, method.name
                ));
            };
            if !ast::is_self_type(expected_receiver_type) {
                return Err(format!(
                    "trait `{}` method `{}` must declare `self` as its first parameter",
                    trait_display, method.name
                ));
            }
            let (actual_receiver_type, actual_parameters, actual_return_type) = match self
                .lookup_method_signature(target_type, &method.name)
            {
                Ok(signature) => signature,
                Err(_) => {
                    if method.default_body.is_some() {
                        if !trait_template.generic_params.is_empty() {
                            return Err(format!(
                                    "generic trait `{}` default methods are not supported yet; implement `{}` explicitly",
                                    trait_template.name, method.name
                                ));
                        }
                        self.synthesize_trait_default_method(trait_template, target_type, &method)?;
                        self.lookup_method_signature(target_type, &method.name)
                            .map_err(|_| {
                                format!(
                                    "type `{}` does not implement required trait method `{}.{}`",
                                    ast::type_to_string(target_type),
                                    trait_display,
                                    method.name
                                )
                            })?
                    } else {
                        return Err(format!(
                            "type `{}` does not implement required trait method `{}.{}`",
                            ast::type_to_string(target_type),
                            trait_display,
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
                    trait_display,
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

    pub(super) fn synthesize_trait_default_method(
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

    pub(super) fn check_trait_bounds(
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
                        let implemented =
                            self.type_implements_trait(actual_type, constraint_name)?;
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

    /// Verifies trait, shape, upper, and lower bounds after generic parameters
    /// have been mapped to candidate concrete types.
    pub(super) fn check_generic_bounds(
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

    /// Evaluates Skunk's subtype relation after alias expansion, including
    /// unions, intersections, nominal traits, and structural shapes.
    pub(super) fn is_subtype(&mut self, subtype: &Type, supertype: &Type) -> Result<bool, String> {
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
            (subtype, supertype @ Type::GenericInstance { base, .. })
                if self.traits.contains_key(base) =>
            {
                self.type_implements_trait_reference(subtype, supertype)
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

    pub(super) fn constraint_kind(&self, name: &str) -> Option<ConstraintKind> {
        if self.traits.contains_key(name) {
            Some(ConstraintKind::Trait)
        } else if self.shapes.contains_key(name) {
            Some(ConstraintKind::Shape)
        } else {
            None
        }
    }

    pub(super) fn type_implements_trait(
        &mut self,
        actual_type: &Type,
        trait_name: &str,
    ) -> Result<bool, String> {
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
                .trait_types
                .iter()
                .any(|trait_type| match trait_type {
                    Type::Custom(name) => self.trait_extends(name, trait_name).unwrap_or(false),
                    Type::GenericInstance { base, .. } => {
                        self.trait_extends(base, trait_name).unwrap_or(false)
                    }
                    _ => false,
                })
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

    /// Resolves a possibly generic trait reference and determines whether a
    /// concrete type has a matching direct or generated conformance.
    pub(super) fn type_implements_trait_reference(
        &mut self,
        actual_type: &Type,
        required_trait_type: &Type,
    ) -> Result<bool, String> {
        let required = self.expand_type(required_trait_type)?;
        let mut matched = false;
        for impl_block in self.impls.clone() {
            let mut substitutions = HashMap::new();
            if impl_block.generic_params.is_empty() {
                if impl_block.target_type != *actual_type {
                    continue;
                }
            } else if self
                .unify_generic_type(
                    &impl_block.target_type,
                    actual_type,
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

            let mut implements_required = false;
            for trait_type in &impl_block.trait_types {
                let instantiated = self.apply_substitutions(trait_type, &substitutions);
                if self.expand_type(&instantiated)? == required {
                    implements_required = true;
                    break;
                }
            }
            if !implements_required {
                continue;
            }
            if matched {
                return Err(format!(
                    "multiple impls of trait `{}` match `{}`",
                    ast::type_to_string(&required),
                    ast::type_to_string(actual_type)
                ));
            }
            matched = true;
        }
        Ok(matched)
    }

    pub(super) fn type_satisfies_shape(
        &mut self,
        actual_type: &Type,
        shape_name: &str,
    ) -> Result<bool, String> {
        let shape = self
            .shapes
            .get(shape_name)
            .ok_or_else(|| format!("unknown shape `{}`", shape_name))?
            .clone();
        self.validate_shape_satisfaction(&shape, actual_type)
            .map(|_| true)
    }

    /// Produces a detailed error when a concrete type does not provide the
    /// method surface required by a structural shape.
    pub(super) fn validate_shape_satisfaction(
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

    /// Materializes generic conformances that become applicable once a source
    /// type has been specialized to a concrete runtime type.
    pub(super) fn ensure_runtime_impls_for_type(
        &mut self,
        source_type: &Type,
        concrete_type: &Type,
    ) -> Result<(), String> {
        let concrete_key = ast::type_to_string(concrete_type);
        for impl_block in self.impls.clone() {
            let mut substitutions = HashMap::new();
            if impl_block.generic_params.is_empty() {
                if impl_block.target_type != *source_type
                    && impl_block.target_type != *concrete_type
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

            for trait_type in impl_block.trait_types {
                let trait_type = self.apply_substitutions(&trait_type, &substitutions);
                let trait_name = self.ensure_trait_for_type(&trait_type)?;
                for implied_trait in self.implied_trait_names(&trait_name)? {
                    let key = format!("{}=>{}", implied_trait, concrete_key);
                    if !self.generated_impl_keys.insert(key) {
                        continue;
                    }
                    self.generated_impls.push(Node::ImplDeclaration {
                        generic_params: Vec::new(),
                        generic_bounds: HashMap::new(),
                        subtype_bounds: HashMap::new(),
                        trait_types: vec![Type::Custom(implied_trait)],
                        target_type: concrete_type.clone(),
                    });
                }
            }
        }
        Ok(())
    }
}
