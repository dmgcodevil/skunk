//! Concrete declaration generation and specialized symbol lookup.

use super::*;

impl Monomorphizer {
    /// Generates and memoizes one concrete function body for a complete set of
    /// generic substitutions.
    pub(super) fn ensure_specialized_function(
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
        if self.generated_functions.contains(&symbol_name) {
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
        self.function_stack.remove(&symbol_name);
        Ok(symbol_name)
    }

    /// Returns the runtime struct symbol for a nominal type, generating a
    /// concrete generic struct declaration on first use.
    pub(super) fn ensure_struct_for_type(&mut self, sk_type: &Type) -> Result<String, String> {
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
                if self.generated_structs.contains(&symbol_name) {
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

    /// Returns the runtime enum symbol for a nominal type, generating a
    /// concrete generic enum declaration on first use.
    pub(super) fn ensure_enum_for_type(&mut self, sk_type: &Type) -> Result<String, String> {
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
                if self.generated_enums.contains(&symbol_name) {
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
                let node = self.transform_enum_decl(
                    &symbol_name,
                    &template.variants,
                    &template.functions,
                    &substitutions,
                    Some(sk_type.clone()),
                )?;
                self.generated_enums.insert(symbol_name.clone(), node);
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

    /// Returns the runtime trait symbol for a trait reference, specializing its
    /// inherited methods and bounds when it has generic arguments.
    pub(super) fn ensure_trait_for_type(&mut self, sk_type: &Type) -> Result<String, String> {
        match sk_type {
            Type::Custom(name) => {
                let template = self
                    .traits
                    .get(name)
                    .ok_or_else(|| format!("unknown trait `{}`", name))?;
                if !template.generic_params.is_empty() {
                    return Err(format!(
                        "generic trait `{}` expects {} type argument{}",
                        name,
                        template.generic_params.len(),
                        if template.generic_params.len() == 1 {
                            ""
                        } else {
                            "s"
                        }
                    ));
                }
                Ok(name.clone())
            }
            Type::GenericInstance {
                base,
                type_arguments,
            } => {
                let template = self
                    .traits
                    .get(base)
                    .cloned()
                    .ok_or_else(|| format!("unknown generic trait `{}`", base))?;
                if template.generic_params.len() != type_arguments.len() {
                    return Err(format!(
                        "generic trait `{}` expects {} type arguments, got {}",
                        base,
                        template.generic_params.len(),
                        type_arguments.len()
                    ));
                }
                let concrete_arguments = type_arguments
                    .iter()
                    .map(|argument| self.concretize_type(argument))
                    .collect::<Result<Vec<_>, _>>()?;
                let symbol_name = specialized_struct_name(base, &concrete_arguments);
                if self.generated_traits.contains(&symbol_name) {
                    return Ok(symbol_name);
                }
                if !self.trait_stack.insert(symbol_name.clone()) {
                    return Ok(symbol_name);
                }
                let substitutions = template
                    .generic_params
                    .iter()
                    .cloned()
                    .zip(concrete_arguments)
                    .collect::<HashMap<_, _>>();
                self.check_generic_bounds(
                    &template.generic_bounds,
                    &template.subtype_bounds,
                    &substitutions,
                    &format!("generic trait `{}`", base),
                )?;
                for supertrait in &template.supertraits {
                    let supertrait_template = self
                        .traits
                        .get(supertrait)
                        .ok_or_else(|| format!("unknown trait `{}`", supertrait))?;
                    if !supertrait_template.generic_params.is_empty() {
                        return Err(format!(
                            "generic trait `{}` must provide type arguments for generic supertrait `{}`",
                            base, supertrait
                        ));
                    }
                }
                let methods = template
                    .methods
                    .iter()
                    .map(|method| {
                        Ok(ast::TraitMethodSignature {
                            name: method.name.clone(),
                            parameters: method
                                .parameters
                                .iter()
                                .map(|(name, sk_type)| {
                                    let substituted =
                                        self.apply_substitutions(sk_type, &substitutions);
                                    self.concretize_type(&substituted)
                                        .map(|sk_type| (name.clone(), sk_type))
                                })
                                .collect::<Result<Vec<_>, String>>()?,
                            return_type: self.concretize_type(
                                &self.apply_substitutions(&method.return_type, &substitutions),
                            )?,
                            // Implementations receive synthesized/default methods
                            // before specialization. The runtime trait only needs
                            // its concrete dispatch signature.
                            default_body: None,
                        })
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                let node = Node::TraitDeclaration {
                    name: symbol_name.clone(),
                    generic_params: Vec::new(),
                    generic_bounds: HashMap::new(),
                    subtype_bounds: HashMap::new(),
                    supertraits: template.supertraits.clone(),
                    methods: methods.clone(),
                };
                self.traits.insert(
                    symbol_name.clone(),
                    TraitTemplate {
                        name: symbol_name.clone(),
                        generic_params: Vec::new(),
                        generic_bounds: HashMap::new(),
                        subtype_bounds: HashMap::new(),
                        supertraits: template.supertraits,
                        methods,
                    },
                );
                self.generated_traits.insert(symbol_name.clone(), node);
                self.trait_stack.remove(&symbol_name);
                Ok(symbol_name)
            }
            other => Err(format!(
                "expected a trait type, found `{}`",
                ast::type_to_string(other)
            )),
        }
    }

    /// Dispatches nominal specialization to the matching struct, enum, or
    /// trait template and returns its concrete symbol.
    pub(super) fn ensure_nominal_type(&mut self, sk_type: &Type) -> Result<String, String> {
        match sk_type {
            Type::Custom(name) => Ok(name.clone()),
            Type::GenericInstance { base, .. } if self.generic_structs.contains_key(base) => {
                self.ensure_struct_for_type(sk_type)
            }
            Type::GenericInstance { base, .. } if self.generic_enums.contains_key(base) => {
                self.ensure_enum_for_type(sk_type)
            }
            Type::GenericInstance { base, .. } if self.traits.contains_key(base) => {
                self.ensure_trait_for_type(sk_type)
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

    pub(super) fn lookup_struct_field_type(
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

    pub(super) fn lookup_method_signature(
        &mut self,
        receiver_type: &Type,
        method_name: &str,
    ) -> Result<(Type, Vec<Type>, Type), String> {
        let result = match receiver_type {
            Type::Custom(name) => {
                if let Some(template) = self.concrete_structs.get(name) {
                    let function =
                        attached_function_signature(&template.functions, method_name, true)
                            .ok_or_else(|| {
                                format!("unknown method `{}` on `{}`", method_name, name)
                            })?;
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
                } else if let Some(template) = self.concrete_enums.get(name) {
                    let function =
                        attached_function_signature(&template.functions, method_name, true)
                            .ok_or_else(|| {
                                format!("unknown method `{}` on `{}`", method_name, name)
                            })?;
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
                        .ok_or_else(|| {
                            format!("unknown method `{}` on trait `{}`", method_name, name)
                        })?;
                    let receiver = method
                        .parameters
                        .first()
                        .map(|(_, sk_type)| sk_type.clone())
                        .ok_or_else(|| {
                            format!(
                                "method `{}` on trait `{}` is missing self",
                                method_name, name
                            )
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
                    Err(format!("unknown nominal type or trait `{}`", name))
                }
            }
            Type::GenericInstance {
                base,
                type_arguments,
            } => {
                if let Some(template) = self.traits.get(base).cloned() {
                    if template.generic_params.len() != type_arguments.len() {
                        return Err(format!(
                            "generic trait `{}` expects {} type arguments, got {}",
                            base,
                            template.generic_params.len(),
                            type_arguments.len()
                        ));
                    }
                    let substitutions = template
                        .generic_params
                        .iter()
                        .cloned()
                        .zip(type_arguments.iter().cloned())
                        .collect::<HashMap<_, _>>();
                    let method = self
                        .collect_trait_methods(base, &mut Vec::new())?
                        .into_iter()
                        .find(|method| method.name == method_name)
                        .ok_or_else(|| {
                            format!("unknown method `{}` on trait `{}`", method_name, base)
                        })?;
                    let receiver = method
                        .parameters
                        .first()
                        .map(|(_, sk_type)| self.apply_substitutions(sk_type, &substitutions))
                        .ok_or_else(|| {
                            format!(
                                "method `{}` on trait `{}` is missing self",
                                method_name, base
                            )
                        })?;
                    return Ok((
                        receiver,
                        method
                            .parameters
                            .iter()
                            .skip(1)
                            .map(|(_, sk_type)| self.apply_substitutions(sk_type, &substitutions))
                            .collect(),
                        self.apply_substitutions(&method.return_type, &substitutions),
                    ));
                }
                let (generic_params, functions) =
                    if let Some(template) = self.generic_structs.get(base) {
                        (&template.generic_params, &template.functions)
                    } else if let Some(template) = self.generic_enums.get(base) {
                        (&template.generic_params, &template.functions)
                    } else {
                        return Err(format!("unknown generic nominal type `{}`", base));
                    };
                let substitutions = generic_params
                    .iter()
                    .cloned()
                    .zip(type_arguments.iter().cloned())
                    .collect::<HashMap<_, _>>();
                let function = attached_function_signature(functions, method_name, true)
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
                    if let Some(method) = methods
                        .into_iter()
                        .find(|method| method.name == method_name)
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
                "method lookup requires a nominal receiver, found `{}`",
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

    pub(super) fn lookup_static_function_signature(
        &mut self,
        target_type: &Type,
        function_name: &str,
    ) -> Result<(Vec<Type>, Type), String> {
        let result = match target_type {
            Type::Custom(name) => {
                let functions = if let Some(template) = self.concrete_structs.get(name) {
                    &template.functions
                } else if let Some(template) = self.concrete_enums.get(name) {
                    &template.functions
                } else {
                    return Err(format!("unknown nominal type `{}`", name));
                };
                let function = attached_function_signature(functions, function_name, false)
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
                let (generic_params, functions) =
                    if let Some(template) = self.generic_structs.get(base) {
                        (&template.generic_params, &template.functions)
                    } else if let Some(template) = self.generic_enums.get(base) {
                        (&template.generic_params, &template.functions)
                    } else {
                        return Err(format!("unknown generic nominal type `{}`", base));
                    };
                let substitutions = generic_params
                    .iter()
                    .cloned()
                    .zip(type_arguments.iter().cloned())
                    .collect::<HashMap<_, _>>();
                let function = attached_function_signature(functions, function_name, false)
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
                "static function lookup requires a nominal type, found `{}`",
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

    pub(super) fn apply_substitutions(
        &self,
        sk_type: &Type,
        substitutions: &HashMap<String, Type>,
    ) -> Type {
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

    /// Fully expands aliases and replaces generic nominal instances with the
    /// concrete symbols emitted for the prepared program.
    pub(super) fn concretize_type(&mut self, sk_type: &Type) -> Result<Type, String> {
        let expanded = self.expand_type(sk_type)?;
        self.concretize_expanded_type(&expanded)
    }

    pub(super) fn concretize_expanded_type(&mut self, sk_type: &Type) -> Result<Type, String> {
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

    pub(super) fn literal_type(
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
            Literal::String(_) => Ok(Type::String),
            Literal::Boolean(_) => Ok(Type::Boolean),
            Literal::Char(_) => Ok(Type::Char),
        }
    }
}
