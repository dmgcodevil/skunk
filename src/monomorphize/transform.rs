//! AST transformation, generic inference, and type unification.

use super::*;

struct ExpressionTransformContext<'a> {
    expected_type: Option<&'a Type>,
    substitutions: &'a HashMap<String, Type>,
    self_type: Option<Type>,
}

impl Monomorphizer {
    /// Transforms one function template into a concrete declaration, seeding
    /// its local type environment with substituted parameters and receiver.
    pub(super) fn transform_named_function(
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

    /// Concretizes a struct's fields and attached functions for one receiver
    /// specialization.
    pub(super) fn transform_struct_decl(
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

    /// Concretizes enum payloads and attached functions for one receiver
    /// specialization.
    pub(super) fn transform_enum_decl(
        &mut self,
        name: &str,
        variants: &[ast::EnumVariant],
        functions: &[Node],
        substitutions: &HashMap<String, Type>,
        self_type: Option<Type>,
    ) -> Result<Node, String> {
        let concrete_self_type = self_type.unwrap_or_else(|| Type::Custom(name.to_string()));
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
                        "unsupported enum member during monomorphization: `{:?}`",
                        other
                    ))
                }
            }
        }

        Ok(Node::EnumDeclaration {
            name: name.to_string(),
            variants: output_variants,
            functions: output_functions,
        })
    }

    /// Rewrites one statement while updating lexical type information and
    /// discovering any generic declarations required by that statement.
    pub(super) fn transform_statement(
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
                let (expression, _) =
                    self.transform_expr(expression, env, None, substitutions, self_type)?;
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
                    ExpressionTransformContext {
                        expected_type: None,
                        substitutions,
                        self_type,
                    },
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
        context: ExpressionTransformContext<'_>,
    ) -> Result<(Node, Type), String> {
        let ExpressionTransformContext {
            substitutions,
            self_type,
            ..
        } = context;
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

    /// Rewrites an expression to concrete types and returns both its prepared
    /// AST node and inferred source-level result type.
    pub(super) fn transform_expr(
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
                if let Some(constructor) = self.transform_inferred_generic_enum_constructor(
                    &substituted,
                    name,
                    arguments,
                    metadata,
                    env,
                    expected_type,
                    substitutions,
                    self_type.clone(),
                )? {
                    return Ok(constructor);
                }
                let internal_type = self.expand_type(&substituted)?;
                let output_type = self.concretize_type(&internal_type)?;
                let mut output_args = Vec::new();
                if name == "size_of" || name == "align_of" {
                    if !arguments.is_empty() {
                        return Err(format!("{} expects no arguments", name));
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
                            return Err(
                                "Window::create expects width, height, and title".to_string()
                            );
                        }
                        let expected_types = [Type::Int, Type::Int, Type::String];
                        for (argument, expected_type) in arguments.iter().zip(expected_types.iter())
                        {
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
                    Type::Custom(color_name) if color_name == "Color" => match name.as_str() {
                        "black" | "white" | "red" | "green" | "blue" => {
                            if !arguments.is_empty() {
                                return Err(format!("Color::{} expects no arguments", name));
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
                    },
                    Type::Custom(keyboard_name)
                        if keyboard_name == "Keyboard" && name == "is_down" =>
                    {
                        if arguments.len() != 2 {
                            return Err("Keyboard::is_down expects window and key".to_string());
                        }
                        let expected_types = [Type::Custom("Window".to_string()), Type::Char];
                        for (argument, expected_type) in arguments.iter().zip(expected_types.iter())
                        {
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
                    Type::Custom(memory_name) if memory_name == "Memory" => match name.as_str() {
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
                    },
                    Type::Custom(bounds_name) if bounds_name == "Bounds" => {
                        if name != "check" {
                            return Err(format!(
                                "unsupported static call during monomorphization: `Bounds::{}`",
                                name
                            ));
                        }
                        if arguments.len() != 2 {
                            return Err("Bounds::check expects index and length".to_string());
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
                        Type::Void
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
                        for (argument, parameter_type) in
                            arguments.iter().zip(parameter_types.iter())
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
                        let payload_types = self
                            .lookup_enum_variant(&internal_type, name)?
                            .unwrap_or_default();
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
                ExpressionTransformContext {
                    expected_type,
                    substitutions,
                    self_type,
                },
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
                                                    call_metadata.span.line,
                                                    call_metadata.span.start,
                                                    method_name
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

                                let (_, parameter_types, return_type) =
                                    self.lookup_method_signature(&current_type, method_name)?;
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
                    UnaryOperator::AddressOf => {
                        match expected_type.map(ast::unwrap_binding_const) {
                            Some(Type::Pointer { .. }) => Type::Pointer {
                                target_type: Box::new(operand_type.clone()),
                            },
                            _ => Type::Reference {
                                target_type: Box::new(operand_type.clone()),
                                mutable: false,
                            },
                        }
                    }
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
                ExpressionTransformContext {
                    expected_type,
                    substitutions,
                    self_type,
                },
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

    pub(super) fn transform_match_pattern(
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

    pub(super) fn lookup_enum_variant(
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

    #[allow(clippy::too_many_arguments)]
    /// Infers an enum constructor's missing generic arguments from its payload
    /// and expected type, then targets the corresponding specialized enum.
    pub(super) fn transform_inferred_generic_enum_constructor(
        &mut self,
        written_type: &Type,
        variant_name: &str,
        arguments: &[Node],
        metadata: &Metadata,
        env: &mut Env,
        expected_type: Option<&Type>,
        outer_substitutions: &HashMap<String, Type>,
        self_type: Option<Type>,
    ) -> Result<Option<(Node, Type)>, String> {
        let Type::Custom(base) = written_type else {
            return Ok(None);
        };
        let Some(template) = self.generic_enums.get(base).cloned() else {
            return Ok(None);
        };
        let Some(variant) = template
            .variants
            .iter()
            .find(|variant| variant.name == variant_name)
            .cloned()
        else {
            return Ok(None);
        };

        if arguments.len() != variant.payload_types.len() {
            return Err(format!(
                "enum variant constructor `{}::{}` expects {} argument(s), got {}",
                base,
                variant_name,
                variant.payload_types.len(),
                arguments.len()
            ));
        }

        let mut inferred = HashMap::<String, Type>::new();
        if let Some(expected_type) = expected_type {
            let expanded_expected = self.expand_type(expected_type)?;
            let expanded_expected =
                ast::unwrap_const_view(ast::unwrap_binding_const(&expanded_expected));
            if let Type::GenericInstance {
                base: expected_base,
                type_arguments,
            } = expanded_expected
            {
                if expected_base == base && type_arguments.len() == template.generic_params.len() {
                    inferred.extend(
                        template
                            .generic_params
                            .iter()
                            .cloned()
                            .zip(type_arguments.iter().cloned()),
                    );
                }
            }
        }

        let mut output_arguments = Vec::new();
        for (argument, payload_type) in arguments.iter().zip(variant.payload_types.iter()) {
            let partially_substituted = self.apply_substitutions(payload_type, &inferred);
            let argument_expected = if contains_unresolved_generic(
                &partially_substituted,
                &template.generic_params,
                &inferred,
            ) {
                None
            } else {
                Some(self.expand_type(&partially_substituted)?)
            };
            let (argument, argument_type) = self.transform_expr(
                argument,
                env,
                argument_expected.as_ref(),
                outer_substitutions,
                self_type.clone(),
            )?;
            let inference_argument_type = if let Some(argument_expected) = &argument_expected {
                if ast::is_numeric_assignable(argument_expected, &argument_type)
                    || self.is_subtype(&argument_type, argument_expected)?
                {
                    argument_expected
                } else {
                    &argument_type
                }
            } else {
                &argument_type
            };
            self.unify_generic_type(
                payload_type,
                inference_argument_type,
                &template.generic_params,
                &mut inferred,
            )?;
            output_arguments.push(argument);
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
                let lower = self.apply_substitutions(lower, &inferred);
                if contains_unresolved_generic(&lower, &template.generic_params, &inferred) {
                    continue;
                }
                let previous = inferred.get(param).cloned();
                self.merge_inferred_lower(param, &lower, &mut inferred)?;
                changed |= inferred.get(param) != previous.as_ref();
            }
            if !changed {
                break;
            }
        }

        let missing = template
            .generic_params
            .iter()
            .filter(|param| !inferred.contains_key(*param))
            .cloned()
            .collect::<Vec<_>>();
        if !missing.is_empty() {
            let arguments = missing
                .iter()
                .map(|param| format!("`{}`", param))
                .collect::<Vec<_>>()
                .join(", ");
            return Err(format!(
                "could not infer type argument{} {} for generic enum constructor `{}::{}`; add an expected `{}` type or write explicit type arguments",
                if missing.len() == 1 { "" } else { "s" },
                arguments,
                base,
                variant_name,
                base
            ));
        }

        self.check_generic_bounds(
            &template.generic_bounds,
            &template.subtype_bounds,
            &inferred,
            &format!("generic enum `{}`", base),
        )?;
        let internal_type = Type::GenericInstance {
            base: base.clone(),
            type_arguments: template
                .generic_params
                .iter()
                .map(|param| {
                    inferred
                        .get(param)
                        .expect("all generic enum arguments were inferred")
                        .clone()
                })
                .collect(),
        };
        let output_type = self.concretize_type(&internal_type)?;
        Ok(Some((
            Node::StaticFunctionCall {
                _type: output_type,
                name: variant_name.to_string(),
                arguments: output_arguments,
                metadata: metadata.clone(),
            },
            internal_type,
        )))
    }

    pub(super) fn lookup_enum_variant_payload_types(
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

    pub(super) fn lookup_struct_pattern_bindings(
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

    /// Resolves a call, infers generic arguments when omitted, and rewrites the
    /// callee to the concrete symbol selected for code generation.
    fn transform_function_call(
        &mut self,
        name: &str,
        explicit_type_arguments: &[Type],
        arguments: &[Vec<Node>],
        metadata: &Metadata,
        env: &mut Env,
        context: ExpressionTransformContext<'_>,
    ) -> Result<(Node, Type), String> {
        let ExpressionTransformContext {
            expected_type,
            substitutions,
            self_type,
        } = context;
        if let Some(template) = self.generic_functions.get(name).cloned() {
            let mut output_args = Vec::new();
            let mut argument_types = Vec::new();
            for args in arguments {
                let transformed = args
                    .iter()
                    .map(|arg| {
                        self.transform_expr(arg, env, None, substitutions, self_type.clone())
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                output_args.push(transformed.iter().map(|(node, _)| node.clone()).collect());
                argument_types.push(
                    transformed
                        .into_iter()
                        .map(|(_, sk_type)| sk_type)
                        .collect::<Vec<_>>(),
                );
            }
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
                            .map(|arg| {
                                self.concretize_type(&self.apply_substitutions(arg, substitutions))
                            })
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

        let mut output_args = Vec::new();
        let mut argument_types = Vec::new();
        let mut current_signature = signature.clone();
        for args in arguments {
            let (expected_parameters, next_signature) = match &current_signature {
                Type::Function {
                    parameters,
                    return_type,
                } => (parameters.clone(), return_type.deref().clone()),
                _ => (Vec::new(), current_signature.clone()),
            };
            let transformed = args
                .iter()
                .enumerate()
                .map(|(index, arg)| {
                    self.transform_expr(
                        arg,
                        env,
                        expected_parameters.get(index),
                        substitutions,
                        self_type.clone(),
                    )
                })
                .collect::<Result<Vec<_>, String>>()?;
            output_args.push(transformed.iter().map(|(node, _)| node.clone()).collect());
            argument_types.push(
                transformed
                    .into_iter()
                    .map(|(_, sk_type)| sk_type)
                    .collect::<Vec<_>>(),
            );
            current_signature = next_signature;
        }

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

    /// Infers a complete substitution map from explicit arguments, parameter
    /// types, argument types, and the optional expected return type.
    pub(super) fn infer_generic_function_arguments(
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
                if contains_unresolved_generic(&lower, &template.generic_params, &substitutions) {
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

    pub(super) fn direct_generic_candidate(
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
            Type::Const { inner } => {
                self.direct_generic_candidate(inner, ast::unwrap_const_view(actual), generic_params)
            }
            Type::Custom(name) if generic_params.iter().any(|param| param == name) => {
                Some((name.clone(), actual.clone()))
            }
            _ => None,
        }
    }

    pub(super) fn merge_inferred_lower(
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

    /// Unifies a generic pattern with an observed type and accumulates the
    /// resulting substitutions, rejecting inconsistent inferences.
    pub(super) fn unify_generic_type(
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
}
