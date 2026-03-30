use crate::ast::{self, Node};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

pub fn load_program(entry_path: &Path) -> Result<Node, String> {
    let entry_path = fs::canonicalize(entry_path)
        .map_err(|err| format!("failed to resolve `{}`: {}", entry_path.display(), err))?;
    let module_root = entry_path
        .parent()
        .ok_or_else(|| format!("`{}` has no parent directory", entry_path.display()))?
        .to_path_buf();
    let mut loader = ProgramLoader::new(module_root);
    let mut statements = loader.load_file(&entry_path, None, true)?;
    statements.push(Node::EOI);
    Ok(Node::Program { statements })
}

struct ProgramLoader {
    module_root: PathBuf,
    visited: HashSet<PathBuf>,
    loading: HashSet<PathBuf>,
}

impl ProgramLoader {
    fn new(module_root: PathBuf) -> Self {
        Self {
            module_root,
            visited: HashSet::new(),
            loading: HashSet::new(),
        }
    }

    fn load_file(
        &mut self,
        file_path: &Path,
        expected_module: Option<&str>,
        is_entry: bool,
    ) -> Result<Vec<Node>, String> {
        let file_path = fs::canonicalize(file_path)
            .map_err(|err| format!("failed to resolve `{}`: {}", file_path.display(), err))?;

        if self.visited.contains(&file_path) {
            return Ok(Vec::new());
        }

        if !self.loading.insert(file_path.clone()) {
            return Err(format!(
                "cyclic import detected while loading `{}`",
                file_path.display()
            ));
        }

        let contents = fs::read_to_string(&file_path)
            .map_err(|err| format!("failed to read `{}`: {}", file_path.display(), err))?;
        let program = ast::parse(&contents);
        let Node::Program { statements } = program else {
            unreachable!("parse always returns a program node")
        };

        let declared_module = extract_module_name(&statements)?;
        if let Some(expected_module) = expected_module {
            match declared_module.as_deref() {
                Some(actual) if actual == expected_module => {}
                Some(actual) => {
                    return Err(format!(
                        "module declaration mismatch in `{}`: expected `{}`, found `{}`",
                        file_path.display(),
                        expected_module,
                        actual
                    ))
                }
                None => {
                    return Err(format!(
                        "imported file `{}` must declare `module {};`",
                        file_path.display(),
                        expected_module
                    ))
                }
            }
        } else if !is_entry && declared_module.is_none() {
            return Err(format!(
                "imported file `{}` is missing a module declaration",
                file_path.display()
            ));
        }

        let mut output = Vec::new();
        let statements = ModuleNormalizer::new(declared_module.clone(), !is_entry, &statements)?
            .normalize(statements)?;
        for statement in statements {
            match statement {
                Node::Module { .. } => {}
                Node::Import { name } => {
                    let import_path = self.module_path(&name);
                    let imported = self.load_file(&import_path, Some(&name), false)?;
                    output.extend(imported);
                }
                Node::EOI => {}
                other => output.push(other),
            }
        }

        self.loading.remove(&file_path);
        self.visited.insert(file_path);
        Ok(output)
    }

    fn module_path(&self, module_name: &str) -> PathBuf {
        let mut path = self.module_root.clone();
        for segment in module_name.split('.') {
            path.push(segment);
        }
        path.set_extension("skunk");
        path
    }
}

struct ModuleNormalizer {
    value_renames: HashMap<String, String>,
    type_renames: HashMap<String, String>,
}

impl ModuleNormalizer {
    fn new(
        module_name: Option<String>,
        rename_private: bool,
        statements: &[Node],
    ) -> Result<Self, String> {
        let mut value_renames = HashMap::new();
        let mut type_renames = HashMap::new();
        let has_explicit_exports = statements
            .iter()
            .any(|statement| matches!(statement, Node::Export { .. }));

        if rename_private && has_explicit_exports {
            let module_name = module_name
                .as_deref()
                .ok_or_else(|| "imported module is missing a module name".to_string())?;
            for statement in statements {
                let (declaration, exported) = match statement {
                    Node::Export { declaration } => (declaration.as_ref(), true),
                    other => (other, false),
                };
                if exported {
                    validate_export_target(declaration)?;
                    continue;
                }

                if let Some((kind, name)) = top_level_decl_name(declaration) {
                    let mangled = mangle_private_name(module_name, &name);
                    match kind {
                        TopLevelDeclKind::Value => {
                            value_renames.insert(name, mangled);
                        }
                        TopLevelDeclKind::Type => {
                            type_renames.insert(name, mangled);
                        }
                    }
                }
            }
        }

        Ok(Self {
            value_renames,
            type_renames,
        })
    }

    fn normalize(&self, statements: Vec<Node>) -> Result<Vec<Node>, String> {
        let mut output = Vec::new();
        for statement in statements {
            let (declaration, exported) = match statement {
                Node::Export { declaration } => (*declaration, true),
                other => (other, false),
            };
            if exported {
                validate_export_target(&declaration)?;
            }
            output.push(self.rename_top_level(declaration, exported)?);
        }
        Ok(output)
    }

    fn rename_top_level(&self, node: Node, exported: bool) -> Result<Node, String> {
        let mut value_scopes = vec![HashSet::new()];
        let mut type_scopes = vec![HashSet::new()];
        self.rename_statement(node, &mut value_scopes, &mut type_scopes, true, exported)
    }

    fn rename_statement(
        &self,
        node: Node,
        value_scopes: &mut Vec<HashSet<String>>,
        type_scopes: &mut Vec<HashSet<String>>,
        top_level: bool,
        exported: bool,
    ) -> Result<Node, String> {
        Ok(match node {
            Node::Module { .. } | Node::Import { .. } | Node::EOI => node,
            Node::Export { .. } => {
                return Err("`export` is only allowed at module scope".to_string());
            }
            Node::Block { statements } => {
                value_scopes.push(HashSet::new());
                let statements =
                    self.rename_statement_list(statements, value_scopes, type_scopes, false)?;
                value_scopes.pop();
                Node::Block { statements }
            }
            Node::VariableDeclaration {
                var_type,
                name,
                value,
                metadata,
            } => {
                let var_type = self.rename_type(var_type, value_scopes, type_scopes)?;
                let value = value
                    .map(|value| {
                        self.rename_expr(*value, value_scopes, type_scopes)
                            .map(Box::new)
                    })
                    .transpose()?;
                let renamed_name = if top_level && !exported {
                    self.value_renames
                        .get(&name)
                        .cloned()
                        .unwrap_or(name.clone())
                } else {
                    name.clone()
                };
                if !top_level {
                    value_scopes
                        .last_mut()
                        .expect("local scope exists")
                        .insert(name.clone());
                }
                Node::VariableDeclaration {
                    var_type,
                    name: renamed_name,
                    value,
                    metadata,
                }
            }
            Node::StructDestructure {
                struct_type,
                fields,
                value,
                metadata,
            } => {
                let struct_type = self.rename_type(struct_type, value_scopes, type_scopes)?;
                let value = Box::new(self.rename_expr(*value, value_scopes, type_scopes)?);
                if !top_level {
                    let scope = value_scopes.last_mut().expect("local scope exists");
                    for field in &fields {
                        scope.insert(field.binding.clone());
                    }
                }
                Node::StructDestructure {
                    struct_type,
                    fields,
                    value,
                    metadata,
                }
            }
            Node::FunctionDeclaration {
                name,
                parameters,
                return_type,
                body,
                lambda,
            } => {
                if lambda {
                    let mut local_scope = HashSet::new();
                    let parameters = parameters
                        .into_iter()
                        .map(|(param_name, param_type)| {
                            if !ast::is_self_type(&param_type) {
                                local_scope.insert(param_name.clone());
                            }
                            self.rename_type(param_type, value_scopes, type_scopes)
                                .map(|param_type| (param_name, param_type))
                        })
                        .collect::<Result<Vec<_>, String>>()?;
                    let return_type = self.rename_type(return_type, value_scopes, type_scopes)?;
                    value_scopes.push(local_scope);
                    let body =
                        self.rename_statement_list(body, value_scopes, type_scopes, false)?;
                    value_scopes.pop();
                    Node::FunctionDeclaration {
                        name,
                        parameters,
                        return_type,
                        body,
                        lambda,
                    }
                } else {
                    let renamed_name = if top_level && !exported {
                        self.value_renames
                            .get(&name)
                            .cloned()
                            .unwrap_or(name.clone())
                    } else {
                        name.clone()
                    };
                    let mut local_scope = HashSet::new();
                    let parameters = parameters
                        .into_iter()
                        .map(|(param_name, param_type)| {
                            if !ast::is_self_type(&param_type) {
                                local_scope.insert(param_name.clone());
                            }
                            self.rename_type(param_type, value_scopes, type_scopes)
                                .map(|param_type| (param_name, param_type))
                        })
                        .collect::<Result<Vec<_>, String>>()?;
                    let return_type = self.rename_type(return_type, value_scopes, type_scopes)?;
                    value_scopes.push(local_scope);
                    let body =
                        self.rename_statement_list(body, value_scopes, type_scopes, false)?;
                    value_scopes.pop();
                    Node::FunctionDeclaration {
                        name: renamed_name,
                        parameters,
                        return_type,
                        body,
                        lambda,
                    }
                }
            }
            Node::GenericFunctionDeclaration {
                name,
                generic_params,
                generic_bounds,
                parameters,
                return_type,
                body,
                lambda,
            } => {
                let renamed_name = if top_level && !exported {
                    self.value_renames
                        .get(&name)
                        .cloned()
                        .unwrap_or(name.clone())
                } else {
                    name.clone()
                };
                let mut type_scope = HashSet::new();
                for param in &generic_params {
                    type_scope.insert(param.clone());
                }
                type_scopes.push(type_scope);
                let generic_bounds = self.rename_generic_bounds(generic_bounds, type_scopes)?;
                let mut local_scope = HashSet::new();
                let parameters = parameters
                    .into_iter()
                    .map(|(param_name, param_type)| {
                        if !ast::is_self_type(&param_type) {
                            local_scope.insert(param_name.clone());
                        }
                        self.rename_type(param_type, value_scopes, type_scopes)
                            .map(|param_type| (param_name, param_type))
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                let return_type = self.rename_type(return_type, value_scopes, type_scopes)?;
                value_scopes.push(local_scope);
                let body = self.rename_statement_list(body, value_scopes, type_scopes, false)?;
                value_scopes.pop();
                type_scopes.pop();
                Node::GenericFunctionDeclaration {
                    name: renamed_name,
                    generic_params,
                    generic_bounds,
                    parameters,
                    return_type,
                    body,
                    lambda,
                }
            }
            Node::TraitDeclaration { name, methods } => {
                let renamed_name = if top_level && !exported {
                    self.type_renames
                        .get(&name)
                        .cloned()
                        .unwrap_or(name.clone())
                } else {
                    name.clone()
                };
                let methods = methods
                    .into_iter()
                    .map(|method| {
                        Ok(ast::TraitMethodSignature {
                            name: method.name,
                            parameters: method
                                .parameters
                                .into_iter()
                                .map(|(param_name, param_type)| {
                                    self.rename_type(param_type, value_scopes, type_scopes)
                                        .map(|param_type| (param_name, param_type))
                                })
                                .collect::<Result<Vec<_>, String>>()?,
                            return_type: self.rename_type(
                                method.return_type,
                                value_scopes,
                                type_scopes,
                            )?,
                        })
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                Node::TraitDeclaration {
                    name: renamed_name,
                    methods,
                }
            }
            Node::ImplDeclaration {
                generic_params,
                generic_bounds,
                trait_names,
                target_type,
            } => {
                let mut type_scope = HashSet::new();
                for param in &generic_params {
                    type_scope.insert(param.clone());
                }
                type_scopes.push(type_scope);
                let generic_bounds = self.rename_generic_bounds(generic_bounds, type_scopes)?;
                let trait_names = trait_names
                    .into_iter()
                    .map(|name| self.rename_type_name(&name, type_scopes))
                    .collect();
                let target_type = self.rename_type(target_type, value_scopes, type_scopes)?;
                type_scopes.pop();
                Node::ImplDeclaration {
                    generic_params,
                    generic_bounds,
                    trait_names,
                    target_type,
                }
            }
            Node::StructDeclaration {
                name,
                fields,
                functions,
            } => {
                let renamed_name = if top_level && !exported {
                    self.type_renames
                        .get(&name)
                        .cloned()
                        .unwrap_or(name.clone())
                } else {
                    name.clone()
                };
                let fields = fields
                    .into_iter()
                    .map(|(field_name, field_type)| {
                        self.rename_type(field_type, value_scopes, type_scopes)
                            .map(|field_type| (field_name, field_type))
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                let functions = functions
                    .into_iter()
                    .map(|function| {
                        self.rename_statement(function, value_scopes, type_scopes, false, true)
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                Node::StructDeclaration {
                    name: renamed_name,
                    fields,
                    functions,
                }
            }
            Node::GenericStructDeclaration {
                name,
                generic_params,
                generic_bounds,
                fields,
                functions,
            } => {
                let renamed_name = if top_level && !exported {
                    self.type_renames
                        .get(&name)
                        .cloned()
                        .unwrap_or(name.clone())
                } else {
                    name.clone()
                };
                let mut type_scope = HashSet::new();
                for param in &generic_params {
                    type_scope.insert(param.clone());
                }
                type_scopes.push(type_scope);
                let generic_bounds = self.rename_generic_bounds(generic_bounds, type_scopes)?;
                let fields = fields
                    .into_iter()
                    .map(|(field_name, field_type)| {
                        self.rename_type(field_type, value_scopes, type_scopes)
                            .map(|field_type| (field_name, field_type))
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                let functions = functions
                    .into_iter()
                    .map(|function| {
                        self.rename_statement(function, value_scopes, type_scopes, false, true)
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                type_scopes.pop();
                Node::GenericStructDeclaration {
                    name: renamed_name,
                    generic_params,
                    generic_bounds,
                    fields,
                    functions,
                }
            }
            Node::EnumDeclaration { name, variants } => {
                let renamed_name = if top_level && !exported {
                    self.type_renames
                        .get(&name)
                        .cloned()
                        .unwrap_or(name.clone())
                } else {
                    name.clone()
                };
                let variants = variants
                    .into_iter()
                    .map(|variant| {
                        Ok(ast::EnumVariant {
                            name: variant.name,
                            payload_types: variant
                                .payload_types
                                .into_iter()
                                .map(|payload_type| {
                                    self.rename_type(payload_type, value_scopes, type_scopes)
                                })
                                .collect::<Result<Vec<_>, String>>()?,
                        })
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                Node::EnumDeclaration {
                    name: renamed_name,
                    variants,
                }
            }
            Node::GenericEnumDeclaration {
                name,
                generic_params,
                generic_bounds,
                variants,
            } => {
                let renamed_name = if top_level && !exported {
                    self.type_renames
                        .get(&name)
                        .cloned()
                        .unwrap_or(name.clone())
                } else {
                    name.clone()
                };
                let mut type_scope = HashSet::new();
                for param in &generic_params {
                    type_scope.insert(param.clone());
                }
                type_scopes.push(type_scope);
                let generic_bounds = self.rename_generic_bounds(generic_bounds, type_scopes)?;
                let variants = variants
                    .into_iter()
                    .map(|variant| {
                        Ok(ast::EnumVariant {
                            name: variant.name,
                            payload_types: variant
                                .payload_types
                                .into_iter()
                                .map(|payload_type| {
                                    self.rename_type(payload_type, value_scopes, type_scopes)
                                })
                                .collect::<Result<Vec<_>, String>>()?,
                        })
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                type_scopes.pop();
                Node::GenericEnumDeclaration {
                    name: renamed_name,
                    generic_params,
                    generic_bounds,
                    variants,
                }
            }
            Node::If {
                condition,
                body,
                else_if_blocks,
                else_block,
            } => Node::If {
                condition: Box::new(self.rename_expr(*condition, value_scopes, type_scopes)?),
                body: self.rename_statement_list(body, value_scopes, type_scopes, false)?,
                else_if_blocks: else_if_blocks
                    .into_iter()
                    .map(|block| {
                        self.rename_statement(block, value_scopes, type_scopes, false, false)
                    })
                    .collect::<Result<Vec<_>, String>>()?,
                else_block: else_block
                    .map(|body| self.rename_statement_list(body, value_scopes, type_scopes, false))
                    .transpose()?,
            },
            Node::Match { value, cases } => {
                let value = Box::new(self.rename_expr(*value, value_scopes, type_scopes)?);
                let cases = cases
                    .into_iter()
                    .map(|case| {
                        let pattern =
                            self.rename_match_pattern(case.pattern, value_scopes, type_scopes)?;
                        value_scopes.push(HashSet::new());
                        let scope = value_scopes.last_mut().expect("case scope exists");
                        match &pattern {
                            ast::MatchPattern::EnumVariant { bindings, .. } => {
                                for binding in bindings {
                                    scope.insert(binding.clone());
                                }
                            }
                            ast::MatchPattern::Struct { fields, .. } => {
                                for field in fields {
                                    scope.insert(field.binding.clone());
                                }
                            }
                            _ => {}
                        }
                        let body = self.rename_statement_list(
                            case.body,
                            value_scopes,
                            type_scopes,
                            false,
                        )?;
                        value_scopes.pop();
                        Ok(ast::MatchCase { pattern, body })
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                Node::Match { value, cases }
            }
            Node::For {
                init,
                condition,
                update,
                body,
            } => {
                value_scopes.push(HashSet::new());
                let init = init
                    .map(|init| {
                        self.rename_statement(*init, value_scopes, type_scopes, false, false)
                            .map(Box::new)
                    })
                    .transpose()?;
                let condition = condition
                    .map(|condition| {
                        self.rename_expr(*condition, value_scopes, type_scopes)
                            .map(Box::new)
                    })
                    .transpose()?;
                let update = update
                    .map(|update| {
                        self.rename_statement(*update, value_scopes, type_scopes, false, false)
                            .map(Box::new)
                    })
                    .transpose()?;
                let body = self.rename_statement_list(body, value_scopes, type_scopes, false)?;
                value_scopes.pop();
                Node::For {
                    init,
                    condition,
                    update,
                    body,
                }
            }
            Node::Return(value) => Node::Return(
                value
                    .map(|value| {
                        self.rename_expr(*value, value_scopes, type_scopes)
                            .map(Box::new)
                    })
                    .transpose()?,
            ),
            Node::Print(value) => Node::Print(Box::new(self.rename_expr(
                *value,
                value_scopes,
                type_scopes,
            )?)),
            Node::Input | Node::Literal(_) | Node::EMPTY => node,
            Node::Assignment {
                var,
                value,
                metadata,
            } => Node::Assignment {
                var: Box::new(self.rename_expr(*var, value_scopes, type_scopes)?),
                value: Box::new(self.rename_expr(*value, value_scopes, type_scopes)?),
                metadata,
            },
            other => self.rename_expr(other, value_scopes, type_scopes)?,
        })
    }

    fn rename_statement_list(
        &self,
        statements: Vec<Node>,
        value_scopes: &mut Vec<HashSet<String>>,
        type_scopes: &mut Vec<HashSet<String>>,
        top_level: bool,
    ) -> Result<Vec<Node>, String> {
        let mut output = Vec::new();
        for statement in statements {
            output.push(self.rename_statement(
                statement,
                value_scopes,
                type_scopes,
                top_level,
                false,
            )?);
        }
        Ok(output)
    }

    fn rename_expr(
        &self,
        node: Node,
        value_scopes: &mut Vec<HashSet<String>>,
        type_scopes: &mut Vec<HashSet<String>>,
    ) -> Result<Node, String> {
        Ok(match node {
            Node::Identifier(name) => Node::Identifier(self.rename_value_name(&name, value_scopes)),
            Node::Literal(_) | Node::Input | Node::EOI | Node::EMPTY => node,
            Node::BinaryOp {
                left,
                operator,
                right,
            } => Node::BinaryOp {
                left: Box::new(self.rename_expr(*left, value_scopes, type_scopes)?),
                operator,
                right: Box::new(self.rename_expr(*right, value_scopes, type_scopes)?),
            },
            Node::UnaryOp { operator, operand } => Node::UnaryOp {
                operator,
                operand: Box::new(self.rename_expr(*operand, value_scopes, type_scopes)?),
            },
            Node::FunctionCall {
                name,
                type_arguments,
                arguments,
                metadata,
            } => Node::FunctionCall {
                name: self.rename_value_name(&name, value_scopes),
                type_arguments: type_arguments
                    .into_iter()
                    .map(|type_argument| self.rename_type(type_argument, value_scopes, type_scopes))
                    .collect::<Result<Vec<_>, String>>()?,
                arguments: arguments
                    .into_iter()
                    .map(|group| {
                        group
                            .into_iter()
                            .map(|arg| self.rename_expr(arg, value_scopes, type_scopes))
                            .collect::<Result<Vec<_>, String>>()
                    })
                    .collect::<Result<Vec<_>, String>>()?,
                metadata,
            },
            Node::ArrayAccess { coordinates } => Node::ArrayAccess {
                coordinates: coordinates
                    .into_iter()
                    .map(|coord| self.rename_expr(coord, value_scopes, type_scopes))
                    .collect::<Result<Vec<_>, String>>()?,
            },
            Node::SliceAccess { start, end } => Node::SliceAccess {
                start: start
                    .map(|node| {
                        self.rename_expr(*node, value_scopes, type_scopes)
                            .map(Box::new)
                    })
                    .transpose()?,
                end: end
                    .map(|node| {
                        self.rename_expr(*node, value_scopes, type_scopes)
                            .map(Box::new)
                    })
                    .transpose()?,
            },
            Node::MemberAccess { member, metadata } => Node::MemberAccess {
                member: Box::new(match *member {
                    Node::FunctionCall {
                        name,
                        type_arguments,
                        arguments,
                        metadata,
                    } => Node::FunctionCall {
                        name,
                        type_arguments: type_arguments
                            .into_iter()
                            .map(|type_argument| {
                                self.rename_type(type_argument, value_scopes, type_scopes)
                            })
                            .collect::<Result<Vec<_>, String>>()?,
                        arguments: arguments
                            .into_iter()
                            .map(|group| {
                                group
                                    .into_iter()
                                    .map(|arg| self.rename_expr(arg, value_scopes, type_scopes))
                                    .collect::<Result<Vec<_>, String>>()
                            })
                            .collect::<Result<Vec<_>, String>>()?,
                        metadata,
                    },
                    Node::Identifier(name) => Node::Identifier(name),
                    other => self.rename_expr(other, value_scopes, type_scopes)?,
                }),
                metadata,
            },
            Node::Access { nodes } => {
                let mut output = Vec::new();
                for (index, node) in nodes.into_iter().enumerate() {
                    let renamed = match node {
                        Node::MemberAccess { .. } => {
                            self.rename_expr(node, value_scopes, type_scopes)?
                        }
                        Node::ArrayAccess { .. } | Node::SliceAccess { .. } => {
                            self.rename_expr(node, value_scopes, type_scopes)?
                        }
                        other if index == 0 => {
                            self.rename_expr(other, value_scopes, type_scopes)?
                        }
                        other => other,
                    };
                    output.push(renamed);
                }
                Node::Access { nodes: output }
            }
            Node::StructInitialization { _type, fields } => Node::StructInitialization {
                _type: self.rename_type(_type, value_scopes, type_scopes)?,
                fields: fields
                    .into_iter()
                    .map(|(name, value)| {
                        self.rename_expr(value, value_scopes, type_scopes)
                            .map(|value| (name, value))
                    })
                    .collect::<Result<Vec<_>, String>>()?,
            },
            Node::StaticFunctionCall {
                _type,
                name,
                arguments,
                metadata,
            } => Node::StaticFunctionCall {
                _type: self.rename_type(_type, value_scopes, type_scopes)?,
                name,
                arguments: arguments
                    .into_iter()
                    .map(|arg| self.rename_expr(arg, value_scopes, type_scopes))
                    .collect::<Result<Vec<_>, String>>()?,
                metadata,
            },
            Node::ArrayInit { elements } => Node::ArrayInit {
                elements: elements
                    .into_iter()
                    .map(|element| self.rename_expr(element, value_scopes, type_scopes))
                    .collect::<Result<Vec<_>, String>>()?,
            },
            Node::Block { .. }
            | Node::If { .. }
            | Node::Match { .. }
            | Node::For { .. }
            | Node::Return(_)
            | Node::Print(_)
            | Node::StructDestructure { .. }
            | Node::VariableDeclaration { .. }
            | Node::FunctionDeclaration { .. }
            | Node::GenericFunctionDeclaration { .. }
            | Node::TraitDeclaration { .. }
            | Node::ImplDeclaration { .. }
            | Node::StructDeclaration { .. }
            | Node::GenericStructDeclaration { .. }
            | Node::EnumDeclaration { .. }
            | Node::GenericEnumDeclaration { .. }
            | Node::Assignment { .. }
            | Node::Program { .. }
            | Node::Module { .. }
            | Node::Import { .. }
            | Node::Export { .. } => {
                self.rename_statement(node, value_scopes, type_scopes, false, false)?
            }
        })
    }

    fn rename_match_pattern(
        &self,
        pattern: ast::MatchPattern,
        value_scopes: &mut Vec<HashSet<String>>,
        type_scopes: &mut Vec<HashSet<String>>,
    ) -> Result<ast::MatchPattern, String> {
        Ok(match pattern {
            ast::MatchPattern::EnumVariant {
                enum_type,
                variant,
                bindings,
            } => ast::MatchPattern::EnumVariant {
                enum_type: enum_type
                    .map(|enum_type| self.rename_type(enum_type, value_scopes, type_scopes))
                    .transpose()?,
                variant,
                bindings,
            },
            ast::MatchPattern::Struct {
                struct_type,
                fields,
            } => ast::MatchPattern::Struct {
                struct_type: self.rename_type(struct_type, value_scopes, type_scopes)?,
                fields,
            },
        })
    }

    fn rename_generic_bounds(
        &self,
        generic_bounds: HashMap<String, Vec<String>>,
        type_scopes: &[HashSet<String>],
    ) -> Result<HashMap<String, Vec<String>>, String> {
        Ok(generic_bounds
            .into_iter()
            .map(|(param, bounds)| {
                (
                    param,
                    bounds
                        .into_iter()
                        .map(|bound| self.rename_type_name(&bound, type_scopes))
                        .collect::<Vec<_>>(),
                )
            })
            .collect())
    }

    fn rename_type(
        &self,
        sk_type: ast::Type,
        value_scopes: &mut Vec<HashSet<String>>,
        type_scopes: &mut Vec<HashSet<String>>,
    ) -> Result<ast::Type, String> {
        Ok(match sk_type {
            ast::Type::Array {
                elem_type,
                dimensions,
            } => ast::Type::Array {
                elem_type: Box::new(self.rename_type(*elem_type, value_scopes, type_scopes)?),
                dimensions: dimensions
                    .into_iter()
                    .map(|dim| self.rename_expr(dim, value_scopes, type_scopes))
                    .collect::<Result<Vec<_>, String>>()?,
            },
            ast::Type::Pointer { target_type } => ast::Type::Pointer {
                target_type: Box::new(self.rename_type(*target_type, value_scopes, type_scopes)?),
            },
            ast::Type::Slice { elem_type } => ast::Type::Slice {
                elem_type: Box::new(self.rename_type(*elem_type, value_scopes, type_scopes)?),
            },
            ast::Type::Const { inner } => ast::Type::Const {
                inner: Box::new(self.rename_type(*inner, value_scopes, type_scopes)?),
            },
            ast::Type::BindingConst { inner } => ast::Type::BindingConst {
                inner: Box::new(self.rename_type(*inner, value_scopes, type_scopes)?),
            },
            ast::Type::MutSelf => ast::Type::MutSelf,
            ast::Type::GenericInstance {
                base,
                type_arguments,
            } => ast::Type::GenericInstance {
                base: self.rename_type_name(&base, type_scopes),
                type_arguments: type_arguments
                    .into_iter()
                    .map(|arg| self.rename_type(arg, value_scopes, type_scopes))
                    .collect::<Result<Vec<_>, String>>()?,
            },
            ast::Type::Custom(name) => ast::Type::Custom(self.rename_type_name(&name, type_scopes)),
            ast::Type::Function {
                parameters,
                return_type,
            } => ast::Type::Function {
                parameters: parameters
                    .into_iter()
                    .map(|param| self.rename_type(param, value_scopes, type_scopes))
                    .collect::<Result<Vec<_>, String>>()?,
                return_type: Box::new(self.rename_type(*return_type, value_scopes, type_scopes)?),
            },
            other => other,
        })
    }

    fn rename_value_name(&self, name: &str, value_scopes: &[HashSet<String>]) -> String {
        if is_shadowed(name, value_scopes) {
            name.to_string()
        } else {
            self.value_renames
                .get(name)
                .cloned()
                .unwrap_or_else(|| name.to_string())
        }
    }

    fn rename_type_name(&self, name: &str, type_scopes: &[HashSet<String>]) -> String {
        if is_shadowed(name, type_scopes) {
            name.to_string()
        } else {
            self.type_renames
                .get(name)
                .cloned()
                .unwrap_or_else(|| name.to_string())
        }
    }
}

#[derive(Clone, Copy)]
enum TopLevelDeclKind {
    Value,
    Type,
}

fn top_level_decl_name(node: &Node) -> Option<(TopLevelDeclKind, String)> {
    match node {
        Node::VariableDeclaration { name, .. }
        | Node::FunctionDeclaration {
            name,
            lambda: false,
            ..
        }
        | Node::GenericFunctionDeclaration {
            name,
            lambda: false,
            ..
        } => Some((TopLevelDeclKind::Value, name.clone())),
        Node::StructDeclaration { name, .. }
        | Node::GenericStructDeclaration { name, .. }
        | Node::EnumDeclaration { name, .. }
        | Node::GenericEnumDeclaration { name, .. }
        | Node::TraitDeclaration { name, .. } => Some((TopLevelDeclKind::Type, name.clone())),
        _ => None,
    }
}

fn validate_export_target(node: &Node) -> Result<(), String> {
    match node {
        Node::VariableDeclaration { .. }
        | Node::FunctionDeclaration { lambda: false, .. }
        | Node::GenericFunctionDeclaration { lambda: false, .. }
        | Node::StructDeclaration { .. }
        | Node::GenericStructDeclaration { .. }
        | Node::EnumDeclaration { .. }
        | Node::GenericEnumDeclaration { .. }
        | Node::TraitDeclaration { .. } => Ok(()),
        _ => Err(
            "`export` supports only top-level variables, functions, structs, enums, and traits"
                .to_string(),
        ),
    }
}

fn mangle_private_name(module_name: &str, name: &str) -> String {
    format!("__{}_{}", sanitize_module_name(module_name), name)
}

fn sanitize_module_name(module_name: &str) -> String {
    module_name
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
        .collect()
}

fn is_shadowed(name: &str, scopes: &[HashSet<String>]) -> bool {
    scopes.iter().rev().any(|scope| scope.contains(name))
}

fn extract_module_name(statements: &[Node]) -> Result<Option<String>, String> {
    let mut module_name = None;
    for statement in statements {
        if let Node::Module { name } = statement {
            if module_name.is_some() {
                return Err("a file may declare at most one module".to_string());
            }
            module_name = Some(name.clone());
        }
    }
    Ok(module_name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use uuid::Uuid;

    fn write_file(path: &Path, contents: &str) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, contents).unwrap();
    }

    #[test]
    fn loads_imported_module_before_entry_statements() {
        let root = env::temp_dir().join(format!("skunk_modules_{}", Uuid::new_v4()));
        let entry = root.join("main.skunk");
        let module = root.join("std").join("math.skunk");

        write_file(
            &module,
            r#"
            module std.math;

            function inc(n: int): int {
                return n + 1;
            }
            "#,
        );
        write_file(
            &entry,
            r#"
            import std.math;

            function main(): void {
                print(inc(6));
            }
            "#,
        );

        let loaded = load_program(&entry).unwrap();
        let Node::Program { statements } = loaded else {
            panic!("expected program node");
        };

        assert!(statements.iter().any(|statement| matches!(
            statement,
            Node::FunctionDeclaration { name, .. } if name == "inc"
        )));
        assert!(statements.iter().any(|statement| matches!(
            statement,
            Node::FunctionDeclaration { name, .. } if name == "main"
        )));

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn rejects_module_name_mismatch() {
        let root = env::temp_dir().join(format!("skunk_modules_{}", Uuid::new_v4()));
        let entry = root.join("main.skunk");
        let module = root.join("std").join("math.skunk");

        write_file(
            &module,
            r#"
            module std.wrong;

            function inc(n: int): int {
                return n + 1;
            }
            "#,
        );
        write_file(
            &entry,
            r#"
            import std.math;

            function main(): void {}
            "#,
        );

        let err = load_program(&entry).unwrap_err();
        assert!(err.contains("module declaration mismatch"));

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn keeps_exported_names_and_mangles_private_imported_names() {
        let root = env::temp_dir().join(format!("skunk_modules_{}", Uuid::new_v4()));
        let entry = root.join("main.skunk");
        let module = root.join("std").join("math.skunk");

        write_file(
            &module,
            r#"
            module std.math;

            function helper(n: int): int {
                return n + 1;
            }

            export function inc(n: int): int {
                return helper(n);
            }
            "#,
        );
        write_file(
            &entry,
            r#"
            import std.math;

            function main(): void {
                print(inc(6));
            }
            "#,
        );

        let loaded = load_program(&entry).unwrap();
        let Node::Program { statements } = loaded else {
            panic!("expected program node");
        };

        assert!(statements.iter().any(|statement| matches!(
            statement,
            Node::FunctionDeclaration { name, .. } if name == "inc"
        )));
        assert!(statements.iter().any(|statement| matches!(
            statement,
            Node::FunctionDeclaration { name, .. } if name == "__std_math_helper"
        )));
        assert!(!statements.iter().any(|statement| matches!(
            statement,
            Node::FunctionDeclaration { name, .. } if name == "helper"
        )));

        let _ = fs::remove_dir_all(root);
    }
}
