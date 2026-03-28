use crate::ast::Type::Void;
use crate::ast::{
    fits_integer_type, is_integral_type, is_numeric_assignable, is_numeric_type, is_scalar_type,
    promoted_numeric_type, type_to_string, Literal, Metadata, Node, Operator, Type,
    UnaryOperator,
};
use std::collections::HashMap;
use std::fmt::format;
use std::ops::Deref;

#[derive(Debug, PartialEq, Clone)]
struct Symbol {
    name: String,
    sk_type: Type,
}

#[derive(Debug, PartialEq, Clone)]
struct StructSymbol {
    name: String,
    fields: HashMap<String, Symbol>,
    functions: HashMap<String, Symbol>,
}

#[derive(Debug, PartialEq, Clone)]
struct SymbolTable {
    vars: HashMap<String, Symbol>,
}

impl SymbolTable {
    fn new() -> Self {
        SymbolTable {
            vars: HashMap::new(),
        }
    }

    fn add(&mut self, s: Symbol) {
        self.vars.insert(s.name.clone(), s);
    }

    fn get(&self, name: &str) -> Option<&Symbol> {
        self.vars.get(name)
    }
    fn has_var(&self, name: &str) -> bool {
        self.vars.contains_key(name)
    }
}

#[derive(Debug, PartialEq, Clone)]
struct SymbolTables {
    tables: Vec<SymbolTable>,
}

impl SymbolTables {
    fn new() -> Self {
        SymbolTables { tables: vec![] }
    }
    fn add(&mut self, var_table: SymbolTable) {
        self.tables.push(var_table);
    }

    fn pop(&mut self) {
        self.tables.pop();
    }

    fn get(&self) -> &SymbolTable {
        &self.tables.last().unwrap()
    }
    fn get_mut(&mut self) -> &mut SymbolTable {
        self.tables.last_mut().unwrap()
    }

    fn get_symbol(&self, name: &str) -> Option<&Symbol> {
        for table in self.tables.iter().rev() {
            if table.has_var(name) {
                return table.get(name);
            }
        }
        None
    }
    fn get_fun(&self, name: &str) -> Option<Symbol> {
        let s = self.get_symbol(name);
        if let Some(s) = s {
            match s.sk_type {
                Type::Function { .. } => Some(s.clone()),
                _ => None,
            }
        } else {
            None
        }
    }
}

fn create_symbols(declarations: &Vec<(String, Type)>) -> Vec<Symbol> {
    declarations
        .iter()
        .map(|d| Symbol {
            name: d.0.clone(),
            sk_type: d.1.clone(),
        })
        .collect()
}

fn func_decl_node_to_symbol(node: &Node) -> Symbol {
    match node {
        Node::FunctionDeclaration {
            name,
            parameters,
            return_type,
            ..
        } => Symbol {
            name: name.to_string(),
            sk_type: Type::Function {
                parameters: parameters.iter().map(|p| p.1.clone()).collect(),
                return_type: Box::new(return_type.clone()),
            },
        },
        _ => panic!("expected function declaration"),
    }
}

#[derive(Debug, PartialEq, Clone)]
struct GlobalScope {
    structs: HashMap<String, StructSymbol>,
    functions: HashMap<String, Symbol>,
    variables: HashMap<String, Symbol>,
}

impl GlobalScope {
    fn new() -> Self {
        GlobalScope {
            structs: HashMap::new(),
            functions: HashMap::new(),
            variables: HashMap::new(),
        }
    }

    fn add(&mut self, node: &Node) {
        match node {
            Node::StructDeclaration {
                name,
                fields,
                functions,
            } => {
                self.structs.insert(
                    name.clone(),
                    StructSymbol {
                        name: name.clone(),
                        fields: fields
                            .iter()
                            .map(|field| {
                                (
                                    field.0.to_string(),
                                    Symbol {
                                        name: field.0.to_string(),
                                        sk_type: field.1.clone(),
                                    },
                                )
                            })
                            .collect::<HashMap<_, _>>(),
                        functions: functions
                            .iter()
                            .map(|n| {
                                let fun_symbol = func_decl_node_to_symbol(n);
                                (fun_symbol.name.clone(), fun_symbol)
                            })
                            .collect::<HashMap<_, _>>(),
                    },
                );
            }
            Node::FunctionDeclaration { name, .. } => {
                self.functions
                    .insert(name.clone(), func_decl_node_to_symbol(node));
            }
            Node::VariableDeclaration {
                var_type,
                name,
                value,
                metadata,
            } => {
                self.variables.insert(
                    name.clone(),
                    Symbol {
                        name: name.clone(),
                        sk_type: var_type.clone(),
                    },
                );
            }
            _ => {}
        }
    }
}

fn is_assignable(expected: &Type, actual: &Type) -> bool {
    expected == actual || is_numeric_assignable(expected, actual)
}

fn resolve_literal_type(literal: &Literal, expected_type_opt: Option<&Type>) -> Result<Type, String> {
    match literal {
        Literal::Integer(value) => {
            if let Some(expected_type) = expected_type_opt {
                if is_integral_type(expected_type) {
                    if *expected_type == Type::Long && !fits_integer_type(*value, &Type::Int) {
                        return Err(format!(
                            "integer literal `{}` is out of range for `int`; use an `L` suffix for `long`",
                            value
                        ));
                    }
                    if fits_integer_type(*value, expected_type) {
                        return Ok(expected_type.clone());
                    }
                    return Err(format!(
                        "integer literal `{}` is out of range for `{}`",
                        value,
                        type_to_string(expected_type)
                    ));
                }
            }

            if fits_integer_type(*value, &Type::Int) {
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

fn resolve_cmp(left: &Type, right: &Type) -> Result<Type, String> {
    if promoted_numeric_type(left, right).is_some() || (*left == Type::Char && *right == Type::Char)
    {
        Ok(Type::Boolean)
    } else {
        Err(format!(
            "unexpected types for < > <= >= : {:?} and {:?}",
            left, right
        ))
    }
}

fn resolve_eq(left: &Type, right: &Type) -> Result<Type, String> {
    if promoted_numeric_type(left, right).is_some()
        || (*left == Type::String && *right == Type::String)
        || (*left == Type::Boolean && *right == Type::Boolean)
        || (*left == Type::Char && *right == Type::Char)
    {
        Ok(Type::Boolean)
    } else {
        Err(format!(
            "unexpected types for '==' : {:?} and {:?}",
            left, right
        ))
    }
}

fn resolve_not_eq(left: &Type, right: &Type) -> Result<Type, String> {
    resolve_eq(left, right)
}

fn resolve_add(left: &Type, right: &Type) -> Result<Type, String> {
    if let Some(promoted) = promoted_numeric_type(left, right) {
        Ok(promoted)
    } else if (*left == Type::String && is_scalar_type(right))
        || (is_scalar_type(left) && *right == Type::String)
    {
        Ok(Type::String)
    } else {
        Err(format!(
            "unexpected types for +: {:?} and {:?}",
            left, right
        ))
    }
}

fn resolve_subtract(left: &Type, right: &Type) -> Result<Type, String> {
    promoted_numeric_type(left, right).ok_or_else(|| {
        format!(
            "unexpected types for -: {:?} and {:?}",
            left, right
        )
    })
}

fn resolve_multiply(left: &Type, right: &Type) -> Result<Type, String> {
    promoted_numeric_type(left, right).ok_or_else(|| {
        format!(
            "unexpected types for *: {:?} and {:?}",
            left, right
        )
    })
}

fn resolve_divide(left: &Type, right: &Type) -> Result<Type, String> {
    promoted_numeric_type(left, right).ok_or_else(|| {
        format!(
            "unexpected types for /: {:?} and {:?}",
            left, right
        )
    })
}

fn resolve_mod(left: &Type, right: &Type) -> Result<Type, String> {
    if is_integral_type(left) && is_integral_type(right) {
        Ok(promoted_numeric_type(left, right).unwrap())
    } else {
        Err(format!(
            "unexpected types for %: {:?} and {:?}",
            left, right
        ))
    }
}

fn resolve_logical(left: &Type, right: &Type) -> Result<Type, String> {
    if *left == Type::Boolean && *right == Type::Boolean {
        Ok(Type::Boolean)
    } else {
        Err(format!(
            "unexpected types for boolean operation: {:?} and {:?}",
            left, right
        ))
    }
}

fn resolve_access(
    global_scope: &GlobalScope,
    symbol_tables: &mut SymbolTables,
    curr: Type,
    i: usize,
    access_nodes: &Vec<Node>,
) -> Result<Type, String> {
    if i == access_nodes.len() {
        return Ok(curr);
    }
    match access_nodes.get(i).unwrap() {
        Node::ArrayAccess { .. } => match curr {
            Type::Array { elem_type, .. } => resolve_access(
                global_scope,
                symbol_tables,
                elem_type.deref().clone(),
                i + 1,
                access_nodes,
            ),
            Type::Slice { elem_type, .. } => Ok(elem_type.deref().clone()),
            _ => Err("array access to not array variable".to_string()),
        },

        Node::MemberAccess { member, metadata } => match curr {
            Type::Custom(struct_name) => {
                if !global_scope.structs.contains_key(&struct_name) {
                    return Err("error: struct doesn't exist".to_string());
                }
                let struct_symbol = global_scope.structs.get(&struct_name).unwrap();
                match member.deref() {
                    Node::Identifier(field_name) => {
                        if !struct_symbol.fields.contains_key(field_name) {
                            return Err(format!(
                                "error {}:{}: no field `{}` on type `{}`",
                                metadata.span.line, metadata.span.start, field_name, struct_name
                            ));
                        }
                        resolve_access(
                            global_scope,
                            symbol_tables,
                            struct_symbol
                                .fields
                                .get(field_name)
                                .unwrap()
                                .sk_type
                                .clone(),
                            i + 1,
                            access_nodes,
                        )
                    }
                    Node::FunctionCall {
                        name,
                        arguments,
                        metadata,
                    } => {
                        if !struct_symbol.functions.contains_key(name) {
                            return Err(format!(
                                "error {}:{}:no method named `{}` found for struct `{}` in the current scope",
                                metadata.span.line,
                                metadata.span.start,
                                name,
                                struct_name,
                            ));
                        }
                        let return_type = resolve_function_call(
                            global_scope,
                            symbol_tables,
                            struct_symbol.functions.get(name).unwrap(),
                            member.deref(),
                        )?;
                        let args_types_res: Result<Vec<ResolveResult>, String> = arguments
                            .first()
                            .expect("at least one arg group is required")
                            .iter()
                            .map(|arg| resolve_type(global_scope, symbol_tables, arg, None))
                            .collect();
                        let args_types = args_types_res?; // why ?
                                                          // println!("args_types = {:?}", args_types);

                        resolve_access(
                            global_scope,
                            symbol_tables,
                            return_type,
                            i + 1,
                            access_nodes,
                        )
                    }
                    _ => panic!("expected member access node"),
                }
            }
            _ => Err(format!("access to member access to not instance structs")),
        },
        _ => unreachable!("unexpected access node"),
    }
}

fn resolve_function_call(
    global_scope: &GlobalScope,
    symbol_tables: &mut SymbolTables,
    function_symbol: &Symbol,
    node: &Node,
) -> Result<Type, String> {
    if let Node::FunctionCall {
        name,
        arguments,
        metadata,
    } = node
    {
        // println!("resolve_function_call={:?}", function_symbol);
        let mut curr_type = &function_symbol.sk_type;
        for args in arguments {
            match curr_type {
                Type::Function {
                    parameters,
                    return_type,
                    ..
                } => {
                    let mut parameters = &parameters[0..];
                    if parameters.len() > 0 && *parameters.first().unwrap() == Type::SkSelf {
                        parameters = &parameters[1..];
                    }
                    if args.len() != parameters.len() {
                        return Err(format!(
                            "incorrect number of args to function {}. expected={}, actual={}",
                            name,
                            parameters.len(),
                            args.len()
                        ));
                    }
                    let mut argument_types = Vec::new();
                    for i in 0..args.len() {
                        argument_types.push(resolve_type(
                            global_scope,
                            symbol_tables,
                            &args[i],
                            Some(&parameters[i]),
                        )?);
                    }
                    for i in 0..argument_types.len() {
                        if !is_assignable(&parameters[i], &argument_types[i].sk_type) {
                            return Err(format!(
                                "error {}:{}:arguments to function '{}' are incorrect. \
                                parameter pos='{}', expected type='{:?}', actual type='{:?}'",
                                metadata.span.line,
                                metadata.span.start,
                                name,
                                i,
                                parameters[i],
                                argument_types[i]
                            ));
                        }
                    }
                    curr_type = return_type;
                }
                _ => panic!("expected function call"),
            }
        }
        Ok(curr_type.clone())
    } else {
        panic!("expected function call");
    }
}

#[derive(Debug, PartialEq, Clone)]
struct ResolveResult {
    sk_type: Type,
    returned: bool,
}

impl ResolveResult {
    fn new(sk_type: Type) -> Self {
        ResolveResult {
            sk_type,
            returned: false,
        }
    }

    fn returned(sk_type: Type) -> Self {
        ResolveResult {
            sk_type,
            returned: true,
        }
    }

    fn to_returned(&self) -> ResolveResult {
        ResolveResult {
            sk_type: self.sk_type.clone(),
            returned: true,
        }
    }
}

fn assert_type(curr: Type, new: Type) -> Result<Type, String> {
    if new != Type::Void {
        if curr != Type::Void && curr != new {
            Err(format!("expected type {:?}, found type {:?}", curr, new))
        } else {
            Ok(new)
        }
    } else {
        Ok(curr)
    }
}

fn resolve_type(
    global_scope: &GlobalScope,
    symbol_tables: &mut SymbolTables,
    node: &Node,
    expected_type_opt: Option<&Type>,
) -> Result<ResolveResult, String> {
    match node {
        Node::Program { statements } => {
            for statement in statements {
                let res = resolve_type(global_scope, symbol_tables, statement, None);
                match &res {
                    Err(e) => return Err(e.to_string()),
                    _ => (),
                }
            }
            Ok(ResolveResult::new(Type::Void))
        }
        Node::StructInitialization { name, .. } => {
            if !global_scope.structs.contains_key(name) {
                Err("struct doesn't exist".to_string())
            } else {
                Ok(ResolveResult::new(Type::Custom(name.clone())))
            }
        }
        Node::StructDeclaration { name, .. } => Ok(ResolveResult::new(Type::Custom(name.clone()))),
        Node::FunctionDeclaration {
            name,
            parameters,
            return_type,
            body,
            lambda,
        } => {
            // println!(
            //     "function declaration {}, return_type={:?}, lambda={:?}, body={:?}",
            //     name, return_type, lambda, body
            // );
            let mut returned = false;
            let mut symbol_table = SymbolTable::new();
            for parameter in parameters {
                symbol_table.add(Symbol {
                    name: parameter.0.clone(),
                    sk_type: parameter.1.clone(),
                })
            }
            symbol_tables.add(symbol_table);
            let mut actual_return_type = Type::Void;
            for n in body {
                let t = resolve_type(global_scope, symbol_tables, n, Some(return_type))?;
                if t.returned {
                    returned = true;
                    actual_return_type = assert_type(actual_return_type, t.sk_type)?;
                }
            }
            symbol_tables.pop();

            if !returned && *return_type != Void {
                return Err("missing return statement".to_string());
            }

            if actual_return_type != *return_type {
                Err(format!(
                    "function '{}' return type mismatch. expected={:?}, actual={:?}",
                    name, return_type, actual_return_type
                ))
            } else {
                Ok(ResolveResult::new(Type::Function {
                    parameters: parameters.iter().map(|p| p.1.clone()).collect(),
                    return_type: Box::new(return_type.clone()),
                }))
            }
        }
        Node::VariableDeclaration {
            var_type,
            name,
            value,
            metadata,
        } => {
            // println!("variable declaration {}:{:?}", name, var_type);
            symbol_tables.get_mut().add(Symbol {
                name: name.clone(),
                sk_type: var_type.clone(),
            });
            if let Some(body) = value {
                let value_type = resolve_type(
                    global_scope,
                    symbol_tables,
                    &body.deref(),
                    Some(var_type),
                )?
                .sk_type;
                if !is_assignable(var_type, &value_type) {
                    Err(format!(
                        "expected '{:?}' var type: {:?}, actual: {:?}. pos: {:?}",
                        name.clone(),
                        var_type,
                        value_type.clone(),
                        metadata
                    ))
                } else {
                    // println!("result variable declaration {}:{:?}", name, var_type);
                    Ok(ResolveResult::new(var_type.clone()))
                }
            } else {
                Ok(ResolveResult::new(var_type.clone()))
            }
        }
        Node::FunctionCall {
            name, arguments, ..
        } => {
            let func_symbol = symbol_tables
                .get_fun(name)
                .or_else(|| global_scope.functions.get(name).cloned())
                .expect(format!("function '{}' not found", name).as_ref());

            resolve_function_call(global_scope, symbol_tables, &func_symbol, node)
                .map(|t| ResolveResult::new(t))
        }
        Node::BinaryOp {
            left,
            operator,
            right,
        } => {
            let left_type =
                resolve_type(global_scope, symbol_tables, left.deref(), expected_type_opt)?.sk_type;
            let right_type = resolve_type(
                global_scope,
                symbol_tables,
                right.deref(),
                expected_type_opt,
            )?
            .sk_type;
            match operator {
                Operator::Add => {
                    resolve_add(&left_type, &right_type).map(|t| ResolveResult::new(t))
                }
                Operator::Subtract => {
                    resolve_subtract(&left_type, &right_type).map(|t| ResolveResult::new(t))
                }
                Operator::Multiply => {
                    resolve_multiply(&left_type, &right_type).map(|t| ResolveResult::new(t))
                }
                Operator::Divide => {
                    resolve_divide(&left_type, &right_type).map(|t| ResolveResult::new(t))
                }
                Operator::Mod => {
                    resolve_mod(&left_type, &right_type).map(|t| ResolveResult::new(t))
                }
                Operator::And | Operator::Or => {
                    resolve_logical(&left_type, &right_type).map(|t| ResolveResult::new(t))
                }
                Operator::LessThan
                | Operator::GreaterThan
                | Operator::GreaterThanOrEqual
                | Operator::LessThanOrEqual => {
                    resolve_cmp(&left_type, &right_type).map(|t| ResolveResult::new(t))
                }
                Operator::Equals => {
                    resolve_eq(&left_type, &right_type).map(|t| ResolveResult::new(t))
                }
                Operator::NotEquals => {
                    resolve_not_eq(&left_type, &right_type).map(|t| ResolveResult::new(t))
                }
                _ => unreachable!("todo {:?}", operator),
            }
        }
        Node::Assignment { var, value, .. } => {
            let var_type =
                resolve_type(global_scope, symbol_tables, var.deref(), expected_type_opt)?.sk_type;
            let value_type = resolve_type(
                global_scope,
                symbol_tables,
                value.deref(),
                Some(&var_type),
            )?
            .sk_type;
            if !is_assignable(&var_type, &value_type) {
                // todo include var name, metadata
                Err(format!(
                    "assignment type mismatch. expected: {:?}, actual: {:?}",
                    var_type, value_type
                ))
            } else {
                // Ok(var_type.clone())
                Ok(ResolveResult::new(var_type.clone()))
            }
        }
        Node::ArrayInit { elements } => {
            assert!(elements.len() > 0, "array init cannot be empty");
            let expected_elem_type_opt = match expected_type_opt {
                Some(Type::Array { elem_type, .. }) | Some(Type::Slice { elem_type, .. }) => {
                    Some(elem_type.deref())
                }
                _ => None,
            };

            let mut curr: Type = resolve_type(
                global_scope,
                symbol_tables,
                &elements[0],
                expected_elem_type_opt,
            )?
            .sk_type;
            for i in 1..elements.len() {
                let t = resolve_type(
                    global_scope,
                    symbol_tables,
                    &elements[i],
                    expected_elem_type_opt,
                )?
                .sk_type;
                if t != curr {
                    if let Some(promoted) = promoted_numeric_type(&curr, &t) {
                        curr = promoted;
                    } else {
                        return Err("invalid arr value type".to_string());
                    }
                }
            }
            Ok(ResolveResult::new(Type::Slice {
                elem_type: Box::new(curr),
            }))
        }
        Node::Literal(literal) => resolve_literal_type(literal, expected_type_opt)
            .map(|t| ResolveResult::new(t)),
        Node::EOI => Ok(ResolveResult::new(Type::Void)),
        Node::Identifier(name) => {
            let res = symbol_tables
                .get_symbol(name)
                .map(|v| ResolveResult::new(v.sk_type.clone()))
                .or(global_scope
                    .variables
                    .get(name)
                    .map(|v| ResolveResult::new(v.sk_type.clone())))
                .ok_or(format!("unknown variable '{}'", name));
            // println!("resolve identifier {}, res={:?}", name, res);
            res
        }
        Node::Access { nodes } => {
            let start = resolve_type(
                global_scope,
                symbol_tables,
                nodes.get(0).unwrap(),
                expected_type_opt,
            )?
            .sk_type;
            resolve_access(global_scope, symbol_tables, start, 1, nodes)
                .map(|t| ResolveResult::new(t))
        }
        Node::StaticFunctionCall {
            _type,
            name,
            arguments,
            metadata,
        } => {
            match _type {
                Type::Void => {}
                Type::Byte => {}
                Type::Short => {}
                Type::Int => {}
                Type::Long => {}
                Type::Float => {}
                Type::Double => {}
                Type::String => {}
                Type::Boolean => {}
                Type::Char => {}
                Type::Array { elem_type, .. } => {
                    if arguments.len() != 1 {
                        return Err(format!(
                            "error {}:{}: array `new` method should have exactly one argument",
                            metadata.span.line, metadata.span.start,
                        ));
                    }
                    let arg_type = resolve_type(
                        global_scope,
                        symbol_tables,
                        arguments.get(0).unwrap(),
                        Some(elem_type.deref()),
                    )?;
                    if !is_assignable(elem_type.deref(), &arg_type.sk_type) {
                        return Err(format!("error {}:{}: array `new`. expected arg type is `{:?}` but given `{:?}`",
                                           metadata.span.line,
                                           metadata.span.start,
                                           elem_type.deref(), arg_type));
                    }
                }
                Type::Slice { .. } => {}
                Type::Custom(_) => {}
                Type::Function { .. } => {}
                Type::SkSelf => {}
            }
            Ok(ResolveResult::new(_type.clone()))
        }
        Node::Block { statements } => {
            let var_table = symbol_tables.get().clone();
            symbol_tables.add(var_table);
            let mut result = Type::Void;
            for statement in statements {
                let res = resolve_type(global_scope, symbol_tables, statement, expected_type_opt)?;
                if res.returned {
                    result = assert_type(result, Type::Void)?;
                }
            }
            if result != Type::Void {
                Ok(ResolveResult::returned(result))
            } else {
                Ok(ResolveResult::new(Type::Void))
            }
        }
        Node::If {
            condition,
            body,
            else_block,
            else_if_blocks,
        } => {
            let mut returned = false;
            let cond_type =
                resolve_type(global_scope, symbol_tables, condition, expected_type_opt)?.sk_type;
            if cond_type != Type::Boolean {
                return Err(format!(
                    "if condition type mismatch. expected boolean, got {:?}",
                    cond_type
                ));
            }

            // let mut types = HashSet::new();
            let mut result = Type::Void;
            for n in body {
                let res = resolve_type(global_scope, symbol_tables, n, expected_type_opt)?;
                if res.returned {
                    returned = true;
                    // println!("if returns = {:?}", res);
                    result = assert_type(result, res.sk_type)?;
                }
            }
            for n in else_if_blocks {
                let res = resolve_type(global_scope, symbol_tables, n, expected_type_opt)?;
                if res.returned {
                    // println!("else_if returns = {:?}", res);
                    result = assert_type(result, res.sk_type)?;
                } else {
                    returned = false;
                }
            }

            if let Some(nodes) = else_block {
                for n in nodes {
                    let res = resolve_type(global_scope, symbol_tables, n, expected_type_opt)?;
                    if res.returned {
                        // println!("else returns = {:?}", res);
                        result = assert_type(result, res.sk_type)?;
                    } else {
                        returned = false;
                    }
                }
            } else {
                returned = false;
            }
            if returned {
                Ok(ResolveResult::returned(result))
            } else {
                Ok(ResolveResult::new(Type::Void))
            }
        }
        Node::For {
            init,
            condition,
            update,
            body,
        } => {
            if let Some(init_node) = init {
                resolve_type(global_scope, symbol_tables, init_node, None)?;
            }

            if let Some(cond_node) = condition {
                let cond_type = resolve_type(global_scope, symbol_tables, cond_node, None)?.sk_type;
                if cond_type != Type::Boolean {
                    return Err(format!(
                        "For loop condition must be of type Boolean, got {:?}",
                        cond_type
                    ));
                }
            }

            if let Some(update_node) = update {
                resolve_type(global_scope, symbol_tables, update_node, None)?;
            }

            let mut returned = false;
            let mut body_return_type = Type::Void;
            for statement in body {
                let res = resolve_type(global_scope, symbol_tables, statement, None)?;
                if res.returned {
                    returned = true;
                    body_return_type = assert_type(body_return_type, res.sk_type)?;
                }
            }
            Ok(if returned {
                ResolveResult::returned(body_return_type)
            } else {
                ResolveResult::new(Type::Void)
            })
        }

        Node::Print(n) => {
            resolve_type(global_scope, symbol_tables, n.deref(), expected_type_opt)?; // verify string
            Ok(ResolveResult::new(Type::Void))
        }
        Node::Return(body_opt) => {
            let mut res = ResolveResult::returned(Type::Void);
            if let Some(body) = body_opt {
                res = resolve_type(global_scope, symbol_tables, body, expected_type_opt)?.to_returned();
            }
            if let Some(expected_type) = expected_type_opt {
                if !is_assignable(expected_type, &res.sk_type) {
                    return Err(format!(
                        "return invalid type. expected={:?}, actual={:?}, body={:?}",
                        expected_type, res.sk_type, body_opt
                    ));
                }
                res = ResolveResult::returned(expected_type.clone());
            }
            Ok(res)
        }
        Node::UnaryOp { operator, operand } => {
            let operand_type =
                resolve_type(global_scope, symbol_tables, operand, expected_type_opt)?.sk_type;
            match operator {
                UnaryOperator::Plus | UnaryOperator::Minus => {
                    if is_numeric_type(&operand_type) {
                        Ok(ResolveResult::new(operand_type))
                    } else {
                        Err(format!(
                            "unary operator `{:?}` requires a numeric operand, got {:?}",
                            operator, operand_type
                        ))
                    }
                }
                UnaryOperator::Negate => {
                    if operand_type == Type::Boolean {
                        Ok(ResolveResult::new(Type::Boolean))
                    } else {
                        Err(format!(
                            "unary operator `!` requires a boolean operand, got {:?}",
                            operand_type
                        ))
                    }
                }
            }
        }
        _ => unreachable!("{}", format!("{:?}", node)),
    }
}

pub fn check(node: &Node) -> Result<(), String> {
    let mut var_tables = SymbolTables::new();
    var_tables.add(SymbolTable::new());
    let mut global_scope = GlobalScope::new();
    match node {
        Node::Program { statements } => {
            for statement in statements {
                global_scope.add(statement);
            }
        }
        _ => panic!("expected program node"),
    }
    match resolve_type(&mut global_scope, &mut var_tables, node, None) {
        Err(e) => Err(e.to_string()),
        Ok(_) => Ok(()),
    }
}

mod tests {
    use super::*;
    use crate::ast;

    #[test]
    fn test_check() {
        let source_code = r#"
        struct Point {
            x:int;
            y:int;

            function set_x(self, x:int) {
                self.x = x;
            }

            function get_x(self):int {
               return self.x;
            }

            function set_y(self, y:int) {
                self.y = y;
            }

            function get_y(self):int {
                return self.y;
            }
        }

        global_var: int = 1;

        function foo(i:int):int {
            return i;
        }

        foo(1);
        p:Point = Point { x: 0, y: 0 };
        p.set_x(1);

        "#;
        let program = ast::parse(source_code);
        check(&program).unwrap();
    }

    #[test]
    fn test_function_call_wrong_args() {
        let source_code = r#"
        function foo(a: int): int {
            return a;
        }
        foo("string");
        "#;
        let program = ast::parse(source_code);
        assert!(check(&program).is_err());
    }

    #[test]
    fn test_struct_field_not_exist() {
        let source_code = r#"
        struct Point {
            x: int;
            y: int;
        }
        p: Point = Point { x: 0, y: 0 };
        a = p.z; // 'z' field does not exist
        "#;
        let program = ast::parse(source_code);
        assert!(check(&program).is_err());
    }

    #[test]
    fn test_function_return_type_mismatch() {
        let source_code = r#"
        function foo(): int {
            return "string";
        }
        "#;
        let program = ast::parse(source_code);
        assert!(check(&program).is_err());
    }

    #[test]
    fn test_array_access_wrong_type() {
        let source_code = r#"
        arr: int[5] = int[5]::new(1);
        s: string = arr[0];
        "#;
        let program = ast::parse(source_code);
        assert!(check(&program).is_err());
    }

    #[test]
    fn test_array_init_wrong_type() {
        let source_code = r#"
        arr: int[5] = int[5]::new("1");
        "#;
        let program = ast::parse(source_code);
        assert!(check(&program).is_err());
    }

    #[test]
    fn test_inline_array_init() {
        let source_code = r#"
            arr:int[] = [1,2,3,4];
        "#;
        let program = ast::parse(source_code);
        check(&program).unwrap();
    }

    #[test]
    fn test_inline_array_init_invalid_type() {
        let source_code = r#"
            arr:int[] = [1,"2"];
        "#;
        let program = ast::parse(source_code);
        assert!(check(&program).is_err());
    }

    #[test]
    fn test_access_slice() {
        let source_code = r#"
             arr:int[] = [1];
             arr[0];
        "#;
        let program = ast::parse(source_code);
        check(&program).unwrap();
    }

    #[test]
    fn test_variable_scope_nested_block() {
        let source_code = r#"
        a: int = 5;
        {
            a: string = "nested";
            print(a); // Should refer to 'string' type
        }
        print(a); // Should refer to 'int' type
        "#;
        let program = ast::parse(source_code);
        check(&program).unwrap();
    }

    #[test]
    fn test_lambda_var() {
        let source_code = r#"
            id: (int) -> int = function (a:int): int {
                return a;
            }
        "#;
        let program = ast::parse(source_code);
        check(&program).unwrap();
    }

    #[test]
    fn test_lambda_var_invalid_type() {
        let source_code = r#"
            id: (string) -> int = function (a:int): int {
                return a;
            }
        "#;
        let program = ast::parse(source_code);
        assert!(check(&program).is_err());
    }
    #[test]
    fn test_anonymous_function() {
        let source_code = r#"
        function f(g: () -> int): int {
            return g();
        }

        f(function ():int {
            return 47;
        });
        "#;

        let program = ast::parse(source_code);
        check(&program).unwrap();
    }

    #[test]
    fn test_anonymous_function_invalid_type() {
        let source_code = r#"
        function f(g: (int) -> int): int {
            return g("1");
        }

        f(function (i:int):int {
            return i;
        });
        "#;

        let program = ast::parse(source_code);
        assert!(check(&program).is_err());
    }

    #[test]
    fn test_function_returns_function() {
        let source_code = r#"
        function f(): (int) -> int {
            return function (a:int): int {
                return a;
            };
        }
        g: (int) -> int = f();
        g(1);
        "#;
        let program = ast::parse(source_code);
        check(&program).unwrap();
    }

    #[test]
    fn test_lambda() {
        let source_code = r#"
        function f(times:int): () -> int {
            count: int = 0;
            g: () -> int = function(): int {
                count = count + 1;
                if (count == times) {
                    return count;
                } else {
                    return g();
                }
            }
            return g;
        }
        f(3)();
        "#;

        let program = ast::parse(source_code);
        check(&program).unwrap();
    }

    #[test]
    fn test_nested_lambda() {
        let source_code = r#"
        nested: () -> () -> int = function(): () -> int {
            inner_count: int = 0;
            return function(): int {
                inner_count = inner_count + 1;
                return inner_count;
            };
        }
        "#;
        let program = ast::parse(source_code);
        check(&program).unwrap();
    }

    #[test]
    fn test_nested_lambdas() {
        let source_code = r#"
        i: int = 0;

        a: () -> () -> int = function(): () -> int {
            i = i + 1;
            j: int = 0;

            b: () -> () -> int = function(): () -> int {
                i = i + 1;
                j = j + 1;
                k: int = 0;

                c: () -> int = function(): int {
                    i = i + 1;
                    j = j + 1;
                    k = k + 1;
                    return i + j + k;
                }

                return c;
            }

            return b();
        }

        f: () -> int = a();
        res: int = f();
        "#;
        let program = ast::parse(source_code);
        check(&program).unwrap();
    }

    #[test]
    fn test_return_lambda() {
        let source_code = r#"
        function counter(): () -> int {
        c: int = 0;
        return function(): int {
            c = c + 1;
            return c;
        };
        }
        count: () -> int = counter();
        "#;
        let program = ast::parse(source_code);
        check(&program).unwrap();
    }

    #[test]
    fn test_missing_return_in_if() {
        let source_code = r#"
        function f(a: int): int {
            if (a > 0) {
                return a;
            }
        }
        "#;
        let program = ast::parse(source_code);
        assert!(check(&program).is_err());
    }

    #[test]
    fn test_missing_return_in_else() {
        let source_code = r#"
        function f(a: int): int {
            if (a > 0) {
                return a;
            } else {
                print("log");
            }
        }
        "#;
        let program = ast::parse(source_code);
        assert!(check(&program).is_err());
    }

    #[test]
    fn test_complete_return_in_if_else() {
        let source_code = r#"
        function f(a: int): int {
            if (a > 0) {
                return a;
            } else {
                return 0;
            }
        }
        "#;
        let program = ast::parse(source_code);
        check(&program).unwrap();
    }

    #[test]
    fn test_return_in_nested_if() {
        let source_code = r#"
        function f(a: int, b: int): int {
            if (a > 0) {
                if (b > 0) {
                    return a + b;
                } else {
                    return a;
                }
            } else {
                return 0;
            }
        }
        "#;
        let program = ast::parse(source_code);
        check(&program).unwrap();
    }

    #[test]
    fn test_missing_return_in_nested_if() {
        let source_code = r#"
        function f(a: int, b: int): int {
            if (a > 0) {
                if (b > 0) {
                    return a + b;
                }
            } else {
                return 0;
            }
        }
        "#;
        let program = ast::parse(source_code);
        assert!(check(&program).is_err());
    }

    #[test]
    fn test_return_in_for_loop() {
        let source_code = r#"
        function f(limit: int): int {
            for (i: int = 0; i < limit; i = i + 1) {
                if (i == 5) {
                    return i;
                }
            }
            return -1;
        }
        "#;
        let program = ast::parse(source_code);
        check(&program).unwrap();
    }

    #[test]
    fn test_missing_return_in_for_loop() {
        let source_code = r#"
        function f(limit: int): int {
            for (i: int = 0; i < limit; i = i + 1) {
                if (i == 5) {
                    return i;
                }
            }
        }
        "#;
        let program = ast::parse(source_code);
        assert!(check(&program).is_err());
    }

    #[test]
    fn test_return_in_nested_for_loops() {
        let source_code = r#"
        function f(): int {
            for (i: int = 0; i < 10; i = i + 1) {
                for (j: int = 0; j < 10; j = j + 1) {
                    if (i + j == 10) {
                        return i * 10 + j;
                    }
                }
            }
            return -1;
        }
        "#;
        let program = ast::parse(source_code);
        check(&program).unwrap();
    }

    #[test]
    fn test_empty_for_loop_body() {
        let source_code = r#"
        function f(limit: int): int {
            for (i: int = 0; i < limit; i = i + 1) {}
            return -1;
        }
        "#;
        let program = ast::parse(source_code);
        check(&program).unwrap();
    }

    #[test]
    fn test_for_loop_without_condition() {
        let source_code = r#"
        function f(): int {
            for (i: int = 0; ; i = i + 1) {
                if (i == 10) {
                    return i;
                }
            }
        }
        "#;
        let program = ast::parse(source_code);
        assert!(check(&program).is_err());
    }

    #[test]
    fn test_for_loop_without_update() {
        let source_code = r#"
        function f(): int {
            for (i: int = 0; i < 10; ) {
                if (i == 5) {
                    return i;
                }
                i = i + 1; // Manual update
            }
            return -1;
        }
        "#;
        let program = ast::parse(source_code);
        check(&program).unwrap();
    }

    #[test]
    fn test_for_loop_without_initialization() {
        let source_code = r#"
        function f(): int {
            i: int = 0;
            for (; i < 10; i = i + 1) {
                if (i == 5) {
                    return i;
                }
            }
            return -1;
        }
        "#;
        let program = ast::parse(source_code);
        check(&program).unwrap();
    }

    #[test]
    fn test_complex_for_loop_with_nested_control() {
        let source_code = r#"
        function f(): int {
            for (i: int = 0; i < 10; i = i + 1) {
                if (i % 2 == 0) {
                    for (j: int = 0; j < 5; j = j + 1) {
                        if (j == 3) {
                            return i * 10 + j;
                        }
                    }
                }
            }
            return -1;
        }
        "#;
        let program = ast::parse(source_code);
        check(&program).unwrap();
    }

    // #[test] todo
    fn test_unreachable_code_after_for_loop() {
        let source_code = r#"
        function f(): int {
            for (i: int = 0; i < 10; i = i + 1) {
                return i;
            }
            // This code should be unreachable
            return -1;
        }
        "#;
        let program = ast::parse(source_code);
        assert!(check(&program).is_err());
    }

    #[test]
    fn test_function_call_inside_control_flow() {
        let source_code = r#"
        function g(): int {
            return 42;
        }

        function f(a: int): int {
            if (a > 0) {
                return g();
            } else {
                return 0;
            }
        }
        "#;
        let program = ast::parse(source_code);
        check(&program).unwrap();
    }

    #[test]
    fn test_new_primitives_and_numeric_widening() {
        let source_code = r#"
        function main(): double {
            b: byte = 10;
            s: short = 20;
            i: int = b + s;
            l: long = i + 30L;
            f: float = 1.5f;
            c: char = 'A';
            ok: bool = c == 'A';
            d: double = l + f;
            return d;
        }
        "#;
        let program = ast::parse(source_code);
        check(&program).unwrap();
    }

    #[test]
    fn test_float_literal_requires_suffix() {
        let source_code = r#"
        function main(): void {
            f: float = 1.5;
        }
        "#;
        let program = ast::parse(source_code);
        assert!(check(&program).is_err());
    }
}
