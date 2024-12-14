use crate::ast::Type::Custom;
use crate::ast::{Literal, Metadata, Node, Operator, Type};
use std::collections::HashMap;
use std::fmt::format;
use std::ops::Deref;

#[derive(Debug, PartialEq, Clone)]
struct Symbol {
    name: String,
    sk_type: Type,
    metadata: Metadata, // we can say where var is defined
}

#[derive(Debug, PartialEq, Clone)]
struct VarTable {
    vars: HashMap<String, Symbol>,
}

impl VarTable {
    fn new() -> Self {
        VarTable {
            vars: HashMap::new(),
        }
    }

    fn add(&mut self, var: Symbol) {
        self.vars.insert(var.name.clone(), var);
    }
}

struct VarTables {
    tables: Vec<VarTable>,
}

impl VarTables {
    fn new() -> Self {
        VarTables { tables: vec![] }
    }
    fn add(&mut self, var_table: VarTable) {
        self.tables.push(var_table);
    }

    fn pop(&mut self) {
        self.tables.pop();
    }

    fn get(&self) -> &VarTable {
        &self.tables.last().unwrap()
    }
    fn get_mut(&mut self) -> &mut VarTable {
        self.tables.last_mut().unwrap()
    }
}

fn create_symbols(declarations: &Vec<(String, Type)>) -> Vec<Symbol> {
    declarations
        .iter()
        .map(|d| Symbol {
            name: d.0.clone(),
            sk_type: d.1.clone(),
            metadata: Metadata::EMPTY,
        })
        .collect()
}

#[derive(Debug, PartialEq, Clone)]
struct FunctionSymbol {
    name: String,
    parameters: Vec<Symbol>,
    return_type: Type,
    // metadata: Metadata, todo
}

impl FunctionSymbol {
    fn from_node(node: &Node) -> Self {
        match node {
            Node::FunctionDeclaration {
                name,
                parameters,
                return_type,
                ..
            } => FunctionSymbol {
                name: name.clone(),
                parameters: create_symbols(parameters),
                return_type: return_type.clone(),
            },
            _ => panic!("expected function declaration"),
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
struct StructSymbol {
    name: String,
    fields: HashMap<String, Symbol>,
    functions: HashMap<String, FunctionSymbol>,
    // metadata: Metadata, todo
}

#[derive(Debug, PartialEq, Clone)]
struct GlobalScope {
    structs: HashMap<String, StructSymbol>,
    functions: HashMap<String, FunctionSymbol>,
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
                                        metadata: Metadata::EMPTY,
                                    },
                                )
                            })
                            .collect::<HashMap<_, _>>(),
                        functions: functions
                            .iter()
                            .map(|n| {
                                let fun_symbol = FunctionSymbol::from_node(n);
                                (fun_symbol.name.clone(), fun_symbol)
                            })
                            .collect::<HashMap<_, _>>(),
                    },
                );
            }
            Node::FunctionDeclaration {
                name,
                parameters,
                return_type,
                ..
            } => {
                self.functions
                    .insert(name.clone(), FunctionSymbol::from_node(node));
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
                        metadata: metadata.clone(),
                    },
                );
            }
            _ => {}
        }
    }
}

fn resolve_add(left: &Type, right: &Type) -> Result<Type, String> {
    match (left, right) {
        (Type::Int, Type::Int) => Ok(Type::Int),
        (Type::String, Type::String) => Ok(Type::String),
        _ => Err(format!(
            "unexpected types for +: {:?} and {:?}",
            left, right
        )),
    }
}

fn resolve_access(
    global_scope: &GlobalScope,
    var_tables: &mut VarTables,
    curr: Type,
    i: usize,
    access_nodes: &Vec<Node>,
) -> Result<Type, String> {
    if i == access_nodes.len() {
        return Ok(curr);
    }
    match access_nodes.get(i).unwrap() {
        Node::ArrayAccess { .. } => {
            if let Type::Array { elem_type, .. } = curr {
                return resolve_access(
                    global_scope,
                    var_tables,
                    elem_type.deref().clone(),
                    i + 1,
                    access_nodes,
                );
            } else {
                return Err(format!("array access to not array variable"));
            }
        }

        Node::MemberAccess { member, metadata } => match curr {
            Type::Custom(struct_name) => {
                if !global_scope.structs.contains_key(&struct_name) {
                    return Err("error: struct doesn't exist".to_string());
                }
                let struct_symbol = global_scope.structs.get(&struct_name).unwrap();
                match member.deref() {
                    Node::Identifier(field_name) => {
                        if !struct_symbol.fields.contains_key(field_name) {
                            // error[E0609]: no field `z` on type `Point`
                            // --> src/main.rs:10:22
                            // |
                            // 10 |     println!("{}", p.z);
                            // |                      ^ unknown field
                            return Err(format!(
                                "error {}:{}: no field `{}` on type `{}`",
                                metadata.span.line, metadata.span.start, field_name, struct_name
                            ));
                        }
                        return resolve_access(
                            global_scope,
                            var_tables,
                            struct_symbol
                                .fields
                                .get(field_name)
                                .unwrap()
                                .sk_type
                                .clone(),
                            i + 1,
                            access_nodes,
                        );
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
                            var_tables,
                            struct_symbol.functions.get(name).unwrap(),
                            member.deref(),
                        )?;
                        // global_scope: &GlobalScope,
                        // var_tables: &mut VarTables,
                        // node: &Node,
                        let args_types_res: Result<Vec<Type>, String> = arguments
                            .iter()
                            .map(|arg| resolve_type(global_scope, var_tables, arg))
                            .collect();
                        let args_types = args_types_res?;
                        println!("args_types = {:?}", args_types);

                        return resolve_access(
                            global_scope,
                            var_tables,
                            return_type,
                            i + 1,
                            access_nodes,
                        );
                    }
                    _ => panic!("expected member access node"),
                }
            }
            _ => return Err(format!("access to member access to not instance structs")),
        },
        _ => unreachable!("unexpected access node"),
    }
}

fn resolve_function_call(
    global_scope: &GlobalScope,
    var_tables: &mut VarTables,
    function_symbol: &FunctionSymbol,
    node: &Node,
) -> Result<Type, String> {
    if let Node::FunctionCall {
        name,
        arguments,
        metadata,
    } = node
    {
        if arguments.len() != function_symbol.parameters.len() {
            return Err(format!(
                "incorrect number of args to '{}' function. \
                    expected={}, actual={}",
                name,
                function_symbol.parameters.len(),
                arguments.len()
            ));
        }
        let mut argument_types = Vec::new();
        for arg in arguments {
            argument_types.push(resolve_type(global_scope, var_tables, arg)?);
        }
        for i in 0..argument_types.len() {
            if argument_types[i] != function_symbol.parameters[i].sk_type {
                return Err(format!(
                    "error {}:{}:arguments to '{}' function are incorrect. parameter='{}', \
                        expected type='{:?}', actual type='{:?}'",
                    metadata.span.line,
                    metadata.span.start,
                    name,
                    function_symbol.parameters[i].name,
                    function_symbol.parameters[i].sk_type,
                    argument_types[i]
                ));
            }
        }
        Ok(function_symbol.return_type.clone())
    } else {
        panic!("incorrect node")
    }
}

fn resolve_type(
    global_scope: &GlobalScope,
    var_tables: &mut VarTables,
    node: &Node,
) -> Result<Type, String> {
    match node {
        Node::Program { statements } => {
            for statement in statements {
                let res = resolve_type(global_scope, var_tables, statement);
                match &res {
                    Err(e) => return Err(e.to_string()),
                    _ => (),
                }
            }
            Ok(Type::Void)
        }
        Node::StructInitialization { name, .. } => {
            if !global_scope.structs.contains_key(name) {
                Err("struct doesn't exist".to_string())
            } else {
                Ok(Custom(name.clone()))
            }
        }
        Node::StructDeclaration { name, .. } => Ok(Custom(name.clone())),
        Node::FunctionDeclaration {
            name,
            parameters,
            return_type,
            body,
        } => {
            let mut var_table = VarTable::new();
            for parameter in parameters {
                var_table.add(Symbol {
                    name: parameter.0.clone(),
                    sk_type: parameter.1.clone(),
                    metadata: Metadata::EMPTY, // todo
                })
            }
            var_tables.add(var_table);
            let mut actual_return_type = Type::Void;
            let mut res = Ok(Type::Void);
            for n in body {
                res = resolve_type(global_scope, var_tables, n);
                match &res {
                    Err(e) => break,
                    Ok(t) => match n {
                        Node::Return(_) => {
                            actual_return_type = (*t).clone();
                            break;
                        }
                        _ => (),
                    },
                }
            }
            var_tables.pop();
            if let Err(_) = res {
                return res;
            }
            if actual_return_type != *return_type {
                Err(format!(
                    "function '{}' return type mismatch. expected={:?}, actual={:?}",
                    name, return_type, actual_return_type
                ))
            } else {
                Ok(actual_return_type)
            }
        }
        Node::VariableDeclaration {
            var_type,
            name,
            value,
            metadata,
        } => {
            var_tables.get_mut().add(Symbol {
                name: name.clone(),
                sk_type: var_type.clone(),
                metadata: metadata.clone(),
            });
            if let Some(body) = value {
                let value_type = resolve_type(global_scope, var_tables, &body.deref())?;
                if *var_type != value_type {
                    Err(format!(
                        "expected '{:?}' var type: {:?}, actual: {:?}. pos: {:?}",
                        name.clone(),
                        var_type,
                        value_type.clone(),
                        metadata
                    ))
                } else {
                    Ok(var_type.clone())
                }
            } else {
                Ok(var_type.clone())
            }
        }
        Node::FunctionCall {
            name, arguments, ..
        } => resolve_function_call(
            global_scope,
            var_tables,
            global_scope.functions.get(name).unwrap(),
            node,
        ),
        Node::BinaryOp {
            left,
            operator,
            right,
        } => {
            let left_type = resolve_type(global_scope, var_tables, left.deref())?;
            let right_type = resolve_type(global_scope, var_tables, right.deref())?;
            match operator {
                Operator::Add => resolve_add(&left_type, &right_type),
                _ => unreachable!("todo"),
            }
        }
        Node::Assignment { var, value, .. } => {
            let var_type = resolve_type(global_scope, var_tables, var.deref())?;
            let value_type = resolve_type(global_scope, var_tables, value.deref())?;
            if var_type != value_type {
                // todo include var name, metadata
                Err(format!(
                    "assignment type mismatch. expected: {:?}, actual: {:?}",
                    var_type, value_type
                ))
            } else {
                Ok(var_type.clone())
            }
        }
        Node::Literal(Literal::Integer(_)) => Ok(Type::Int),
        Node::Literal(Literal::Boolean(_)) => Ok(Type::Boolean),
        Node::Literal(Literal::StringLiteral(_)) => Ok(Type::String),
        Node::EOI => Ok(Type::Void),
        Node::Identifier(name) => var_tables
            .get()
            .vars
            .get(name)
            .map(|v| v.sk_type.clone())
            .or(global_scope.variables.get(name).map(|v| v.sk_type.clone()))
            .ok_or(format!("unknown variable '{}'", name)),
        Node::Access { nodes } => {
            let start = resolve_type(global_scope, var_tables, nodes.get(0).unwrap())?;
            resolve_access(global_scope, var_tables, start, 1, nodes)
        }
        Node::StaticFunctionCall {
            _type,
            name,
            arguments,
            metadata,
        } => {
            match _type {
                Type::Void => {}
                Type::Int => {}
                Type::String => {}
                Type::Boolean => {}
                Type::Array { elem_type, .. } => {
                    if arguments.len() != 1 {
                        return Err(format!(
                            "error {}:{}: array `new` method should have exactly one argument",
                            metadata.span.line, metadata.span.start,
                        ));
                    }
                    let arg_type =
                        resolve_type(global_scope, var_tables, arguments.get(0).unwrap())?;
                    if arg_type != *elem_type.deref() {
                        return Err(format!("error {}:{}: array `new`. expected arg type is `{:?}` but given `{:?}`",
                                           metadata.span.line,
                                           metadata.span.start,
                                           elem_type.deref(), arg_type ));
                    }
                }
                Type::Slice { .. } => {}
                Custom(_) => {}
                Type::Function { .. } => {}
                Type::SkSelf => {}
            }
            Ok(_type.clone())
        }
        Node::Return(body) => resolve_type(global_scope, var_tables, body),
        Node::Block { statements } => {
            let var_table = var_tables.get().clone();
            var_tables.add(var_table);

            for statement in statements {
                let res = resolve_type(global_scope, var_tables, statement);
                if let Err(_) = res {
                    return res;
                }
            }
            Ok(Type::Void) // do we need to check Return node ?
        }
        Node::Print(n) => resolve_type(global_scope, var_tables, n.deref()),
        _ => unreachable!("{}", format!("{:?}", node)),
    }
}

pub fn check(node: &Node) -> Result<(), String> {
    let mut var_tables = VarTables::new();
    var_tables.add(VarTable::new());
    let mut global_scope = GlobalScope::new();
    match node {
        Node::Program { statements } => {
            for statement in statements {
                global_scope.add(statement);
            }
        }
        _ => panic!("expected program node"),
    }
    match resolve_type(&mut global_scope, &mut var_tables, node) {
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

            function set_x(x:int) {
                self.x = x;
            }

            function get_x():int {
               return self.x;
            }

            function set_y(y:int) {
                self.y = y;
            }

            function get_y():int {
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
    foo("string"); // This should fail
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
        return "string"; // Should fail since return type is expected to be int
    }
    "#;
        let program = ast::parse(source_code);
        assert!(check(&program).is_err());
    }

    #[test]
    fn test_array_access_wrong_type() {
        let source_code = r#"
    arr: int[5] = int[5]::new(1);
    s: string = arr[0]; // Should fail: assigning int to string
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
}
