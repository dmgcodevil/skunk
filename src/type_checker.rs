use crate::ast::{Literal, Metadata, Node, Operator, Type};
use std::collections::HashMap;
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

#[derive(Debug, PartialEq, Clone)]
struct FunctionSymbol {
    name: String,
    parameters: Vec<Symbol>,
    return_type: Type,
    // metadata: Metadata, todo
}

#[derive(Debug, PartialEq, Clone)]
struct StructSymbol {
    name: String,
    fields: Vec<Symbol>,
    functions: HashMap<String, FunctionSymbol>,
    // metadata: Metadata, todo
}

#[derive(Debug, PartialEq, Clone)]
struct GlobalScope {
    structs: HashMap<String, StructSymbol>,
    functions: HashMap<String, FunctionSymbol>,
    variables: HashMap<String, Symbol>,
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

fn resolve_type(var_tables: &mut VarTables, node: &Node) -> Result<Type, String> {
    match node {
        Node::Program { statements } => {
            for statement in statements {
                let res = resolve_type(var_tables, statement);
                match &res {
                    Err(e) => return Err(e.to_string()),
                    _ => (),
                }
            }
            Ok(Type::Void)
        }
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
                res = resolve_type(var_tables, n);
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
                let value_type = resolve_type(var_tables, &body.deref())?;
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
        Node::FunctionCall { name, arguments } => Ok(Type::Void),
        Node::BinaryOp {
            left,
            operator,
            right,
        } => {
            let left_type = resolve_type(var_tables, left.deref())?;
            let right_type = resolve_type(var_tables, right.deref())?;
            match operator {
                Operator::Add => resolve_add(&left_type, &right_type),
                _ => unreachable!("todo"),
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
            .ok_or(format!("unknown variable '{}'", name)),
        Node::Access { nodes } => {
            // todo member access, array, etc.
            let mut res = Ok(Type::Void);
            for node in nodes {
                res = resolve_type(var_tables, node);
            }
            res
        }
        Node::Return(body) => resolve_type(var_tables, body),
        _ => Ok(Type::Void),
    }
}

pub fn check(node: &Node) -> Result<(), String> {
    let mut var_tables = VarTables::new();
    var_tables.add(VarTable::new());
    match resolve_type(&mut var_tables, node) {
        Err(e) => Err(e.to_string()),
        Ok(_) => Ok(()),
    }
}

mod tests {
    use super::*;
    use crate::ast;

    #[test]
    fn test_var_decl() {
        let source_code = r#"
        function bar(i:int):int {
            return j;
        }
        "#;
        let program = ast::parse(source_code);
    }
}
