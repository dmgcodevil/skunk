use std::collections::HashMap;

use crate::ast;
use ast::Literal;
use ast::Node;
use ast::Operator;
use ast::Type;
use std::fmt;
use std::io::BufRead;

#[derive(Clone, Debug)]
pub enum Value {
    Integer(i64),
    String(String),
    Boolean(bool),
    Variable(String),
    StructInstance {
        name: String,
        fields: HashMap<String, Value>,
    },
    Struct {
        name: String,
        fields: Vec<(String, Type)>,
    },
    Function {
        name: String,
        parameters: Vec<(String, Type)>,
        return_type: Type,
    },
    Return(Box<Value>),
    Executed, // indicates that a branch has been executed
    Void,
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Value::Return(_) => write!(f, "return"),
            Value::Executed => write!(f, "executed"),
            Value::Variable(v) => write!(f, "{}", v), // todo unwrap
            Value::Integer(v) => write!(f, "{}", v),
            Value::String(v) => write!(f, "{}", v),
            Value::Boolean(v) => write!(f, "{}", v),
            Value::Function {
                name,
                parameters,
                return_type,
            } => {
                let params_str = parameters
                    .into_iter()
                    .map(|(n, t)| format!("{}:{}", n, ast::type_to_string(t)))
                    .collect::<Vec<_>>()
                    .join(", ");
                write!(
                    f,
                    "function {}({}):{}",
                    name.to_string(),
                    params_str,
                    ast::type_to_string(return_type)
                )
            }
            Value::StructInstance {
                name: struct_name,
                fields,
            } => {
                write!(f, "struct {} {{}}", struct_name)
            }
            &Value::Struct { .. } => todo!(),
            Value::Void => write!(f, ""),
        }
    }
}

#[derive(Clone, Debug)]
struct StructDefinition {
    name: String,
    fields: Vec<(String, Type)>,
}

#[derive(Clone, Debug)]
struct FunctionDefinition {
    name: String,
    parameters: Vec<(String, Type)>,
    return_type: Type,
    body: Vec<Node>,
}

struct GlobalEnvironment {
    structs: HashMap<String, StructDefinition>,
    functions: HashMap<String, FunctionDefinition>,
}

impl GlobalEnvironment {
    fn new() -> Self {
        GlobalEnvironment {
            structs: HashMap::new(),
            functions: HashMap::new(),
        }
    }

    fn add_struct(&mut self, name: String, definition: StructDefinition) {
        self.structs.insert(name, definition);
    }

    fn get_struct(&self, name: &str) -> Option<&StructDefinition> {
        self.structs.get(name)
    }

    fn add_function(&mut self, name: String, definition: FunctionDefinition) {
        self.functions.insert(name, definition);
    }

    fn get_function(&self, name: &str) -> Option<&FunctionDefinition> {
        self.functions.get(name)
    }
}

struct Environment {
    variables: HashMap<String, Value>,
}

fn set_nested_field(instance: &mut Value, parts: &[&str], value: Value) {
    if parts.len() == 1 {
        // Base case: reached the field to set
        if let Value::StructInstance { fields, .. } = instance {
            if let Some(field) = fields.get_mut(parts[0]) {
                *field = value;
            } else {
                panic!("field {} not found", parts[0]);
            }
        } else {
            panic!("expected struct instance, found something else");
        }
    } else {
        if let Value::StructInstance { fields, .. } = instance {
            if let Some(nested_instance) = fields.get_mut(parts[0]) {
                set_nested_field(nested_instance, &parts[1..], value);
            } else {
                panic!("field {} not found", parts[0]);
            }
        } else {
            panic!("expected struct instance, found something else");
        }
    }
}

fn get_struct_field_value<'a>(instance: &'a Value, parts: &[&str]) -> &'a Value {
    if let Value::StructInstance { fields, .. } = instance {
        if parts.len() == 1 {
            fields.get(parts[0]).unwrap()
        } else {
            get_struct_field_value(fields.get(parts[0]).unwrap(), &parts[1..])
        }
    } else {
        panic!("expected struct instance, found {:?}", instance);
    }
}


impl Environment {
    fn new() -> Self {
        Environment {
            variables: HashMap::new(),
        }
    }

    fn declare_variable(&mut self, name: &str, value: Value) {
        self.variables.insert(name.to_string(), value);
    }

    fn assign_variable(&mut self, name: &str, value: Value) {
        let parts: Vec<&str> = name.split('.').collect();
        let size = parts.len();

        if size == 1 {
            // Base case: if no nested fields, assign directly
            if self.variables.contains_key(name) {
                self.variables.insert(name.to_string(), value);
            } else {
                panic!("variable {} not declared", name);
            }
        } else {
            let root_name = parts[0];
            if let Some(root_instance) = self.variables.get_mut(root_name) {
                set_nested_field(root_instance, &parts[1..], value);
            } else {
                panic!("variable {} not declared", root_name);
            }
        }
    }

    fn get_variable_value<'a>(&'a self, name: &'a str) -> &'a Value {
        let parts: Vec<&str> = name.split(".").collect();
        if parts.len() == 1 {
            self.variables
                .get(name)
                // .cloned()
                .unwrap_or_else(|| panic!("variable {} not found", name))
        } else {
            let root_name = parts[0];
            if let Some(root_instance) = self.variables.get(root_name) {
                get_struct_field_value(root_instance, &parts[1..])
            } else {
                panic!("variable {} not declared", root_name);
            }
        }
    }
}

pub struct CallFrame {
    name: String,
    locals: Environment,
}

pub struct CallStack {
    frames: Vec<CallFrame>,
}

impl CallStack {
    pub fn new() -> Self {
        CallStack { frames: Vec::new() }
    }

    fn push(&mut self, frame: CallFrame) {
        self.frames.push(frame);
    }

    fn pop(&mut self) -> Option<CallFrame> {
        self.frames.pop()
    }

    fn current_frame(&mut self) -> &mut CallFrame {
        self.frames
            .last_mut()
            .expect("Call stack underflow: No active frames")
    }
}

pub fn evaluate(node: &Node) -> Value {
    let mut stack = CallStack::new();
    let mut ge = GlobalEnvironment::new();
    stack.push(CallFrame {
        name: "main".to_string(),
        locals: Environment::new(),
    });
    evaluate_node(node, &mut stack, &mut ge)
}

pub fn evaluate_node(
    node: &Node,
    stack: &mut CallStack,
    global_environment: &mut GlobalEnvironment,
) -> Value {
    match node {
        Node::Program { statements } => {
            let mut last = Value::Void;
            for i in 0..statements.len() - 1 {
                last = evaluate_node(&statements[i], stack, global_environment);
            }
            last
        }
        Node::StructDeclaration { name, declarations } => {
            let sd = StructDefinition {
                name: name.to_string(),
                fields: declarations.clone(),
            };
            let val = Value::Struct {
                name: name.to_string(),
                fields: sd.fields.clone(),
            };
            global_environment.add_struct(name.to_string(), sd);
            val
        }
        Node::FunctionDeclaration {
            name,
            parameters,
            return_type,
            body,
        } => {
            // if name == "main" {
            //     stack.push(CallFrame {
            //         name: name.clone(),
            //         locals: Environment::new(),
            //     });
            //
            //     for statement in body {
            //         evaluate_node(statement, stack, global_environment);
            //     }
            //     stack.pop();
            // }

            global_environment.functions.insert(
                name.clone(),
                FunctionDefinition {
                    name: name.to_string(),
                    parameters: parameters.clone(),
                    return_type: return_type.clone(),
                    body: body.clone(),
                },
            );
            Value::Function {
                name: name.to_string(),
                parameters: parameters.clone(),
                return_type: return_type.clone(),
            }
        }
        Node::Literal(Literal::Integer(x)) => Value::Integer(*x),
        Node::VariableDeclaration { name, value, .. } => {
            let value = evaluate_node(value, stack, global_environment);
            stack
                .current_frame()
                .locals
                .declare_variable(name, value.clone());
            Value::Variable(name.to_string())
        }
        Node::StructInitialization { name, fields } => {
            let mut field_values: HashMap<String, Value> = HashMap::new();
            for (name, node) in fields {
                field_values.insert(
                    name.to_string(),
                    evaluate_node(node, stack, global_environment),
                );
            }
            let value = Value::StructInstance {
                name: name.to_string(),
                fields: field_values,
            };
            // stack
            //     .current_frame()
            //     .locals
            //     .declare_variable(name, value.clone());
            value
        }
        Node::Assignment { identifier, value } => {
            let val = evaluate_node(value, stack, global_environment);
            stack
                .current_frame()
                .locals
                .assign_variable(identifier, val.clone());
            val
        }
        Node::FunctionCall { name, arguments } => {
            let args_values: Vec<_> = arguments
                .into_iter()
                .map(|arg| evaluate_node(arg, stack, global_environment))
                .collect();
            let fun = global_environment.get_function(name).unwrap().clone();
            let parameters: Vec<(&(String, Type), Value)> =
                fun.parameters.iter().zip(args_values.into_iter()).collect();
            let mut frame = CallFrame {
                name: name.to_string(),
                locals: Environment::new(),
            };
            for p in parameters {
                let first = p.0.clone().0;
                let second = p.1;
                frame.locals.variables.insert(first, second);
            }
            stack.push(frame);

            let mut result = Value::Void;
            for n in fun.body {
                if let Value::Return(v) = evaluate_node(&n, stack, global_environment) {
                    result = v.as_ref().clone();
                }
            }
            stack.pop();

            result
        }
        Node::If {
            condition,
            body,
            else_if_blocks,
            else_block,
        } => {
            if let Value::Boolean(ok) = evaluate_node(condition.as_ref(), stack, global_environment)
            {
                if ok {
                    for n in body {
                        if let Value::Return(v) = evaluate_node(n, stack, global_environment) {
                            return Value::Return(v);
                        }
                    }
                    return Value::Executed;
                }
            } else {
                panic!("non boolean expression: {:?}", condition.as_ref())
            }

            for n in else_if_blocks {
                let v = evaluate_node(n, stack, global_environment);
                if let Value::Return(result) = v {
                    return Value::Return(result);
                }
                if let Value::Executed = v {
                    return Value::Executed;
                }
            }

            if let Some(nodes) = else_block {
                for n in nodes {
                    let v = evaluate_node(n, stack, global_environment);
                    if let Value::Return(result) = v {
                        return Value::Return(result);
                    }
                    if let Value::Executed = v {
                        return Value::Executed;
                    }
                }
            }

            Void
        }
        Node::For {
            init,
            condition,
            update,
            body
        } => {
            if let Some(n) = init {
                evaluate_node(n.as_ref(), stack, global_environment);
            }
            while condition.as_ref().map(|cond| {
                let res = evaluate_node(cond.as_ref(), stack, global_environment);
                match res {
                    Value::Boolean(v) => v,
                    _ => panic!("for condition in should be a boolean expression")
                }
            }).unwrap_or(true) {
                for n in body {
                    if let Value::Return(v) = evaluate_node(n, stack, global_environment) {
                        return Value::Return(v);
                    }
                }
                if let Some(n) = update {
                    evaluate_node(n, stack, global_environment);
                }
            }
            Value::Void
        }
        Node::BinaryOp {
            left,
            operator,
            right,
        } => {
            let left_val = evaluate_node(left, stack, global_environment);
            let right_val = evaluate_node(right, stack, global_environment);
            match (left_val, right_val) {
                (Value::Integer(l), Value::Integer(r)) => match operator {
                    Operator::Add => Value::Integer(l + r),
                    Operator::Subtract => Value::Integer(l - r),
                    Operator::Multiply => Value::Integer(l * r),
                    Operator::Divide => Value::Integer(l / r),
                    Operator::LessThanOrEqual => Value::Boolean(l <= r),
                    Operator::LessThan => Value::Boolean(l < r),
                    Operator::GreaterThanOrEqual => Value::Boolean(l >= r),
                    Operator::GreaterThan => Value::Boolean(l > r),
                    _ => panic!("Unsupported operator"),
                },
                (Value::Boolean(a), Value::Boolean(b)) => match operator {
                    Operator::And => Value::Boolean(a && b),
                    Operator::Or => Value::Boolean(a || b),
                    _ => panic!("Unsupported operator"),
                },
                _ => panic!("Unsupported binary operation"),
            }
        }
        Node::Print(value) => {
            let val = evaluate_node(value, stack, global_environment);
            println!("{}", val);
            Value::Void
        }
        Node::Identifier(name) => {
            let v = stack.current_frame().locals.get_variable_value(name);
            v.clone()   // todo get rid of clone()
        }
        Node::EOI => Value::Void,
        Node::Return(n) => Value::Return(Box::new(evaluate_node(n, stack, global_environment))),
        _ => panic!("Unexpected node type: {:?}", node),
    }
}

use crate::interpreter::Value::Void;
use rustyline::error::ReadlineError;
use rustyline::{DefaultEditor, Result};
use std::cmp;
use std::fmt::{write, Octal};

pub fn repl() -> Result<()> {
    let mut stack = CallStack::new();
    let mut ge = GlobalEnvironment::new();
    stack.push(CallFrame {
        name: "main".to_string(),
        locals: Environment::new(),
    });

    let mut rl = DefaultEditor::new()?;
    println!("Welcome to Skunk REPL!");
    println!("Type your code and press Enter to execute. Use Shift+Enter to insert a new line.");
    println!("Press Ctrl-C to exit.");

    loop {
        let mut input = String::new();
        let mut prompt = ">>> ".to_string(); // Primary prompt
        let mut indent_level = 0;
        loop {
            // Read a line of input with the prompt
            match rl.readline(&prompt) {
                Ok(line) => {
                    if line.trim().is_empty() {
                        // empty line indicates the end of input
                        if !input.trim().is_empty() {
                            // Add the input to history
                            rl.add_history_entry(input.trim_end());
                            // process the input
                            // println!("{}", &input);

                            let program = ast::parse(&input);
                            println!("{:?}", evaluate(&program));
                        }
                        // Reset the prompt and break to start new input
                        prompt = ">>> ".to_string();
                        break;
                    } else {
                        // append the line to the input buffer
                        input.push_str(&line);
                        input.push('\n');

                        for ch in line.chars() {
                            match ch {
                                '{' => indent_level += 1,
                                '}' => indent_level = cmp::max(0, indent_level - 1),
                                _ => {}
                            }
                        }

                        prompt = format!("{}... ", "  ".repeat(indent_level));
                    }
                }
                Err(ReadlineError::Interrupted) => {
                    println!("^C");
                    break;
                }
                Err(ReadlineError::Eof) => {
                    println!("^D");
                    break;
                }
                Err(err) => {
                    println!("Error: {:?}", err);
                    break;
                }
            }
        }
    }
}

mod tests {
    use super::*;

    #[test]
    fn test_struct() {
        let source_code = r#"
         struct Foo {
            i: int;
         }
          f: Foo = Foo{i:1};
        "#;

        let program = ast::parse(source_code);
        println!("{:?}", evaluate(&program));
    }

    #[test]
    fn test_fn_declaration() {
        let source_code = r#"
            function max(a:int, b:int):int {
                if (a > b) {
                    return a;
                } else if (b > a){
                    return b;
                } else {
                    return a;
                }
            }
            print(max(2, 2));
        "#;
        let program = ast::parse(source_code);
        // println!("{:?}", program);
        println!("{}", evaluate(&program));
    }

    #[test]
    fn test_assign_instance_field() {
        let source_code = r#"
            struct Foo {
                i:int;
            }
            f:Foo = Foo{i:1};
            f.i = 2;
            print(f.i);
        "#;
        let program = ast::parse(source_code);
        println!("{:?}", program);
        println!("{}", evaluate(&program));
    }
    #[test]
    fn test_early_return() {
        let source_code = r#"

        "#;
    }

    #[test]
    fn test_for1() {
        let source_code = r#"
        for(i:int=0; i < 5; i = i + 1) {
            print(i);
        }
      "#;
        let program = ast::parse(source_code);
        println!("{:?}", program);
        println!("{}", evaluate(&program));
    }

    #[test]
    fn test_for2() {
        let source_code = r#"
        i: int = 0;
        for(i=1; i < 5; i = i + 1) {
            print(i);
        }
      "#;
        let program = ast::parse(source_code);
        println!("{:?}", program);
        println!("{}", evaluate(&program));
    }

    #[test]
    fn test_for3() {
        let source_code = r#"
        i: int = 0;
        for(i=1; i < 5;) {
            print(i);
            i = i + 1;
        }
      "#;
        let program = ast::parse(source_code);
        println!("{:?}", program);
        println!("{}", evaluate(&program));
    }
}
