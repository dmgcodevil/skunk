use std::any::Any;
use std::collections::HashMap;

use crate::ast;
use ast::Literal;
use ast::Node;
use ast::Operator;
use ast::Type;
use std::cell::{Ref, RefCell};
use std::fmt;
use std::io::BufRead;
use std::rc::Rc;

#[derive(Debug, PartialEq, Clone)]
pub enum Value {
    Integer(i64),
    String(String),
    Boolean(bool),
    Variable(String),
    Array {
        arr: Vec<Rc<RefCell<Value>>>,
        dimensions: Vec<i64>,
    },
    StructInstance {
        name: String,
        fields: HashMap<String, Rc<RefCell<Value>>>,
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
    Undefined,
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
            Value::Array { arr, dimensions } => write!(f, "array dimensions={:?}", dimensions),
            Void => write!(f, ""),
            _ => write!(f, "{:?}", "unknown"),
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

// it should be scoped: module, package, struct, function
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

#[derive(Debug, PartialEq, Clone)]
struct Environment {
    variables: HashMap<String, Rc<RefCell<Value>>>,
}

fn get_element_from_array(value: &Value, coordinates: &Vec<i64>) -> Rc<RefCell<Value>> {
    if let Value::Array {
        ref arr,
        ref dimensions,
    } = value
    {
        let pos = to_1d_pos(coordinates, dimensions);
        Rc::clone(&arr[pos])
    } else {
        panic!("not an array")
    }
}

fn set_array_element(value: &mut Value, new_value: Rc<RefCell<Value>>, coordinates: &Vec<i64>) {
    if let Value::Array {
        ref mut arr,
        ref dimensions,
    } = value
    {
        let pos = to_1d_pos(coordinates, dimensions);
        arr[pos] = new_value;
    }
}

fn to_1d_pos(coordinates: &Vec<i64>, dimensions: &Vec<i64>) -> usize {
    if coordinates.len() != dimensions.len() {
        panic!(
            "invalid dimensions. expected={}, actual={}",
            dimensions.len(),
            coordinates.len()
        );
    }
    let mut res: usize = 0;
    let mut multiplier: i64 = 1;

    for i in (0..dimensions.len()).rev() {
        if coordinates[i] >= dimensions[i] {
            panic!(
                "invalid coordinate. expected less than {}, actual={}. pos={}",
                dimensions[i], coordinates[i], i
            );
        }
        res += (coordinates[i] * multiplier) as usize;
        multiplier *= dimensions[i];
    }

    res
}

fn set_nested_field(instance: &Rc<RefCell<Value>>, parts: &[&str], value: Value) {
    if parts.len() == 1 {
        // Base case: reached the field to set
        if let Value::StructInstance { fields, .. } = instance.borrow().deref() {
            if let Some(field) = fields.get(parts[0]) {
                *field.borrow_mut() = value;
            } else {
                panic!("field {} not found", parts[0]);
            }
        } else {
            panic!("expected struct instance, found something else");
        }
    } else {
        if let Value::StructInstance { fields, .. } = instance.borrow().deref() {
            if let Some(nested_instance) = fields.get(parts[0]) {
                set_nested_field(nested_instance, &parts[1..], value);
            } else {
                panic!("field {} not found", parts[0]);
            }
        } else {
            panic!("expected struct instance, found something else");
        }
    }
}

fn get_struct_field_value<'a>(instance: &Rc<RefCell<Value>>, parts: &[&str]) -> Rc<RefCell<Value>> {
    if let Value::StructInstance { fields, .. } = instance.borrow().deref() {
        if parts.len() == 1 {
            Rc::clone(fields.get(parts[0]).unwrap())
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

    fn declare_variable(&mut self, name: &str, value: Rc<RefCell<Value>>) {
        // todo asserts
        self.variables.insert(name.to_string(), value);
    }
    fn assign_variable(&mut self, name: &str, value: Rc<RefCell<Value>>) {
        // todo asserts
        self.variables.insert(name.to_string(), value);
    }

    fn get_variable_value(&self, name: &str) -> Rc<RefCell<Value>> {
        if let Some(var) = self.variables.get(name) {
            Rc::clone(var)
        } else {
            panic!("variable {} not declared", name);
        }
    }
}

#[derive(Debug, Clone)]
pub struct CallFrame {
    name: String,
    locals: Environment,
}

#[derive(Debug, Clone)]
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

    fn current_frame(&self) -> &CallFrame {
        self.frames
            .last()
            .expect("Call stack underflow: No active frames")
    }

    fn current_frame_mut(&mut self) -> &mut CallFrame {
        self.frames
            .last_mut()
            .expect("Call stack underflow: No active frames")
    }
}

pub fn evaluate(node: &Node) -> Rc<RefCell<Value>> {
    let stack = Rc::new(RefCell::new(CallStack::new()));
    let ge = Rc::new(RefCell::new(GlobalEnvironment::new()));
    stack.borrow_mut().push(CallFrame {
        name: "main".to_string(),
        locals: Environment::new(),
    });
    evaluate_node(node, &stack, &ge)
}

fn get_value_by_name(
    stack: &Rc<RefCell<CallStack>>,
    global_environment: &Rc<RefCell<GlobalEnvironment>>,
    current_value: &Value,
    name: &str,
) -> Rc<RefCell<Value>> {
    let frame = stack.borrow().current_frame();
    match current_value {
        Value::Undefined => Rc::new(RefCell::new(Value::Undefined)),
        Value::StructInstance { name, fields } => Rc::clone(fields.get(name).unwrap()),
        _ => panic!("expected current_value={:?}", current_value),
    }
}

fn assert_value_is_struct(v: &Value) {
    if !matches!(v, StructInstance { .. }) {
        panic!("expected struct instance, found {:?}", v);
    }
}

fn get_struct_fields_mut(v: &mut Value) -> &mut HashMap<String, Rc<RefCell<Value>>> {
    if let StructInstance { name, fields } = v {
        fields
    } else {
        panic!("expected struct instance, found {:?}", v);
    }
}

fn set_or_get_value(
    stack: &Rc<RefCell<CallStack>>,
    global_environment: &Rc<RefCell<GlobalEnvironment>>,
    current_value: Rc<RefCell<Value>>,
    i: usize,
    access_nodes: &Vec<Node>,
    new_value_opt: Option<Rc<RefCell<Value>>>,
) -> Rc<RefCell<Value>> {
    let mut current_value_borrow = current_value.borrow_mut();
    let access_node = &access_nodes[i];
    if let Node::MemberAccess { member } = access_node {
        assert_value_is_struct(current_value_borrow.deref());
        let mut struct_fields = get_struct_fields_mut(current_value_borrow.deref_mut());
        if let Node::Identifier(name) = member.as_ref() {
            if i == access_nodes.len() - 1 {
                if let Some(new_value) = new_value_opt {
                    struct_fields.insert(name.to_string(), new_value.clone());
                    return Rc::new(RefCell::new(Undefined)); // todo return new value or struct instance ?
                } else {
                    struct_fields.get(name).unwrap().clone();
                }
            }
            return set_or_get_value(
                stack,
                global_environment,
                struct_fields.get(name).unwrap().clone(),
                i + 1,
                access_nodes,
                new_value_opt,
            );
        } else if let Node::FunctionCall { name, arguments } = member.as_ref() {
            if new_value_opt.is_some() {
                // not allowed to do `f() = new_val`
                panic!("Cannot assign value to function call");
            }
            if i == access_nodes.len() - 1 {
                // todo
                // or introduce scopes...
                // call function within struct instance scope
                return evaluate_node(member.as_ref(), stack, global_environment);
            }
            return set_or_get_value(
                stack,
                global_environment,
                evaluate_node(member.as_ref(), stack, global_environment).clone(),
                i + 1,
                access_nodes,
                new_value_opt,
            );
        }
    } else if let Node::ArrayAccess { coordinates } = access_node {
        let _coordinates: Vec<i64> = coordinates
            .iter()
            .map(
                |c| match evaluate_node(c, stack, global_environment).borrow().deref() {
                    Value::Integer(dim) => *dim,
                    _ => panic!("expected a number to access array"),
                },
            )
            .collect();
        if let Value::Array { .. } = current_value_borrow.deref() {
            if i == access_nodes.len() - 1 {
                if let Some(new_value) = new_value_opt {
                    set_array_element(
                        current_value_borrow.deref_mut(),
                        Rc::clone(&new_value),
                        &_coordinates,
                    );
                    return Rc::new(RefCell::new(Undefined)); // or return new_value or array
                } else {
                    return Rc::clone(&get_element_from_array(
                        current_value_borrow.deref(),
                        &_coordinates,
                    ));
                }
            }
            return set_or_get_value(
                stack,
                global_environment,
                Rc::clone(&get_element_from_array(
                    current_value_borrow.deref(),
                    &_coordinates,
                )),
                i + 1,
                access_nodes,
                new_value_opt,
            );
        } else {
            panic!(
                "expected array value, found {:?}",
                current_value_borrow.deref()
            );
        }
    } else if let Node::Identifier(name) = access_node {
        println!("name={:?}", name);
        let mut stack_ref = stack.borrow_mut();
        let mut frame = stack_ref.current_frame_mut();
        if i == access_nodes.len() - 1 {
            if let Some(new_value) = new_value_opt {
                frame.locals.assign_variable(name, Rc::clone(&new_value));
                return Rc::new(RefCell::new(Undefined)); // todo return new value ?
            } else {
                return Rc::clone(&frame.locals.get_variable_value(name));
            }
        } else {
            return set_or_get_value(
                stack,
                global_environment,
                Rc::clone(&frame.locals.get_variable_value(name)),
                i + 1,
                access_nodes,
                new_value_opt,
            );
        }
    }
    panic!("unreachable code")
}

pub fn evaluate_node(
    node: &Node,
    stack: &Rc<RefCell<CallStack>>,
    global_environment: &Rc<RefCell<GlobalEnvironment>>,
) -> Rc<RefCell<Value>> {
    match node {
        Node::Program { statements } => {
            let mut last = Rc::new(RefCell::new(Void));
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

            global_environment
                .borrow_mut()
                .add_struct(name.to_string(), sd.clone());
            Rc::new(RefCell::new(Value::Struct {
                name: name.to_string(),
                fields: sd.fields.clone(),
            }))
        }
        Node::FunctionDeclaration {
            name,
            parameters,
            return_type,
            body,
        } => {
            global_environment.borrow_mut().functions.insert(
                name.clone(),
                FunctionDefinition {
                    name: name.to_string(),
                    parameters: parameters.clone(),
                    return_type: return_type.clone(),
                    body: body.clone(),
                },
            );
            Rc::new(RefCell::new(Value::Function {
                name: name.to_string(),
                parameters: parameters.clone(),
                return_type: return_type.clone(),
            }))
        }
        Node::Literal(Literal::Integer(x)) => Rc::new(RefCell::new(Value::Integer(*x))),
        Node::Literal(Literal::StringLiteral(x)) => {
            Rc::new(RefCell::new(Value::String(x.to_string())))
        }
        Node::Literal(Literal::Boolean(x)) => Rc::new(RefCell::new(Value::Boolean(*x))),
        Node::VariableDeclaration { name, value, .. } => {
            let value = if let Some(body) = value {
                evaluate_node(body, stack, global_environment)
            } else {
                Rc::new(RefCell::new(Undefined))
            };
            stack
                .borrow_mut()
                .current_frame_mut()
                .locals
                .declare_variable(name, value);
            Rc::new(RefCell::new(Value::Variable(name.to_string())))
        }
        Node::StructInitialization { name, fields } => {
            let mut field_values: HashMap<String, Rc<RefCell<Value>>> = HashMap::new();
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
            Rc::new(RefCell::new(value))
        }
        Node::Access { nodes } => set_or_get_value(
            stack,
            global_environment,
            Rc::new(RefCell::new(Undefined)),
            0,
            nodes,
            None,
        )
        .clone(),
        Node::Assignment { var, value } => match var.as_ref() {
            Node::Access { nodes } => set_or_get_value(
                stack,
                global_environment,
                Rc::new(RefCell::new(Undefined)),
                0,
                nodes,
                Some(Rc::clone(&evaluate_node(
                    value.as_ref(),
                    stack,
                    global_environment,
                ))),
            )
            .clone(),
            _ => panic!("expected access to assignment node "),
        },
        Node::Access { nodes } => set_or_get_value(
            stack,
            global_environment,
            Rc::new(RefCell::new(Undefined)),
            0,
            nodes,
            None,
        )
        .clone(),
        Node::FunctionCall { name, arguments } => {
            let args_values: Vec<_> = arguments
                .into_iter()
                .map(|arg| {
                    evaluate_node(arg, stack, global_environment)
                        .clone()
                        .borrow()
                        .deref()
                        .clone()
                })
                .collect();
            let fun = global_environment
                .borrow()
                .get_function(name)
                .unwrap()
                .clone();
            let parameters: Vec<(&(String, Type), Value)> =
                fun.parameters.iter().zip(args_values.into_iter()).collect();
            let mut frame = CallFrame {
                name: name.to_string(),
                locals: Environment::new(),
            };
            for p in parameters {
                let first = p.0.clone().0;
                let second = p.1;
                frame
                    .locals
                    .variables
                    .insert(first, Rc::new(RefCell::new(second)));
            }
            stack.borrow_mut().push(frame);

            let mut result = Rc::new(RefCell::new(Void));
            for n in fun.body {
                if let Value::Return(v) = evaluate_node(&n, stack, global_environment)
                    .borrow()
                    .deref()
                {
                    result = Rc::new(RefCell::new(v.as_ref().clone()));
                }
            }
            stack.borrow_mut().pop();

            result
        }
        Node::StaticFunctionCall {
            _type,
            name,
            arguments,
        } => match _type {
            Type::Array {
                elem_type,
                dimensions,
            } => {
                let mut size: usize = 1;
                for d in dimensions {
                    size = size * (*d) as usize;
                }
                let mut arr = Vec::with_capacity(size);
                for i in 0..size {
                    arr.push(evaluate_node(&arguments[0], stack, global_environment));
                }
                Rc::new(RefCell::new(Value::Array {
                    arr,
                    dimensions: dimensions.clone(),
                }))
            }
            _ => panic!("unsupported static function call type"),
        },
        Node::If {
            condition,
            body,
            else_if_blocks,
            else_block,
        } => {
            if let Value::Boolean(ok) =
                *evaluate_node(condition.as_ref(), stack, global_environment)
                    .borrow()
                    .deref()
            {
                if ok {
                    for n in body {
                        let n_val = evaluate_node(n, stack, global_environment);
                        let val = n_val.borrow().deref().clone();
                        if let Value::Return(v) = val {
                            return Rc::new(RefCell::new(Value::Return(v.clone())));
                        }
                    }
                    return Rc::new(RefCell::new(Value::Executed));
                }
            } else {
                panic!("non boolean expression: {:?}", condition.as_ref())
            }

            for n in else_if_blocks {
                let val_ref = evaluate_node(n, stack, global_environment);
                let val = val_ref.borrow().deref().clone();
                if let Value::Return(result) = val {
                    return Rc::clone(&val_ref);
                }
                if let Value::Executed = val {
                    return Rc::clone(&val_ref);
                }
            }

            if let Some(nodes) = else_block {
                for n in nodes {
                    let val_ref = evaluate_node(n, stack, global_environment);
                    let val = val_ref.borrow().deref().clone();
                    if let Value::Return(result) = val {
                        return Rc::clone(&val_ref);
                    }
                    if let Value::Executed = val {
                        return Rc::clone(&val_ref);
                    }
                }
            }

            Rc::new(RefCell::new(Void))
        }
        Node::For {
            init,
            condition,
            update,
            body,
        } => {
            if let Some(n) = init {
                evaluate_node(n.as_ref(), stack, global_environment);
            }
            while condition
                .as_ref()
                .map(|cond| {
                    let binding = evaluate_node(cond.as_ref(), stack, global_environment);
                    let res = binding.borrow().deref().clone();
                    match res {
                        Value::Boolean(v) => v,
                        _ => panic!("for condition in should be a boolean expression"),
                    }
                })
                .unwrap_or(true)
            {
                for n in body {
                    let val = evaluate_node(n, stack, global_environment);
                    let val_ref = val.borrow();
                    if matches!(val_ref.deref(), Value::Return(_)) {
                        return Rc::clone(&val);
                    }
                }
                if let Some(n) = update {
                    evaluate_node(n, stack, global_environment);
                }
            }
            Rc::new(RefCell::new(Void))
        }
        Node::BinaryOp {
            left,
            operator,
            right,
        } => {
            let left_val = evaluate_node(left, stack, global_environment);
            let right_val = evaluate_node(right, stack, global_environment);
            let left_val_ref = left_val.borrow();
            let right_val_ref = right_val.borrow();
            match (left_val_ref.deref(), right_val_ref.deref()) {
                (Value::Integer(l), Value::Integer(r)) => match operator {
                    Operator::Add => Rc::new(RefCell::new(Value::Integer(l + r))),
                    Operator::Subtract => Rc::new(RefCell::new(Value::Integer(l - r))),
                    Operator::Multiply => Rc::new(RefCell::new(Value::Integer(l * r))),
                    Operator::Divide => Rc::new(RefCell::new(Value::Integer(l / r))),
                    Operator::LessThanOrEqual => Rc::new(RefCell::new(Value::Boolean(l <= r))),
                    Operator::LessThan => Rc::new(RefCell::new(Value::Boolean(l < r))),
                    Operator::GreaterThanOrEqual => Rc::new(RefCell::new(Value::Boolean(l >= r))),
                    Operator::GreaterThan => Rc::new(RefCell::new(Value::Boolean(l > r))),
                    _ => panic!("Unsupported operator"),
                },
                (Value::Boolean(a), Value::Boolean(b)) => match operator {
                    Operator::And => Rc::new(RefCell::new(Value::Boolean(*a && *b))),
                    Operator::Or => Rc::new(RefCell::new(Value::Boolean(*a || *b))),
                    _ => panic!("Unsupported operator"),
                },
                _ => panic!("Unsupported binary operation"),
            }
        }
        Node::Print(value) => {
            let val = evaluate_node(value, stack, global_environment);
            println!("{}", val.borrow());
            Rc::new(RefCell::new(Void))
        }
        Node::Identifier(name) => {
            /*
            why we need stack_ref as a separate var:
            The issue you're encountering stems from the fact that calling stack.borrow() creates a
            temporary value (Ref from the RefCell), and when you chain .current_frame() on it,
            that temporary Ref is dropped at the end of the statement, leading to a lifetime issue.
            */
            let stack_ref = stack.borrow(); // Borrow the RefCell
            let current_frame = stack_ref.current_frame(); // Access the current frame with a valid borrow
            let v = current_frame.locals.get_variable_value(name);
            v.clone() // Return the cloned value
        }
        Node::EOI => Rc::new(RefCell::new(Void)),
        Node::Return(n) => {
            let val = evaluate_node(n, stack, global_environment);
            let val_ref = val.borrow();
            Rc::new(RefCell::new(Value::Return(Box::new(val_ref.clone()))))
        }
        _ => panic!("Unexpected node type: {:?}", node),
    }
}

use crate::interpreter::Value::{StructInstance, Undefined, Void};
use rustyline::error::ReadlineError;
use rustyline::{DefaultEditor, Result};
use std::cmp;
use std::fmt::Octal;
use std::ops::{Deref, DerefMut};

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
    fn test_array_modify() {
        let source_code = r#"
         arr: int[5] = int[5]::new(1);
         arr[0] = 2;
         arr[0];
        "#;
        let program = ast::parse(source_code);
        let res = evaluate(&program);
        let res_ref = res.borrow();
        assert_eq!(Value::Integer(2), *res_ref.deref());
    }

    #[test]
    fn test_2d_array_modify() {
        let source_code = r#"
         arr: int[2][3] = int[2][3]::new(1);
         arr[1][2] = 2;
         arr[1][2];
        "#;
        let program = ast::parse(source_code);
        let res = evaluate(&program);
        let res_ref = res.borrow();
        assert_eq!(Value::Integer(2), *res_ref.deref());
    }

    #[test]
    fn test_struct_int_field_modify() {
        let source_code = r#"
            struct Point {
                x:int;
                y:int;
            }
            p: Point = Point{x:1, y:2};
            p.x = 3;
            p.y = 4;
            p;
        "#;
        let program = ast::parse(source_code);
        let res = evaluate(&program);
        let res_ref = res.borrow();
        assert_eq!(
            Value::StructInstance {
                name: "Point".to_string(),
                fields: HashMap::from([
                    ("x".to_string(), Rc::new(RefCell::new(Value::Integer(3)))),
                    ("y".to_string(), Rc::new(RefCell::new(Value::Integer(4))))
                ])
            },
            *res_ref.deref()
        )
    }

    #[test]
    fn test_struct_arr_field_modify() {
        let source_code = r#"
            struct Point {
                c: int[2];
            }
            p: Point = Point{ c: int[2]::new(0) };
            p.c[0] = 1;
            p.c[1] = 2;
            p;
        "#;
        let program = ast::parse(source_code);
        let res = evaluate(&program);
        let res_ref = res.borrow();
        assert_eq!(
            StructInstance {
                name: "Point".to_string(),
                fields: HashMap::from([(
                    "c".to_string(),
                    Rc::new(RefCell::new(Value::Array {
                        arr: [
                            Rc::new(RefCell::new(Value::Integer(1))),
                            Rc::new(RefCell::new(Value::Integer(2)))
                        ]
                        .to_vec(),
                        dimensions: [2].to_vec()
                    }))
                )])
            },
            *res_ref
        )
    }

    // fixme introduce scopes
    // #[test]
    fn test_array_dynamic_init() {
        let source_code = r#"
         i: int = 0;
         function incAndGet():int {
            i = i + 1;
            return i;
         }
         arr: int[5] = int[5]::new(incAndGet());
         arr[0];
        "#;
        let program = ast::parse(source_code);
        println!("{:?}", evaluate(&program));
    }
}
