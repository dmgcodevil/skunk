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
    Return(Rc<RefCell<Value>>),
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
        let struct_fields = get_struct_fields_mut(current_value_borrow.deref_mut());
        if let Node::Identifier(name) = member.as_ref() {
            if i == access_nodes.len() - 1 {
                return if let Some(new_value) = new_value_opt {
                    struct_fields.insert(name.to_string(), new_value.clone());
                    Rc::new(RefCell::new(Value::Undefined))
                } else {
                    struct_fields
                        .get(name)
                        .unwrap_or_else(|| panic!("field `{}` not found", name))
                        .clone()
                };
            }
            return set_or_get_value(
                stack,
                global_environment,
                struct_fields
                    .get(name)
                    .unwrap_or_else(|| panic!("field `{}` not found", name))
                    .clone(),
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
                    _ => panic!("expected integer index for array access"),
                },
            )
            .collect();
        if let Value::Array { .. } = current_value_borrow.deref() {
            if i == access_nodes.len() - 1 {
                return if let Some(new_value) = new_value_opt {
                    set_array_element(
                        current_value_borrow.deref_mut(),
                        Rc::clone(&new_value),
                        &_coordinates,
                    );
                    Rc::new(RefCell::new(Value::Undefined))
                } else {
                    Rc::clone(&get_element_from_array(
                        current_value_borrow.deref(),
                        &_coordinates,
                    ))
                };
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
        if i == access_nodes.len() - 1 {
            return if let Some(new_value) = new_value_opt {
                stack
                    .borrow_mut()
                    .current_frame_mut()
                    .locals
                    .assign_variable(name, Rc::clone(&new_value));
                Rc::new(RefCell::new(Undefined)) // todo return new value ?
            } else {
                Rc::clone(
                    &stack
                        .borrow()
                        .current_frame()
                        .locals
                        .get_variable_value(name),
                )
            };
        } else {
            return set_or_get_value(
                stack,
                global_environment,
                Rc::clone(
                    &stack
                        .borrow()
                        .current_frame()
                        .locals
                        .get_variable_value(name),
                ),
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
    println!("evaluate_node={:?}", node);
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
                .map(|arg| evaluate_node(arg, stack, global_environment))
                .collect();
            let global_environment_ref = global_environment.borrow();
            let fun = global_environment_ref.get_function(name).unwrap();
            let parameters: Vec<(&(String, Type), Rc<RefCell<Value>>)> =
                fun.parameters.iter().zip(args_values.into_iter()).collect();
            let mut frame = CallFrame {
                name: name.to_string(),
                locals: Environment::new(),
            };
            for p in parameters {
                let first = p.0.clone().0;
                let second = p.1;
                frame.locals.variables.insert(first, Rc::clone(&second));
            }
            stack.borrow_mut().push(frame);

            let mut result = Rc::new(RefCell::new(Undefined));
            for n in &fun.body {
                if let Value::Return(v) = evaluate_node(&n, stack, global_environment)
                    .borrow()
                    .deref()
                {
                    result = Rc::clone(v);
                    break;
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
                        let val = evaluate_node(n, stack, global_environment);
                        let val_ref = val.borrow();
                        if let Value::Return(return_val) = val_ref.deref() {
                            return Rc::new(RefCell::new(Value::Return(Rc::clone(&return_val))));
                        }
                    }
                    return Rc::new(RefCell::new(Value::Executed));
                }
            } else {
                panic!("non boolean expression: {:?}", condition.as_ref())
            }

            for n in else_if_blocks {
                let val = evaluate_node(n, stack, global_environment);
                let val_ref = val.borrow();
                if let Value::Return(return_val) = val_ref.deref() {
                    return Rc::new(RefCell::new(Value::Return(Rc::clone(&return_val))));
                }
                if let Value::Executed = val_ref.deref() {
                    return Rc::new(RefCell::new(Value::Executed));
                }
            }

            if let Some(nodes) = else_block {
                for n in nodes {
                    let val_ref = evaluate_node(n, stack, global_environment);
                    let val = val_ref.borrow().deref().clone();
                    if let Value::Return(return_val) = val {
                        return Rc::new(RefCell::new(Return(Rc::clone(&return_val))));
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
                    if let Value::Return(return_val) = val_ref.deref() {
                        return Rc::new(RefCell::new(Value::Return(Rc::clone(&return_val))));
                    }
                }
                if let Some(n) = update {
                    evaluate_node(n, stack, global_environment);
                }
            }
            Rc::new(RefCell::new(Void))
        }
        Node::UnaryOp { operator, operand } => {
            let operand_val = evaluate_node(operand, stack, global_environment);
            match operator {
                ast::UnaryOperator::Plus => operand_val, // unary `+` doesn't change the value
                ast::UnaryOperator::Minus => match *operand_val.borrow() {
                    Value::Integer(val) => Rc::new(RefCell::new(Value::Integer(-val))),
                    _ => panic!("Unary minus is only supported for integers"),
                },
            }
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
                    Operator::Mod => Rc::new(RefCell::new(Value::Integer(l % r))),
                    Operator::Equals => Rc::new(RefCell::new(Value::Boolean(l == r))),
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
                _ => panic!(
                    "Unsupported binary operation={:?}, left={:?}, right={:?}",
                    operator, left_val_ref, right_val_ref
                ),
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
            Rc::clone(&v)
        }
        Node::EOI => Rc::new(RefCell::new(Void)),
        Node::Return(n) => {
            let val = evaluate_node(n, stack, global_environment);
            Rc::new(RefCell::new(Return(Rc::clone(&val))))
        }
        _ => panic!("Unexpected node type: {:?}", node),
    }
}

use crate::interpreter::Value::{Executed, Return, StructInstance, Undefined, Void};
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

    #[test]
    fn test_if() {
        let source_code = r#"
            function max(a:int, b:int):int {
                if (a > b) {
                    return a;
                } else {
                    return b;
                }
            }
            max(2, 3);
        "#;
        let program = ast::parse(source_code);
        let res = evaluate(&program);
        let res_ref = res.borrow();
        assert_eq!(Value::Integer(3), *res_ref.deref());
    }

    #[test]
    fn test_nested_if() {
        let parameters = [
            (1, 2, 3),
            (1, 3, 2),
            (2, 1, 3),
            (2, 3, 1),
            (3, 1, 2),
            (3, 2, 1),
        ];

        for p in &parameters {
            let source_code = format!(
                r#"
             function max(a:int, b:int, c:int):int {{
                if (a > b) {{
                    if (a > c) {{
                        return a;
                    }} else {{
                        return c;
                    }}
                }} else {{
                    if (b > c) {{
                        return b;
                    }} else {{
                        return c;
                    }}
                }}
            }}
            max({}, {}, {});
            "#,
                p.0, p.1, p.2
            );
            let program = ast::parse(source_code.as_str());
            let res = evaluate(&program);
            let res_ref = res.borrow();
            assert_eq!(Value::Integer(3), *res_ref.deref());
        }
    }
    #[cfg(test)]
    mod tests {
        use super::*;
        use std::collections::HashMap;

        #[test]
        fn test_for_loop_simple_increment() {
            let source_code = r#"
            a:int = 0;
            for (i:int = 0; i < 5; i = i + 1) {
                a = a + 1;
            }
            a;
        "#;
            let program = ast::parse(source_code);
            let res = evaluate(&program);
            let res_ref = res.borrow();
            assert_eq!(Value::Integer(5), *res_ref.deref());
        }

        #[test]
        fn test_for_loop_with_array_modification() {
            let source_code = r#"
            arr: int[5] = int[5]::new(0);
            for (i:int = 0; i < 5; i = i + 1) {
                arr[i] = i * 2;
            }
            arr[4];
        "#;
            let program = ast::parse(source_code);
            let res = evaluate(&program);
            let res_ref = res.borrow();
            assert_eq!(Value::Integer(8), *res_ref.deref());
        }

        #[test]
        fn test_nested_for_loops_2d_array() {
            let source_code = r#"
            arr: int[2][2] = int[2][2]::new(0);
            for (i:int = 0; i < 2; i = i + 1) {
                for (j:int = 0; j < 2; j = j + 1) {
                    arr[i][j] = i + j;
                }
            }
            arr[1][1];
        "#;
            let program = ast::parse(source_code);
            let res = evaluate(&program);
            let res_ref = res.borrow();
            assert_eq!(Value::Integer(2), *res_ref.deref());
        }

        #[test]
        fn test_struct_with_array_field() {
            let source_code = r#"
            struct Container {
                values: int[3];
            }
            c: Container = Container{ values: int[3]::new(1) };
            c.values[1] = 2;
            c.values[2] = 3;
            c.values[1];
        "#;
            let program = ast::parse(source_code);
            let res = evaluate(&program);
            let res_ref = res.borrow();
            assert_eq!(Value::Integer(2), *res_ref.deref());
        }

        #[test]
        fn test_array_return_value_from_function() {
            let source_code = r#"
            function createArray(): int[3] {
                arr: int[3] = int[3]::new(1);
                return arr;
            }
            arr = createArray();
            arr[2];
        "#;
            let program = ast::parse(source_code);
            let res = evaluate(&program);
            let res_ref = res.borrow();
            assert_eq!(Value::Integer(1), *res_ref.deref());
        }

        #[test]
        fn test_struct_instance_with_nested_structs() {
            let source_code = r#"
            struct Inner {
                value: int;
            }
            struct Outer {
                inner: Inner;
            }
            outer: Outer = Outer{ inner: Inner{ value: 10 } };
            outer.inner.value;
        "#;
            let program = ast::parse(source_code);
            let res = evaluate(&program);
            let res_ref = res.borrow();
            assert_eq!(Value::Integer(10), *res_ref.deref());
        }

        #[test]
        fn test_function_returning_struct_instance() {
            let source_code = r#"
            struct Point {
                x: int;
                y: int;
            }
            function createPoint(): Point {
                return Point{ x: 10, y: 20 };
            }
            p: Point = createPoint();
            p.y;
        "#;
            let program = ast::parse(source_code);
            let res = evaluate(&program);
            let res_ref = res.borrow();
            assert_eq!(Value::Integer(20), *res_ref.deref());
        }

        #[test]
        fn test_array_out_of_bounds_error() {
            let source_code = r#"
            arr: int[3] = int[3]::new(1);
            arr[3] = 10;
        "#;
            let program = ast::parse(source_code);
            let result = std::panic::catch_unwind(|| {
                evaluate(&program);
            });
            assert!(result.is_err());
        }

        #[test]
        fn test_modify_struct_array_field_through_for_loop() {
            let source_code = r#"
            struct Container {
                values: int[3];
            }
            c: Container = Container{ values: int[3]::new(0) };
            for (i:int = 0; i < 3; i = i + 1) {
                c.values[i] = i * 2;
            }
            c.values[2];
        "#;
            let program = ast::parse(source_code);
            let res = evaluate(&program);
            let res_ref = res.borrow();
            assert_eq!(Value::Integer(4), *res_ref.deref());
        }

        #[test]
        fn test_return_void_function_call_in_expression() {
            let source_code = r#"
            function setValues(a:int[3]):void {
                a[0] = 10;
            }
            arr: int[3] = int[3]::new(1);
            setValues(arr);
            arr[0];
        "#;
            let program = ast::parse(source_code);
            let res = evaluate(&program);
            let res_ref = res.borrow();
            assert_eq!(Value::Integer(10), *res_ref.deref());
        }
    }

    #[test]
    fn test_mod() {
        let source_code = r#"
            function isEven(i: int):boolean {
                if (i % 2 == 0) {
                    return true;
                } else {
                    return false;
                }
            }
            isEven(3);
        "#;
        let program = ast::parse(source_code);
        let res = evaluate(&program);
        println!("{:?}", res);
    }

    #[test]
    fn test_return_from_for_loop() {
        let source_code = r#"
            function findFirstEven(limit: int): int {
                for (i:int = 0; i < limit; i = i + 1) {
                    if (i % 2 == 0) {
                        return i;
                    }
                }
                return -1;
            }
            findFirstEven(5);
        "#;
        let program = ast::parse(source_code);
        let res = evaluate(&program);
        let res_ref = res.borrow();
        assert_eq!(Value::Integer(0), *res_ref.deref());
    }

    #[test]
    fn test_return_from_nested_for_loop() {
        let source_code = r#"
            function findFirstMatch(): int {
                for (i:int = 0; i < 3; i = i + 1) {
                    for (j:int = 0; j < 3; j = j + 1) {
                        if (i + j == 2) {
                            return i * 10 + j;
                        }
                    }
                }
                return -1;
            }
            findFirstMatch();
        "#;
        let program = ast::parse(source_code);
        let res = evaluate(&program);
        let res_ref = res.borrow();
        assert_eq!(Value::Integer(2), *res_ref.deref());
    }

    #[test]
    fn test_for_with_nested_if() {
        let source_code = r#"
            function countEvenOdds(limit: int): int {
                even_count: int = 0;
                odd_count: int = 0;
                for (i:int = 0; i < limit; i = i + 1) {
                    if (i % 2 == 0) {
                        even_count = even_count + 1;
                    } else {
                        odd_count = odd_count + 1;
                    }
                }
                return even_count * 10 + odd_count;
            }
            countEvenOdds(5);
        "#;
        let program = ast::parse(source_code);
        let res = evaluate(&program);
        let res_ref = res.borrow();
        assert_eq!(Value::Integer(30 + 2), *res_ref.deref());
    }

    #[test]
    fn test_for_inside_if() {
        let source_code = r#"
            function conditionalLoop(cond: bool): int {
                sum: int = 0;
                if (cond) {
                    for (i:int = 1; i <= 3; i = i + 1) {
                        sum = sum + i;
                    }
                } else {
                    sum = -1;
                }
                return sum;
            }
            conditionalLoop(true);
        "#;
        let program = ast::parse(source_code);
        let res = evaluate(&program);
        let res_ref = res.borrow();
        assert_eq!(Value::Integer(6), *res_ref.deref());
    }

    #[test]
    fn test_nested_for_with_early_return_in_if() {
        let source_code = r#"
            function complexLoop(): int {
                for (i:int = 0; i < 4; i = i + 1) {
                    for (j:int = 0; j < 4; j = j + 1) {
                        if (i * j == 6) {
                            return i * 10 + j;
                        }
                    }
                }
                return -1;
            }
            complexLoop();
        "#;
        let program = ast::parse(source_code);
        let res = evaluate(&program);
        let res_ref = res.borrow();
        assert_eq!(Value::Integer(23), *res_ref.deref()); // i=2, j=3 gives 6
    }

    #[test]
    fn test_for_loop_with_if_break_condition() {
        let source_code = r#"
            function sumUntilLimit(limit: int): int {
                sum: int = 0;
                for (i:int = 1; i < 10; i = i + 1) {
                    if (sum + i > limit) {
                        return sum;
                    }
                    sum = sum + i;
                }
                return sum;
            }
            sumUntilLimit(10);
        "#;
        let program = ast::parse(source_code);
        let res = evaluate(&program);
        let res_ref = res.borrow();
        assert_eq!(Value::Integer(10), *res_ref.deref()); // Sum is 1+2+3 before hitting limit
    }

    #[test]
    fn test_for_with_nested_for_and_if_with_else() {
        let source_code = r#"
            function complexLoopWithElse(): int {
                sum: int = 0;
                for (i:int = 1; i <= 3; i = i + 1) {
                    for (j:int = 1; j <= 3; j = j + 1) {
                        if (i == j) {
                            sum = sum + 10 * i;
                        } else {
                            sum = sum + j;
                        }
                    }
                }
                return sum;
            }
            complexLoopWithElse();
        "#;
        let program = ast::parse(source_code);
        let res = evaluate(&program);
        let res_ref = res.borrow();
        assert_eq!(Value::Integer(72), *res_ref.deref());
    }

    #[test]
    fn test_return_in_for_inside_if_with_fallback() {
        let source_code = r#"
            function loopInsideIfWithFallback(cond: bool): int {
                if (cond) {
                    for (i:int = 1; i <= 5; i = i + 1) {
                        if (i == 3) {
                            return i * 10;
                        }
                    }
                }
                return -1;
            }
            loopInsideIfWithFallback(true);
        "#;
        let program = ast::parse(source_code);
        let res = evaluate(&program);
        let res_ref = res.borrow();
        assert_eq!(Value::Integer(30), *res_ref.deref());
    }

    #[test]
    fn test_for_with_multiple_if_conditions() {
        let source_code = r#"
            function classifyNumbers(limit: int): int {
                even_sum: int = 0;
                odd_sum: int = 0;
                for (i:int = 1; i <= limit; i = i + 1) {
                    if (i % 2 == 0) {
                        even_sum = even_sum + i;
                    } else if (i % 3 == 0) {
                        odd_sum = odd_sum + i * 2;
                    } else {
                        odd_sum = odd_sum + i;
                    }
                }
                return even_sum * 100 + odd_sum;
            }
            classifyNumbers(6);
        "#;
        let program = ast::parse(source_code);
        let res = evaluate(&program);
        let res_ref = res.borrow();
        assert_eq!(Value::Integer(1212), *res_ref.deref()); // even_sum = 12, odd_sum = 6
    }
}
