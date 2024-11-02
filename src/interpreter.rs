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
            Value::Void => write!(f, ""),
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

fn get_element_from_array(value: &Value, coordinates: Vec<i64>) -> Rc<RefCell<Value>> {
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

fn set_array_element(value: &mut Value, new_value: &Rc<RefCell<Value>>, coordinates: Vec<i64>) {
    if let Value::Array {
        ref mut arr,
        ref dimensions,
    } = value
    {
        let pos = to_1d_pos(coordinates, dimensions);
        arr[pos] = Rc::clone(new_value);
    }
}

fn to_1d_pos(coordinates: Vec<i64>, dimensions: &Vec<i64>) -> usize {
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
        self.variables.insert(name.to_string(), value);
    }

    fn assign_variable(&mut self, name: &str, value: Value) {
        let parts: Vec<&str> = name.split('.').collect();
        let size = parts.len();

        if size == 1 {
            // Base case: if no nested fields, assign directly
            if self.variables.contains_key(name) {
                self.variables.insert(name.to_string(), Rc::new(RefCell::new(value)));
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
        Value::Undefined => {
            Rc::new(RefCell::new(Value::Undefined))
        }
        Value::StructInstance { name, fields} =>
            Rc::clone(fields.get(name).unwrap()),
        _ => panic!("expected current_value={:?}", current_value),
    }
}

fn get_value_nested(stack: &Rc<RefCell<CallStack>>,
                    global_environment: &Rc<RefCell<GlobalEnvironment>>,
                    current_value: Rc<RefCell<Value>>,
                    access_node: &Node
) -> Rc<RefCell<Value>> {
    let stack_ref = stack.borrow();
    let frame = stack_ref.current_frame();
    match access_node {
        Node::ArrayAccess {name, coordinates} => {
            match current_value.borrow().deref()  {
                Value::Undefined => Rc::clone(&frame.locals.get_variable_value(name)),
                Value::StructInstance { name, fields} =>
                    Rc::clone(fields.get(name).unwrap()),
                _ => panic!("expected current_value={:?}", current_value.borrow().deref()),
            }
        }
        Node::Identifier(name) => {
            match current_value.borrow().deref() {
                Value::Undefined => Rc::clone(&frame.locals.get_variable_value(name)),
                Value::StructInstance { name, fields} =>
                    Rc::clone(fields.get(name).unwrap()),
                _ => panic!("expected current_value={:?}", current_value.borrow().deref()),
            }
        }
        Node::FunctionCall { name, arguments } => {
            match current_value.borrow().deref()  {
                Value::Undefined => evaluate_node(access_node, stack, global_environment),
                _ => panic!("function cannot be called on {:?}", current_value.borrow().deref()),
            }
        }
        _ => panic!("unsupported access node={:?}", access_node),
    }
}
fn evaluate_assigment(stack: &Rc<RefCell<CallStack>>,
                      global_environment: &Rc<RefCell<GlobalEnvironment>>,
                      current_value: Rc<RefCell<Value>>,
                      i: usize, access_nodes: &Vec<Node>,
                      new_value: Rc<RefCell<Value>>) {
    let frame = stack.borrow().current_frame();
    let access = &access_nodes[i];
    if i == access_nodes.len() - 1 {
        match access {
            Node::ArrayAccess { name, coordinates } => {
                if let Value::Array { arr, .. } = current_value.borrow().deref() {
                    let _coordinates: Vec<i64> = coordinates.iter().map(|n| {
                        match evaluate_node(n, stack, global_environment).borrow().deref() {
                            Value::Integer(i) => *i,
                            _ => panic!("array access expected int")
                        }
                    }).collect();
                    set_array_element(current_value.borrow_mut().deref_mut(),
                                      &new_value, _coordinates);
                } else {
                    panic!("expected array value. actual={}", current_value.borrow().deref())
                }
            }
            Node::Identifier(name) => {
                if let Value::StructInstance { name, ref mut fields } =
                    current_value.borrow_mut().deref_mut() {
                    fields.insert(name.to_string(), new_value.clone());
                } else {
                    panic!("expected struct instance, found: {:?}", current_value.borrow().deref());
                }
            }
            _ => panic!("expected or unsupported access node. actual={:?}", access)
        }
    } else {

    }

}

pub fn evaluate_node(
    node: &Node,
    stack: &Rc<RefCell<CallStack>>,
    global_environment: &Rc<RefCell<GlobalEnvironment>>,
) -> Rc<RefCell<Value>> {
    match node {
        Node::Program { statements } => {
            let mut last  = Rc::new(RefCell::new(Void));
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
        Node::Literal(Literal::StringLiteral(x)) => Rc::new(RefCell::new(Value::String(x.to_string()))),
        Node::Literal(Literal::Boolean(x)) => Rc::new(RefCell::new(Value::Boolean(*x))),
        Node::VariableDeclaration { name, value, .. } => {
            let value = evaluate_node(value, stack, global_environment);
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
        Node::Access { nodes } => {
            /*
struct C {
    d:int;
}
struct B {
    c:C;
}
struct A {
    b: B[];
}
a: A[1] = A[1]::new();
a[0].b[0].c.d = 1;


we need to interpret Assigment node:

Assignment {
    var: Access {
        nodes: [
            ArrayAccess { name: "a", coordinates: [Literal(Integer(0))] },
            ArrayAccess { name: "b", coordinates: [Literal(Integer(0))] },
            Identifier("c"), Identifier("d")] },
    value: Literal(Integer(1))
}

1. evaluate `a[0]` => gives struct A instance
2. evaluate `b[0]` => gives struct B instance
3. evaluate `c` => gives struct C instance
4. `d` is last

            */
            Rc::new(RefCell::new(Undefined))

        }
        Node::Assignment { var, value } => {
            Rc::new(RefCell::new(Undefined))
        }
        Node::FunctionCall { name, arguments } => {
            let args_values: Vec<_> = arguments
                .into_iter()
                .map(|arg| evaluate_node(arg, stack, global_environment).clone().borrow().deref().clone())
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
                frame.locals.variables.insert(first, Rc::new(RefCell::new(second)));
            }
            stack.borrow_mut().push(frame);

            let mut result = Rc::new(RefCell::new(Void));
            for n in fun.body {
                if let Value::Return(v) = evaluate_node(&n, stack, global_environment).borrow().deref() {
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
            if let Value::Boolean(ok) = *evaluate_node(condition.as_ref(), stack, global_environment).borrow().deref()
            {
                if ok {
                    for n in body {
                        let n_val = evaluate_node(n, stack, global_environment);
                        let val = n_val.borrow().deref().clone();
                        if let Value::Return(v) = val {
                            return Rc::new(RefCell::new(Value::Return(v.clone()) ))
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
                    return  Rc::clone(&val_ref);
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

            Rc::new(RefCell::new(Value::Void))
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
                        Value::Boolean(v) =>v,
                        _ => panic!("for condition in should be a boolean expression"),
                    }
                })
                .unwrap_or(true)
            {
                for n in body {
                    let val_ref = evaluate_node(n, stack, global_environment);
                    let val_borrowed = val_ref.borrow();
                    if matches!(val_borrowed.deref(), Value::Return(_)) {
                        return Rc::clone(&val_ref);
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
            let left_val_ref = evaluate_node(left, stack, global_environment);
            let right_val_ref = evaluate_node(right, stack, global_environment);
            let left_val_borrowed = left_val_ref.borrow();
            let right_val_borrowed = right_val_ref.borrow();
            match (left_val_borrowed.deref(), right_val_borrowed.deref()) {
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
        Node::ArrayAccess { name, coordinates } => {
            let mut pos: usize = 0;
            let mut multiplier: i64 = 1;
            let stack_ref = stack.borrow();
            let current = stack_ref.current_frame();
            let array_value_ref = current.locals.get_variable_value(name);

            if let Value::Array {
                ref arr,
                ref dimensions,
            } = array_value_ref.borrow().deref()
            {
                for i in (0..coordinates.len()).rev() {
                    if let Value::Integer(coord) = *evaluate_node(
                        &coordinates[i].clone(),
                        &Rc::clone(stack),
                        global_environment,
                    ).borrow().deref() {
                        if coord >= dimensions[i] {
                            panic!(
                                "array out of bound. {} >= {}. dim={}",
                                coord, dimensions[i], i
                            );
                        }
                        pos += (coord * multiplier) as usize;
                        multiplier *= dimensions[i];
                    } else {
                        panic!("expected coordinate type to be an integer");
                    }
                }
                return arr[pos].clone();
            }
            panic!(
                "expected array, actual={:?}",
                "current.locals.get_variable_value(name)"
            )
        }
        Node::EOI => Rc::new(RefCell::new(Void)),
        Node::Return(n) =>
            Rc::new(RefCell::new(
            Value::Return(Box::new(evaluate_node(n, stack, global_environment).borrow().deref().clone())))),
        _ => panic!("Unexpected node type: {:?}", node),
    }
}

use crate::interpreter::Value::{Undefined, Void};
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
    fn test_array_access() {
        let source_code = r#"
         arr: int[5] = int[5]::new(1);
         arr[0];
        "#;
        let program = ast::parse(source_code);
        println!("{:?}", program);
    }

    //#[test]
    fn test_array_set() {
        let source_code = r#"
            arr: int[1] = int[1]::new(0);
            arr[0] = 1;
            return arr[0];
        "#;
        let program = ast::parse(source_code);
        println!("{:?}", evaluate(&program));
    }

    //#[test]
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
