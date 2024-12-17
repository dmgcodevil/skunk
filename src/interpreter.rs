use std::any::Any;
use std::collections::{HashMap, LinkedList};

use crate::ast;
use ast::Literal;
use ast::Node;
use ast::Operator;
use ast::Type;
use std::cell::RefCell;
use std::fmt;
use std::io::BufRead;
use std::mem;
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
        parameters: Vec<Parameter>,
        return_type: Type,
        body: Vec<Node>,
    },
    Return(Rc<RefCell<Value>>),
    Executed, // indicates that a branch has been executed
    Void,
    Undefined,
}

enum Scope {
    Module { name: String },
    Struct { name: String },
    FunctionScope { name: String },
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
                body,
            } => {
                let params_str = parameters
                    .into_iter()
                    .map(|p| format!("{}:{}", p.name, ast::type_to_string(&p.sk_type)))
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
                fields: _,
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
    functions: HashMap<String, FunctionDefinition>,
}

#[derive(Debug, Clone, PartialEq)]
struct Parameter {
    name: String,
    sk_type: Type,
}

#[derive(Clone, Debug, PartialEq)]
struct FunctionDefinition {
    name: String,
    parameters: Vec<Parameter>,
    return_type: Type,
    body: Vec<Node>,
    lambda: bool,
}

struct GlobalEnvironment {
    structs: HashMap<String, StructDefinition>,
    functions: HashMap<String, Rc<RefCell<FunctionDefinition>>>,
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
        self.functions
            .insert(name, Rc::new(RefCell::new(definition)));
    }

    fn get_function(&self, name: &str) -> Option<&Rc<RefCell<FunctionDefinition>>> {
        self.functions.get(name)
    }
    fn to_struct_def(&self, v: &Value) -> &StructDefinition {
        match v {
            StructInstance { name, .. } => self.structs.get(name).unwrap(),
            _ => panic!("expected struct instance. given: {:?}", v),
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
struct Environment {
    variables: HashMap<String, Rc<RefCell<Value>>>,
    overridable: bool,
}

trait ValueModifier {
    fn set(&mut self, value: Rc<RefCell<Value>>);
    fn get(&self) -> Rc<RefCell<Value>>;
}

struct StructInstanceModifier {
    instance: Rc<RefCell<Value>>,
    field: String,
}
impl ValueModifier for StructInstanceModifier {
    fn set(&mut self, value: Rc<RefCell<Value>>) {
        if let Value::StructInstance {
            name,
            ref mut fields,
        } = self.instance.borrow_mut().deref_mut()
        {
            fields.insert(self.field.clone(), value);
        } else {
            panic!("expected a struct instance");
        }
    }

    fn get(&self) -> Rc<RefCell<Value>> {
        if let Value::StructInstance {
            name: _,
            ref fields,
        } = self.instance.borrow().deref()
        {
            fields.get(&self.field).expect("Field not found").clone()
        } else {
            panic!("expected a struct instance");
        }
    }
}

struct ArrayModifier {
    array: Rc<RefCell<Value>>,
    coordinates: Vec<i64>,
}

impl ValueModifier for ArrayModifier {
    fn set(&mut self, value: Rc<RefCell<Value>>) {
        set_array_element(
            self.array.borrow_mut().deref_mut(),
            value,
            &self.coordinates,
        )
    }
    fn get(&self) -> Rc<RefCell<Value>> {
        get_array_element(self.array.borrow().deref(), &self.coordinates)
    }
}

struct ReadValueModifier {
    value: Rc<RefCell<Value>>,
}

impl ValueModifier for ReadValueModifier {
    fn set(&mut self, _value: Rc<RefCell<Value>>) {
        panic!("attempted to set a read value");
    }

    fn get(&self) -> Rc<RefCell<Value>> {
        Rc::clone(&self.value)
    }
}

struct StackVariableModifier {
    stack: Rc<RefCell<CallStack>>,
    name: String,
}

impl ValueModifier for StackVariableModifier {
    fn set(&mut self, value: Rc<RefCell<Value>>) {
        let mut stack = self.stack.borrow_mut();
        stack
            .deref_mut()
            .current_frame_mut()
            .locals
            .assign_variable(&self.name.clone(), Rc::clone(&value));
    }

    fn get(&self) -> Rc<RefCell<Value>> {
        let stack = self.stack.borrow();
        Rc::clone(
            &stack
                .deref()
                .current_frame()
                .locals
                .get_variable_value(&self.name.clone()),
        )
    }
}

fn get_array_element(value: &Value, coordinates: &Vec<i64>) -> Rc<RefCell<Value>> {
    if let Value::Array {
        ref arr,
        ref dimensions,
    } = value
    {
        let pos = to_1d_pos(coordinates, dimensions);
        Rc::clone(&arr[pos])
    } else {
        panic!("expected array value. given: {:?}", value)
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
    } else {
        panic!("expected array value. given: {:?}", value)
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
            overridable: false,
        }
    }
    fn overridable() -> Self {
        Environment {
            variables: HashMap::new(),
            overridable: true,
        }
    }

    fn declare_variable(&mut self, name: &str, value: Rc<RefCell<Value>>) {
        // todo assert that variable doesn't already exists
        self.variables.insert(name.to_string(), value);
    }

    fn assign_variable(&mut self, name: &str, value: Rc<RefCell<Value>>) {
        // todo assert that var exists
        self.variables.insert(name.to_string(), value);
    }

    fn get_variable_value(&self, name: &str) -> &Rc<RefCell<Value>> {
        if let Some(var) = self.variables.get(name) {
            var
        } else {
            panic!("variable '{}' not declared", name);
        }
    }
}

#[derive(Debug, Clone)]
pub struct CallFrame {
    name: String,
    locals: Environment,
}

impl CallFrame {
    fn to_overridable(&self, name: String) -> CallFrame {
        CallFrame {
            name,
            locals: Environment {
                variables: self.locals.variables.clone(),
                overridable: true,
            },
        }
    }
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
            .expect("call stack underflow: no active frames")
    }

    fn current_frame_mut(&mut self) -> &mut CallFrame {
        self.frames
            .last_mut()
            .expect("call stack underflow: no active frames")
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

fn assert_value_is_struct(v: &Value) {
    if !matches!(v, StructInstance { .. }) {
        panic!("expected struct instance, found {:?}", v);
    }
}

fn resolve_member_access(
    stack: &Rc<RefCell<CallStack>>,
    global_environment: &Rc<RefCell<GlobalEnvironment>>,
    current_value: Rc<RefCell<Value>>,
    node: &Node,
) -> Box<dyn ValueModifier> {
    if let Node::MemberAccess { member, .. } = node {
        let member_ref = member.as_ref();
        match member_ref {
            Node::Identifier(name) => {
                let current_value_ref = current_value.borrow();
                match current_value_ref.deref() {
                    StructInstance { .. } => {
                        drop(current_value_ref);
                        Box::new(StructInstanceModifier {
                            instance: current_value,
                            field: name.clone(),
                        })
                    }
                    Value::Array { arr, .. } => match name.as_str() {
                        "len" => Box::new(ReadValueModifier {
                            value: Rc::new(RefCell::new(Value::Integer(arr.len() as i64))),
                        }),
                        _ => unreachable!("{}", format!("unsupported array member: `{}`", name)),
                    },
                    _ => unreachable!("unsupported member access"),
                }
            }
            Node::FunctionCall { name, .. } => {
                let global_environment_ref = global_environment.borrow();
                let struct_def =
                    global_environment_ref.to_struct_def(current_value.borrow().deref());
                let fun_def = struct_def.functions.get(name).unwrap();
                assert_eq!(fun_def.parameters.first().unwrap().sk_type, Type::SkSelf);

                let res = evaluate_function(
                    stack,
                    &fun_def.parameters,
                    &fun_def.body,
                    member_ref,
                    global_environment,
                    || {
                        let mut frame = CallFrame {
                            name: format!("{}.{}", struct_def.name, name),
                            locals: Environment::new(),
                        };
                        frame
                            .locals
                            .assign_variable("self", Rc::clone(&current_value));
                        frame
                    },
                );
                mem::drop(global_environment_ref);
                Box::new(ReadValueModifier { value: res })
            }
            _ => panic!("expected member access, found {:?}", member_ref),
        }
    } else {
        panic!("expected member access, found {:?}", node);
    }
}

fn resolve_array_access(
    stack: &Rc<RefCell<CallStack>>,
    global_environment: &Rc<RefCell<GlobalEnvironment>>,
    current_value: Rc<RefCell<Value>>,
    node: &Node,
) -> Box<dyn ValueModifier> {
    if let Node::ArrayAccess { coordinates } = node {
        let _coordinates: Vec<i64> = coordinates
            .iter()
            .map(
                |c| match evaluate_node(c, stack, global_environment).borrow().deref() {
                    Value::Integer(dim) => *dim,
                    _ => panic!("expected integer index for array access"),
                },
            )
            .collect();
        Box::new(ArrayModifier {
            array: current_value,
            coordinates: _coordinates,
        })
    } else {
        panic!("expected array access, found {:?}", node)
    }
}

fn resolve_variable_access(stack: &Rc<RefCell<CallStack>>, node: &Node) -> Box<dyn ValueModifier> {
    if let Node::Identifier(name) = node {
        Box::new(StackVariableModifier {
            stack: Rc::clone(stack),
            name: name.clone(),
        })
    } else {
        panic!("expected identifier, found {:?}", node)
    }
}

fn resolve_access(
    stack: &Rc<RefCell<CallStack>>,
    global_environment: &Rc<RefCell<GlobalEnvironment>>,
    current_value: Rc<RefCell<Value>>,
    level: usize,
    access_nodes: &Vec<Node>,
) -> Box<dyn ValueModifier> {
    let mut current_modifier: Box<dyn ValueModifier> = Box::new(ReadValueModifier {
        value: Rc::clone(&current_value),
    });

    for access_node in access_nodes {
        current_modifier = match access_node {
            Node::MemberAccess { .. } => resolve_member_access(
                stack,
                global_environment,
                current_modifier.get(),
                access_node,
            ),
            Node::ArrayAccess { .. } => resolve_array_access(
                stack,
                global_environment,
                current_modifier.get(),
                access_node,
            ),
            Node::Identifier(..) => resolve_variable_access(stack, access_node),
            _ => panic!("unexpected access node: {:?}", access_node),
        };
    }

    current_modifier
}

fn evaluate_function<F>(
    stack: &Rc<RefCell<CallStack>>,
    parameters: &Vec<Parameter>,
    body: &Vec<Node>,
    call_node: &Node,
    global_environment: &Rc<RefCell<GlobalEnvironment>>,
    frame_creator: F,
) -> Rc<RefCell<Value>>
where
    F: Fn() -> CallFrame,
{
    if let Node::FunctionCall {
        name: _, arguments, ..
    } = call_node
    {
        let args_values: Vec<_> = arguments
            .into_iter()
            .map(|arg| evaluate_node(arg, stack, global_environment))
            .collect();
        let arguments: Vec<(&Parameter, Rc<RefCell<Value>>)> = parameters
            .iter()
            .filter(|p| p.sk_type != Type::SkSelf) // self is added to call stack implicitly
            .zip(args_values.into_iter())
            .collect();
        let mut frame = frame_creator();
        for arg in arguments {
            let first = arg.0.name.clone();
            let second = arg.1;
            frame.locals.variables.insert(first, Rc::clone(&second));
        }
        stack.borrow_mut().push(frame);

        let mut result = Rc::new(RefCell::new(Undefined));
        for n in body {
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
    } else {
        panic!("expected Node::FunctionCall")
    }
}

fn create_function_definition(n: &Node) -> FunctionDefinition {
    if let Node::FunctionDeclaration {
        name,
        parameters,
        return_type,
        body,
        lambda,
    } = n
    {
        FunctionDefinition {
            name: name.to_string(),
            parameters: parameters
                .iter()
                .map(|p| Parameter {
                    name: p.0.clone(),
                    sk_type: p.1.clone(),
                })
                .collect(),
            return_type: return_type.clone(),
            body: body.clone(), // todo avoid clone
            lambda: *lambda,
        }
    } else {
        panic!("expected Node::FunctionDeclaration")
    }
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
        Node::StructDeclaration {
            name,
            fields,
            functions,
        } => {
            let mut function_defs: HashMap<String, FunctionDefinition> = HashMap::new();
            for fun_node in functions {
                let fun_decl = create_function_definition(&fun_node);
                function_defs.insert(fun_decl.name.clone(), fun_decl);
            }

            let sd = StructDefinition {
                name: name.to_string(),
                fields: fields.clone(),
                functions: function_defs,
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
            lambda,
        } => {
            if !lambda {
                global_environment.borrow_mut().functions.insert(
                    name.clone(),
                    Rc::new(RefCell::new(create_function_definition(node))),
                );
            }
            Rc::new(RefCell::new(Value::Function {
                name: name.to_string(),
                parameters: parameters
                    .iter()
                    .map(|p| Parameter {
                        name: p.0.clone(),
                        sk_type: p.1.clone(),
                    })
                    .collect(),
                return_type: return_type.clone(),
                body: if *lambda { body.clone() } else { Vec::new() },
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
        Node::ArrayInit { elements } => {
            let mut dimensions: Vec<i64> = vec![];
            dimensions.push(elements.len() as i64);

            let mut node_stack: LinkedList<&Node> = LinkedList::new();
            if !elements.is_empty() {
                node_stack.push_back(&elements[0]);
            }
            while !node_stack.is_empty() {
                let n = node_stack.pop_back().unwrap();
                match n {
                    Node::ArrayInit { elements } => {
                        if !elements.is_empty() {
                            dimensions.push(elements.len() as i64);
                            node_stack.push_back(&elements[0])
                        }
                    }
                    _ => {}
                }
            }
            node_stack.clear();
            for e in elements {
                node_stack.push_back(e)
            }
            let mut values: Vec<Rc<RefCell<Value>>> = vec![];
            while !node_stack.is_empty() {
                let n = node_stack.pop_front().unwrap();
                match n {
                    Node::ArrayInit { elements } => {
                        for e in elements {
                            node_stack.push_back(e)
                        }
                    }
                    _ => values.push(Rc::clone(&evaluate_node(n, stack, global_environment))),
                }
            }

            Rc::new(RefCell::new(Value::Array {
                arr: values,
                dimensions,
            }))
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
        Node::Access { nodes } => resolve_access(
            stack,
            global_environment,
            Rc::new(RefCell::new(Undefined)),
            0,
            nodes,
        )
        .get()
        .clone(),
        Node::Assignment { var, value, .. } => match var.as_ref() {
            Node::Access { nodes } => {
                let mut modifier = resolve_access(
                    stack,
                    global_environment,
                    Rc::new(RefCell::new(Undefined)),
                    0,
                    nodes,
                );
                let new_value =
                    Rc::clone(&evaluate_node(value.as_ref(), stack, global_environment));
                modifier.set(new_value);
                modifier.get()
            }
            _ => panic!("expected access to assignment node"),
        },
        Node::Access { nodes } => resolve_access(
            stack,
            global_environment,
            Rc::new(RefCell::new(Undefined)),
            0,
            nodes,
        )
        .get(),
        Node::FunctionCall { name, .. } => {
            let value_opt = {
                let stack_ref = stack.borrow();
                stack_ref
                    .current_frame()
                    .locals
                    .variables
                    .get(name)
                    .cloned()
            };

            let res = if let Some(value) = value_opt {
                let value_ref = value.borrow();
                match value_ref.deref() {
                    Value::Function {
                        name,
                        parameters,
                        return_type,
                        body,
                    } => Some(evaluate_function(
                        stack,
                        &parameters,
                        &body,
                        node,
                        global_environment,
                        || {
                            stack
                                .borrow()
                                .current_frame()
                                .to_overridable(name.to_string())
                        },
                    )),
                    _ => None,
                }
            } else {
                None
            };

            if let Some(res) = res {
                res
            } else {
                let global_environment_ref = global_environment.borrow();
                let fun = global_environment_ref
                    .get_function(name)
                    .expect(format!("function `{}` doesn't exist", name).as_str())
                    .clone();
                let fun_ref = fun.borrow();
                mem::drop(global_environment_ref);
                evaluate_function(
                    stack,
                    &fun_ref.parameters,
                    &fun_ref.body,
                    node,
                    global_environment,
                    || CallFrame {
                        name: name.to_string(),
                        locals: Environment::new(),
                    },
                )
            }
        }
        Node::Block { statements } => {
            let mut frame = CallFrame {
                name: "".to_string(),
                locals: Environment::overridable(),
            };
            let mut stack_ref = stack.borrow_mut();
            for (n, v) in &stack_ref.deref().current_frame().locals.variables {
                frame.locals.variables.insert(n.to_string(), v.clone());
            }
            stack_ref.frames.push(frame);
            mem::drop(stack_ref);
            let mut res = Rc::new(RefCell::new(Undefined));
            for statement in statements {
                res = evaluate_node(statement, stack, global_environment);
                if let Value::Return(v) = res.borrow_mut().deref() {
                    break;
                }
            }
            stack_ref = stack.borrow_mut();
            stack_ref.frames.pop();
            mem::drop(stack_ref);
            Rc::clone(&res)
        }
        Node::StaticFunctionCall {
            _type,
            name: _,
            arguments,
            ..
        } => match _type {
            Type::Array {
                elem_type: _,
                dimensions,
            } => {
                let mut size: usize = 1;
                let mut int_dimensions: Vec<i64> = vec![];
                for dim_node in dimensions {
                    let dim_val = evaluate_node(dim_node, stack, global_environment);
                    let dim_val_ref = dim_val.borrow();
                    match dim_val_ref.deref() {
                        Value::Integer(i) => {
                            int_dimensions.push(*i);
                            size = size * (*i) as usize;
                        }
                        _ => panic!("expected integer in array size"),
                    };
                }
                let mut arr = Vec::with_capacity(size);
                for _ in 0..size {
                    arr.push(evaluate_node(&arguments[0], stack, global_environment));
                }
                Rc::new(RefCell::new(Value::Array {
                    arr,
                    dimensions: int_dimensions.clone(),
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
                _ => panic!("unsupported unary operator: {:?}", operator),
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
                (Value::String(s), Value::Integer(i)) => match operator {
                    Operator::Add => Rc::new(RefCell::new(Value::String(format!("{}{}", s, i)))),
                    _ => panic!("Unsupported operator for string concatenation"),
                },
                (Value::Integer(i), Value::String(s)) => match operator {
                    Operator::Add => Rc::new(RefCell::new(Value::String(format!("{}{}", i, s)))),
                    _ => panic!("Unsupported operator for string concatenation"),
                },
                (Value::Boolean(i), Value::String(s)) => match operator {
                    Operator::Add => Rc::new(RefCell::new(Value::String(format!("{}{}", i, s)))),
                    _ => panic!("Unsupported operator for string concatenation"),
                },
                (Value::String(s), Value::Boolean(i)) => match operator {
                    Operator::Add => Rc::new(RefCell::new(Value::String(format!("{}{}", s, i)))),
                    _ => panic!("Unsupported operator for string concatenation"),
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
            let res = Rc::clone(&v);
            mem::drop(stack_ref);
            res
        }
        Node::EOI => Rc::new(RefCell::new(Void)),
        Node::Return(n) => {
            let val = evaluate_node(n, stack, global_environment);
            Rc::new(RefCell::new(Return(Rc::clone(&val))))
        }
        _ => panic!("Unexpected node type: {:?}", node),
    }
}

use crate::ast::Node::FunctionDeclaration;
use crate::interpreter::Value::{Return, StructInstance, Undefined, Void};
use std::fmt::Octal;
use std::ops::{Deref, DerefMut};

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

    #[test]
    fn test_resolve_access_struct_field() {
        let source_code = r#"
        struct Point {
            x: int;
            y: int;
        }

        p: Point = Point { x: 10, y: 20 };
        p.x;
        "#;
        let program = ast::parse(source_code);
        let res = evaluate(&program);
        assert_eq!(Value::Integer(10), *res.borrow().deref());
    }

    #[test]
    fn test_resolve_access_nested_struct() {
        let source_code = r#"
        struct Inner {
            value: int;
        }

        struct Outer {
            inner: Inner;
        }

        o: Outer = Outer { inner: Inner { value: 42 } };
        o.inner.value;
        "#;
        let program = ast::parse(source_code);
        let res = evaluate(&program);
        assert_eq!(Value::Integer(42), *res.borrow().deref());
    }

    #[test]
    fn test_resolve_access_mixed() {
        let source_code = r#"
        struct Point {
            x: int;
        }

        arr: Point[2] = [Point { x: 10 }, Point { x: 20 }];
        arr[1].x;
        "#;
        let program = ast::parse(source_code);
        let res = evaluate(&program);
        assert_eq!(Value::Integer(20), *res.borrow().deref());
    }

    #[test]
    fn test_2d_array_init_inline() {
        let source_code = r#"
            arr: int[2][2] = [[1,2], [2,4]];
            arr;
            arr[1][1] = 5;
            arr[1][1];
        "#;
        let program = ast::parse(source_code);
        let res = evaluate(&program);
        assert_eq!(Value::Integer(5), *res.borrow().deref());
    }

    #[test]
    fn test_3d_array_init_inline_modify_flatten() {
        let source_code = r#"
            arr: int[3][2][1]= [[[1],[2]], [[3],[4]], [[5],[6]]];
            for(i:int = 0; i < 3; i = i + 1) {
                for(j:int = 0; j < 2; j = j + 1) {
                    for(k:int = 0; k < 1; k = k + 1) {
                        arr[i][j][k] = arr[i][j][k] + 10;
                    }
                }
            }
            n:int = 0;
            res: int[6] = int[6]::new(0);
            for(i:int = 0; i < 3; i = i + 1) {
                for(j:int = 0; j < 2; j = j + 1) {
                    for(k:int = 0; k < 1; k = k + 1) {
                        res[n] = arr[i][j][k];
                        n = n + 1;
                    }
                }
            }
            res;
        "#;
        let program = ast::parse(source_code);
        let res = evaluate(&program);
        let res_ref = res.borrow();
        match res_ref.deref() {
            Value::Array { arr, .. } => {
                let mut n = 1;
                for e in arr.iter() {
                    let e_ref = e.borrow();
                    if let Value::Integer(i) = e_ref.deref() {
                        assert_eq!(n + 10, *i);
                        n += 1;
                    } else {
                        panic!("expected integer");
                    }
                }
            }
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn test_instance_self() {
        let source_code = r#"
        struct Point {
            x: int;
            y: int;

            function set_x(self, x:int) {
                self.x = x;
            }

            function get_x(self){
                return self.x;
            }
        }

        p: Point = Point{x: 1, y: 2};
        p.set_x(3);
        p.get_x();
        "#;
        let program = ast::parse(source_code);
        println!("{:#?}", program);
        let res = evaluate(&program);
        assert_eq!(Value::Integer(3), *res.borrow().deref());
    }

    #[test]
    fn test_set_field_from_function() {
        let source_code = r#"
        struct Point {
            x: int;
            y: int;

            function set_x(self, x:int) {
                self.x = x;
            }

            function get_x(self){
                return self.x;
            }
        }

        struct Line {
            start: Point;
            end: Point;

            function get_start(self):Point {
                return self.start;
            }
        }

        line: Line = Line {start: Point { x: 0, y: 0 }, end: Point { x: 5, y: 5 }};
        line.get_start().set_x(1);
        line.get_start().get_x();
        "#;

        let program = ast::parse(source_code);
        let res = evaluate(&program);
        assert_eq!(Value::Integer(1), *res.borrow().deref());
    }

    // nested blocks tests
    #[test]
    fn test_nested_block_shadowing() {
        let source_code = r#"
            function f(): int {
                i: int = 1;
                {
                    i:int = 2;
                    print(i);
                }
                return i;
            }
            f();
        "#;
        let program = ast::parse(source_code);
        let res = evaluate(&program);
        assert_eq!(Value::Integer(1), *res.borrow().deref());
    }

    #[test]
    fn test_list() {
        let source_code = r#"
            struct List {
                size:int;
                capacity:int;

                function has_space(self):boolean {
                    return self.size > self.capacity;
                }

                function grow(self) {
                    if(self.size > self.capacity) {
                        print("increase capacity");
                        self.capacity = 4; //self.capacity * 2;
                    }
                }

            }
            list = List{size: 3, capacity: 2};
            //list.has_space();
            list.grow();
            print("capacity=" + list.capacity);
        "#;
        let program = ast::parse(source_code);
        let res = evaluate(&program);
        println!("{:#?}", res.borrow().deref());
    }

    #[test]
    fn lambda_recursive() {
        let source_code = r#"
        factorial: (int) -> int = function(n: int): int {
            if (n == 0) {
                return 1;
            } else {
                return n * factorial(n - 1);
            }
        }
        factorial(3);
        "#;
        let program = ast::parse(source_code);
        let res = evaluate(&program);
        assert_eq!(Value::Integer(6), *res.borrow().deref());
    }

    #[test]
    fn lambda_path_as_arg() {
        let source_code = r#"
            function f(g: () -> int):int {
                return g();
            }
            g: () -> int = function(): int {
                return 1;
            }
            f(g);
        "#;
        let program = ast::parse(source_code);
        let res = evaluate(&program);
        assert_eq!(Value::Integer(1), *res.borrow().deref());
    }

    #[test]
    fn anonymous_function() {
        let source_code = r#"
        function f(g: () -> int): int {
            return g();
        }

        f(function ():int {
            return 47;
        });
        "#;

        let program = ast::parse(source_code);
        let res = evaluate(&program);
        assert_eq!(Value::Integer(47), *res.borrow().deref());
    }

    #[test]
    fn anonymous_function_with_params() {
        let source_code = r#"
        function f(g: (int) -> int): int {
            return g(47);
        }

        f(function (i:int):int {
            return i;
        });
        "#;

        let program = ast::parse(source_code);
        let res = evaluate(&program);
        assert_eq!(Value::Integer(47), *res.borrow().deref());
    }

    #[test]
    fn test_function_return_function() {
        let source_code = r#"
        function f(): (int) -> int {
            return function (a:int): int {
                return a;
            }
        }

        g: (int) -> int = f();
        g(47);
        "#;
        let program = ast::parse(source_code);
        let res = evaluate(&program);
        assert_eq!(Value::Integer(47), *res.borrow().deref());
    }

    #[test]
    fn test_closure() {
        let source_code = r#"
        function f(): () -> int {
            counter: int = 0;
            return function (): int {
                counter = counter + 1;
                return counter;
            }
        }

        g: () -> int = f();
        g();
        "#;
        let program = ast::parse(source_code);
        let res = evaluate(&program);
        assert_eq!(Value::Integer(1), *res.borrow().deref());
    }
}
