use std::any::Any;
use std::collections::{HashMap, LinkedList};
use sysinfo::{Pid, System};

use crate::ast;
use ast::Literal;
use ast::Node;
use ast::Operator;
use ast::Type;
use std::cell::RefCell;
use std::f32::consts::E;
use std::fmt;
use std::io::BufRead;
use std::mem;
use std::ops::{Deref, DerefMut};
use std::rc::Rc;

#[derive(Debug, PartialEq, Clone)]
pub enum Value {
    Integer(i64),
    String(String),
    Boolean(bool),
    Variable(String),
    Array {
        arr: Vec<ValueRef>,
        dimensions: Vec<i64>,
    },
    StructInstance {
        name: String,
        fields: HashMap<String, ValueRef>,
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
    Closure {
        function: FunctionDefinition,
        env: Rc<RefCell<Environment>>,
    },
    // Return(Box<ValueRef>),
    Executed, // indicates that a branch has been executed
    Void,
    Undefined,
}

// enum Scope {
//     Module { name: String },
//     Struct { name: String },
//     FunctionScope { name: String },
// }

#[derive(Debug, PartialEq, Clone)]
enum ValueRef {
    Stack {
        value: Value,
        returned: bool,
    }, // Represents stack-allocated values.
    Heap {
        value: Rc<RefCell<Value>>,
        returned: bool,
    }, // Represents heap-allocated values.
}

impl ValueRef {
    fn stack(value: Value) -> Self {
        ValueRef::Stack {
            value,
            returned: false,
        }
    }

    fn heap(value: Rc<RefCell<Value>>) -> Self {
        ValueRef::Heap {
            value,
            returned: false,
        }
    }
    fn returned(&self) -> bool {
        match self {
            ValueRef::Stack { returned, .. } => *returned,
            ValueRef::Heap { returned, .. } => *returned,
        }
    }

    fn to_returned(self) -> ValueRef {
        match self {
            ValueRef::Stack { value, .. } => ValueRef::Stack {
                value,
                returned: true,
            },
            ValueRef::Heap { value, .. } => ValueRef::Heap {
                value,
                returned: true,
            },
        }
    }

    fn map_value<F>(&self, f: F) -> ValueRef
    where
        F: Fn(&Value) -> Value,
    {
        match self {
            ValueRef::Stack { value, returned } => ValueRef::Stack {
                value: f(value),
                returned: *returned,
            },
            ValueRef::Heap { value, returned } => {
                let value_ref = value.borrow();
                let res = f(value_ref.deref());
                drop(value_ref);
                ValueRef::Heap {
                    value: Rc::new(RefCell::new(res)),
                    returned: *returned,
                }
            }
        }
    }

    fn map<T, F>(&self, f: F) -> Option<T>
    where
        F: Fn(&Value) -> Option<T>,
    {
        match self {
            ValueRef::Stack { value, .. } => f(value),
            ValueRef::Heap { value, .. } => {
                let value_ref = value.borrow();
                let res = f(value_ref.deref());
                drop(value_ref);
                res
            }
        }
    }

    fn unwrap<F>(&self, f: F)
    where
        F: Fn(&Value),
    {
        self.is_match(|a| {
            f(a);
            true
        });
    }

    fn is_match<F>(&self, f: F) -> bool
    where
        F: Fn(&Value) -> bool,
    {
        match self {
            ValueRef::Stack { value, .. } => f(value),
            ValueRef::Heap { value, .. } => {
                let value_ref = value.borrow();
                let res = f(value_ref.deref());
                drop(value_ref);
                res
            }
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
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
            Value::Void => write!(f, ""),
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
    fn to_struct_def(&self, value_ref: &ValueRef) -> &StructDefinition {
        value_ref
            .map(|v| match v {
                Value::StructInstance { name, .. } => Some(self.structs.get(name).unwrap()),
                _ => panic!("expected struct instance. given: {:?}", v),
            })
            .unwrap()
    }
}

#[derive(Debug, PartialEq, Clone)]
struct Environment {
    id: String,
    parent: Option<Rc<RefCell<Environment>>>,
    variables: HashMap<String, ValueRef>,
}

impl Environment {
    fn new() -> Self {
        Environment {
            id: "".to_string(), //uuid::Uuid::new_v4().to_string(), // uncomment for debugging
            parent: None,
            variables: HashMap::new(),
        }
    }

    fn has_variable(&self, name: &str) -> bool {
        //println!("Environment::has_variable={}", name);
        if self.variables.contains_key(name) {
            return true;
        }
        if let Some(parent) = &self.parent {
            let parent_ref = parent.borrow();
            let result = parent_ref.has_variable(name);
            return result;
        }
        false
    }

    fn get_var(&self, name: &str) -> Option<ValueRef> {
        //println!("Environment::get_var={}", name);
        if self.variables.contains_key(name) {
            self.variables.get(name).cloned()
        } else if let Some(parent) = &self.parent {
            let mut parent_ref = parent.borrow();
            parent_ref.get_var(name)
        } else {
            None
        }
    }

    fn assign_variable(&mut self, name: &str, value: ValueRef) {
        //println!("Environment::assign_variable={}", name);
        if self.variables.contains_key(name) {
            self.variables.insert(name.to_string(), value);
        } else if let Some(parent) = self.parent.clone() {
            let mut parent_ref = parent.borrow_mut();
            parent_ref.assign_variable(name, value);
        }
    }

    fn declare_variable(&mut self, name: &str, value: ValueRef) {
        //println!("Environment::declare_variable={}", name);
        self.variables.insert(name.to_string(), value);
    }
}

trait ValueModifier {
    fn set(&mut self, value: ValueRef);
    fn get(&self) -> ValueRef;
}

struct StructInstanceModifier {
    instance: ValueRef,
    field: String,
}
impl ValueModifier for StructInstanceModifier {
    fn set(&mut self, new_value: ValueRef) {
        match &self.instance {
            ValueRef::Stack { value, .. } => {
                panic!("struct on stack")
                // if let Value::StructInstance {
                //     name,
                //     ref mut fields,
                // } = value
                // {
                //     fields.insert(self.field.clone(), new_value);
                // } else {
                //     panic!("expected a struct instance");
                // }
            }
            ValueRef::Heap { value, .. } => {
                let mut value_ref = value.borrow_mut();
                if let Value::StructInstance {
                    name,
                    ref mut fields,
                } = value_ref.deref_mut()
                {
                    fields.insert(self.field.clone(), new_value);
                } else {
                    panic!("expected a struct instance");
                }
            }
        }
    }

    fn get(&self) -> ValueRef {
        match &self.instance {
            ValueRef::Stack { .. } => {
                panic!("struct on stack")
            }
            ValueRef::Heap { value, .. } => {
                let mut value_ref = value.borrow_mut();
                if let Value::StructInstance {
                    name,
                    ref mut fields,
                } = value_ref.deref_mut()
                {
                    fields.get(&self.field).expect("Field not found").clone()
                } else {
                    panic!("expected a struct instance");
                }
            }
        }
    }
}

struct ArrayModifier {
    array: ValueRef,
    coordinates: Vec<i64>,
}

impl ValueModifier for ArrayModifier {
    fn set(&mut self, new_value: ValueRef) {
        match &self.array {
            ValueRef::Stack { .. } => {
                panic!("cannot mutate array allocated on stack")
            }
            ValueRef::Heap { value, .. } => {
                let mut value_ref = value.borrow_mut();
                set_array_element(value_ref.deref_mut(), new_value, &self.coordinates)
            }
        }
    }

    fn get(&self) -> ValueRef {
        match &self.array {
            ValueRef::Stack { ref value, .. } => {
                get_array_element(value, &self.coordinates).clone()
            }
            ValueRef::Heap { ref value, .. } => {
                let value_ref = value.borrow();
                get_array_element(value_ref.deref(), &self.coordinates).clone()
            }
        }
    }
}

fn get_array_element(value: &Value, coordinates: &Vec<i64>) -> ValueRef {
    if let Value::Array {
        ref arr,
        ref dimensions,
    } = value
    {
        let pos = to_1d_pos(coordinates, dimensions);
        arr[pos].clone()
    } else {
        panic!("expected array value. given: {:?}", value)
    }
}

fn set_array_element(value: &mut Value, new_value: ValueRef, coordinates: &Vec<i64>) {
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

struct ReadValueModifier {
    value: ValueRef,
}

impl ValueModifier for ReadValueModifier {
    fn set(&mut self, _value: ValueRef) {
        panic!("attempted to set a read value");
    }

    fn get(&self) -> ValueRef {
        self.value.clone()
    }
}

struct StackVariableModifier<'a> {
    stack: &'a mut CallStack,
    name: String,
}

impl ValueModifier for StackVariableModifier<'_> {
    fn set(&mut self, value: ValueRef) {
        self.stack.assign_variable(&self.name, value);
    }

    fn get(&self) -> ValueRef {
        self.stack
            .get_variable(&self.name)
            .expect(format!("Variable '{}' not found", self.name).as_str())
            .clone()
    }
}

struct Profiling {}

impl Profiling {
    fn sk_debug_stack(stack: &CallStack) {
        println!("--- Debugging Call Stack ---");
        for (i, frame) in stack.frames.iter().enumerate() {
            println!("Frame {}: {}", i, frame.name);

            let mut current_env = Some(frame.env.clone());

            while let Some(env_rc) = current_env {
                let env = env_rc.borrow();

                for (var_name, var_value) in &env.variables {
                    // let var_value_ref = var_value.borrow();
                    println!("  Environment id: '{}'", &env.id);
                    println!("      Variable '{}' -> {:?}", var_name, var_value);
                }
                current_env = env.parent.clone();
            }

            println!("----------------------------");
        }
    }

    fn sk_debug_mem_sysinfo() {
        let mut system = System::new_all();
        system.refresh_all();
        let current_pid = Pid::from_u32(std::process::id());
        if let Some(process) = system.process(current_pid) {
            println!("Current Process Information:");
            println!("  Name: {:?}", process.name());
            println!("  Executable Path: {:?}", process.exe());
            println!("  Current Working Directory: {:?}", process.cwd());
            println!("  Memory Usage: {:?} KB", process.memory());
            println!("  Virtual Memory Usage: {:?} KB", process.virtual_memory());
            println!("  CPU Usage: {:.2}%", process.cpu_usage());
            println!("  Status: {:?}", process.status());
        } else {
            println!("Failed to retrieve information about the current process.");
        }
    }

    fn estimate_value_size(value: &Value) -> usize {
        match value {
            Value::Integer(_) => std::mem::size_of::<i64>(),
            Value::String(s) => std::mem::size_of::<String>() + s.capacity(),
            Value::Array { arr, .. } => {
                arr.iter()
                    .map(|elem| Profiling::estimate_value_ref_size(elem))
                    .sum::<usize>()
                    + std::mem::size_of::<Vec<Rc<RefCell<Value>>>>()
            }
            Value::StructInstance { fields, .. } => {
                fields
                    .iter()
                    .map(|(_, v)| Profiling::estimate_value_ref_size(v))
                    .sum::<usize>()
                    + std::mem::size_of::<HashMap<String, Rc<RefCell<Value>>>>()
            }
            Value::Function {
                name,
                parameters,
                body,
                ..
            } => {
                std::mem::size_of::<String>()
                    + name.capacity()
                    + parameters.len() * std::mem::size_of::<Parameter>()
                    + body.len() * std::mem::size_of::<Node>()
            }
            Value::Closure { function, env } => {
                std::mem::size_of_val(&function) + std::mem::size_of_val(env)
            }
            _ => 0, // Handle other cases as needed
        }
    }

    fn estimate_value_ref_size(value_ref: &ValueRef) -> usize {
        match value_ref {
            ValueRef::Stack { value, .. } => Self::estimate_value_size(value),
            ValueRef::Heap { value, .. } => {
                let value = value.borrow();
                Self::estimate_value_size(value.deref())
            }
        }
    }

    fn sk_debug_mem_programmatic(stack: &CallStack) {
        println!("Memory Usage (programmatic):");
        for frame in &stack.frames {
            println!("  Frame: {}", frame.name);
            let frame_mem: usize = frame
                .env
                .borrow()
                .variables
                .values()
                .map(|val| Profiling::estimate_value_ref_size(val))
                .sum();
            println!("    Memory Used: {} bytes", frame_mem);
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CallFrame {
    name: String,
    env: Rc<RefCell<Environment>>,
}

impl CallFrame {
    fn new(name: String) -> Self {
        CallFrame {
            name: name.to_string(),
            env: Rc::new(RefCell::new(Environment::new())),
        }
    }

    fn set_parent(&mut self, parent: Rc<RefCell<Environment>>) {
        // println!("CallFrame-{}:: set_parent", self.name);
        let mut env = self.env.borrow_mut();
        if let Some(old_parent) = &env.parent {
            let old_parent = old_parent.borrow();
            assert_ne!(*old_parent.deref(), *parent.borrow().deref());
        }
        env.parent = Some(parent);
        drop(env)
    }

    fn has_variable(&self, name: &str) -> bool {
        //println!("CallFrame-{}::has_variable {}", self.name, name);
        let env_ref = self.env.borrow();
        env_ref.has_variable(name)
    }

    fn assign_variable(&mut self, name: &str, value: ValueRef) {
        //println!("CallFrame-{}::has_variable {}", self.name, name);
        let mut env_ref = self.env.borrow_mut();
        env_ref.assign_variable(name, value);
        drop(env_ref);
    }

    fn declare_variable(&mut self, name: &str, value: ValueRef) {
        // println!("CallFrame-{}::declare_variable {}", self.name, name);
        let mut env_ref = self.env.borrow_mut();
        env_ref.declare_variable(name, value);
        drop(env_ref);
    }

    fn get_variable(&self, name: &str) -> Option<ValueRef> {
        //println!("CallFrame-{}::get_variable {}", self.name, name);
        let env_ref = self.env.borrow();
        let res = env_ref.get_var(name);
        drop(env_ref);
        res
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

    /// create frame and sets its env.parent = current_frame.env
    fn create_frame(&self, name: String) -> CallFrame {
        let mut frame = CallFrame {
            name: name.to_string(),
            env: Rc::new(RefCell::new(Environment::new())),
        };
        if let Some(curr) = self.frames.last() {
            frame.set_parent(curr.env.clone());
        }
        frame
    }

    fn create_frame_push(&mut self, name: String) {
        let frame = self.create_frame(name);
        self.frames.push(frame);
    }

    fn get_closure(&self, name: &str) -> Option<ValueRef> {
        for frame in self.frames.iter().rev() {
            if frame.has_variable(name) {
                let val = frame.get_variable(name).unwrap();
                if val.is_match(|v| match v {
                    Value::Closure { .. } => true,
                    _ => false,
                }) {
                    return frame.get_variable(name);
                }
            }
        }
        None
    }

    fn get_variable(&self, name: &str) -> Option<ValueRef> {
        for frame in self.frames.iter().rev() {
            if frame.has_variable(name) {
                return frame.get_variable(name);
            }
        }
        None
    }

    fn assign_variable(&mut self, name: &str, value: ValueRef) -> bool {
        for frame in self.frames.iter_mut().rev() {
            if frame.has_variable(name) {
                frame.assign_variable(name, value);
                return true;
            }
        }
        unreachable!("var {} doesn't exist", name);
    }

    fn declare_variable(&mut self, name: &str, value: ValueRef) {
        self.current_frame_mut().declare_variable(name, value);
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

pub fn evaluate(node: &Node) -> ValueRef {
    let mut stack = CallStack::new();
    let ge = Rc::new(RefCell::new(GlobalEnvironment::new()));
    stack.push(CallFrame {
        name: "main".to_string(),
        env: Rc::new(RefCell::new(Environment::new())),
    });
    evaluate_node(node, &mut stack, &ge)
}

fn assert_value_is_struct(v: &Value) {
    if !matches!(v, Value::StructInstance { .. }) {
        panic!("expected struct instance, found {:?}", v);
    }
}

fn resolve_member_access<'a>(
    stack: &'a mut CallStack,
    global_environment: &Rc<RefCell<GlobalEnvironment>>,
    current_value: ValueRef,
    node: &Node,
) -> Box<dyn ValueModifier + 'a> {
    if let Node::MemberAccess { member, .. } = node {
        let member_ref = member.as_ref();
        match member_ref {
            Node::Identifier(name) => {
                let struct_match = current_value.map(|v| match v {
                    Value::StructInstance { .. } => Some(Box::new(StructInstanceModifier {
                        instance: current_value.clone(),
                        field: name.clone(),
                    })),
                    _ => None,
                });
                if let Some(m) = struct_match {
                    return m;
                }

                let array_match = current_value.map(|v| match v {
                    Value::Array { arr, .. } => Some(match name.as_str() {
                        "len" => Box::new(ReadValueModifier {
                            value: ValueRef::stack(Value::Integer(arr.len() as i64)),
                        }),
                        _ => unreachable!("{}", format!("unsupported array member: `{}`", name)),
                    }),
                    _ => None,
                });

                if let Some(m) = array_match {
                    return m;
                }

                unreachable!("{}", format!("unsupported array member: `{}`", name));
            }
            Node::FunctionCall {
                name, arguments, ..
            } => {
                let global_environment_ref = global_environment.borrow();
                let struct_def = global_environment_ref.to_struct_def(&current_value);
                let fun_def = struct_def.functions.get(name).unwrap();
                assert_eq!(fun_def.parameters.first().unwrap().sk_type, Type::SkSelf);

                stack.create_frame_push(format!("{}.{}", struct_def.name, name));
                stack.declare_variable("self", current_value.clone());

                let res =
                    evaluate_function_call(member_ref, stack, global_environment, Some(fun_def));
                stack.pop();
                drop(global_environment_ref);
                Box::new(ReadValueModifier { value: res })
            }
            _ => panic!("expected member access, found {:?}", member_ref),
        }
    } else {
        panic!("expected member access, found {:?}", node);
    }
}

fn resolve_array_access(
    stack: &mut CallStack,
    global_environment: &Rc<RefCell<GlobalEnvironment>>,
    current_value: ValueRef,
    node: &Node,
) -> Box<dyn ValueModifier> {
    if let Node::ArrayAccess { coordinates } = node {
        let _coordinates: Vec<i64> = coordinates
            .iter()
            .map(|c| {
                evaluate_node(c, stack, global_environment)
                    .map(|v| match v {
                        Value::Integer(dim) => Some(*dim),
                        _ => panic!("expected integer index for array access"),
                    })
                    .unwrap()
            })
            .collect();
        Box::new(ArrayModifier {
            array: current_value,
            coordinates: _coordinates,
        })
    } else {
        panic!("expected array access, found {:?}", node)
    }
}

fn resolve_variable_access<'a>(
    stack: &'a mut CallStack,
    node: &Node,
) -> Box<dyn ValueModifier + 'a> {
    if let Node::Identifier(name) = node {
        Box::new(StackVariableModifier {
            stack,
            name: name.clone(),
        })
    } else {
        panic!("expected identifier, found {:?}", node)
    }
}

fn resolve_access<'a>(
    stack: &'a mut CallStack,
    global_environment: &Rc<RefCell<GlobalEnvironment>>,
    current_value: ValueRef,
    level: usize,
    access_nodes: &Vec<Node>,
) -> Box<dyn ValueModifier + 'a> {
    let mut current_modifier: Box<dyn ValueModifier + 'a> = Box::new(ReadValueModifier {
        value: current_value,
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

fn evaluate_function(
    stack: &mut CallStack,
    arguments: &Vec<Node>,
    parameters: &Vec<Parameter>,
    body: &Vec<Node>,
    global_environment: &Rc<RefCell<GlobalEnvironment>>,
) -> ValueRef {
    let args_values: Vec<_> = arguments
        .into_iter()
        .map(|arg| evaluate_node(arg, stack, global_environment))
        .collect();
    let arguments: Vec<(&Parameter, ValueRef)> = parameters
        .iter()
        .filter(|p| p.sk_type != Type::SkSelf) // self is added to call stack implicitly
        .zip(args_values.into_iter())
        .collect();
    for arg in arguments {
        let first = arg.0.name.clone();
        let second = arg.1;
        stack.declare_variable(first.as_str(), second)
    }
    let mut result = ValueRef::stack(Value::Undefined);
    for n in body {
        let v = evaluate_node(&n, stack, global_environment);
        if v.returned() {
            result = v;
            break;
        }
    }
    result
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

fn evaluate_closure(
    value: &ValueRef,
    arguments: &Vec<Vec<Node>>,
    stack: &mut CallStack,
    global_environment: &Rc<RefCell<GlobalEnvironment>>,
) -> ValueRef {
    let mut curr = value;
    let mut result = ValueRef::stack(Value::Undefined);
    for arg in arguments {
        match curr {
            ValueRef::Stack { value, .. } => match value {
                Value::Closure { function, env } => {
                    let mut frame = CallFrame::new(function.name.clone());
                    frame.set_parent(env.clone());
                    stack.push(frame);
                    let r = evaluate_function(
                        stack,
                        arg,
                        &function.parameters,
                        &function.body,
                        global_environment,
                    );
                    stack.pop();
                    result = r;
                    curr = &result;
                }
                _ => panic!("expected closure"),
            },
            ValueRef::Heap { value, .. } => {
                let value_ref = value.borrow();
                match value_ref.deref() {
                    Value::Closure { function, env } => {
                        let mut frame = CallFrame::new(function.name.clone());
                        frame.set_parent(env.clone());
                        stack.push(frame);
                        let r = evaluate_function(
                            stack,
                            arg,
                            &function.parameters,
                            &function.body,
                            global_environment,
                        );
                        stack.pop();
                        result = r;
                        curr = &result;
                    }
                    _ => panic!("expected closure"),
                }
            }
        }
    }
    result
}

fn evaluate_function_call(
    node: &Node,
    stack: &mut CallStack,
    global_environment: &Rc<RefCell<GlobalEnvironment>>,
    fun: Option<&FunctionDefinition>,
) -> ValueRef {
    match node {
        Node::FunctionCall {
            name, arguments, ..
        } => {
            // if name == "sk_debug_mem" {
            //     Profiling::sk_debug_mem_sysinfo();
            //     Profiling::sk_debug_mem_programmatic(stack.borrow().deref());
            //     return Rc::new(RefCell::new(Value::Undefined));
            // }
            //
            // if name == "sk_debug_stack" {
            //     Profiling::sk_debug_stack(stack);
            //     return Rc::new(RefCell::new(Value::Undefined));
            // }

            let value_opt = { stack.get_closure(name) };

            let res = if let Some(value) = value_opt {
                Some(evaluate_closure(
                    &value,
                    arguments,
                    stack,
                    global_environment,
                ))
            } else {
                None
            };

            if let Some(res) = res {
                res
            } else {
                let global_environment_ref = global_environment.borrow();
                let fun = fun
                    .or_else(|| global_environment_ref.get_function(name))
                    .expect(format!("function `{}` doesn't exist", name).as_str())
                    .clone();
                drop(global_environment_ref);
                stack.create_frame_push(format!("function_call_{}", name));
                let r = evaluate_function(
                    stack,
                    arguments.first().unwrap(),
                    &fun.parameters,
                    &fun.body,
                    global_environment,
                );
                stack.pop();
                if arguments.len() > 1 {
                    // r must be a function (closure)
                    evaluate_closure(&r, &arguments[1..].to_vec(), stack, global_environment)
                } else {
                    r
                }
            }
        }
        _ => panic!("expected function call"),
    }
}

pub fn evaluate_node(
    node: &Node,
    stack: &mut CallStack,
    global_environment: &Rc<RefCell<GlobalEnvironment>>,
) -> ValueRef {
    match node {
        Node::Program { statements } => {
            let mut last = ValueRef::stack(Value::Undefined);
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
            ValueRef::stack(Value::Struct {
                name: name.to_string(),
                fields: sd.fields.clone(),
            })
        }
        Node::FunctionDeclaration {
            name,
            parameters,
            return_type,
            body,
            lambda,
        } => {
            let func_def = create_function_definition(node);
            if !lambda {
                global_environment
                    .borrow_mut()
                    .functions
                    .insert(name.clone(), func_def);

                ValueRef::stack(Value::Function {
                    name: name.to_string(),
                    parameters: parameters
                        .iter()
                        .map(|p| Parameter {
                            name: p.0.clone(),
                            sk_type: p.1.clone(),
                        })
                        .collect(),
                    return_type: return_type.clone(),
                    body: Vec::new(),
                })
            } else {
                let frame = stack.current_frame().clone();
                ValueRef::stack(Value::Closure {
                    function: func_def,
                    env: frame.env,
                })
            }
        }
        Node::Literal(Literal::Integer(x)) => ValueRef::stack(Value::Integer(*x)),
        Node::Literal(Literal::StringLiteral(x)) => ValueRef::stack(Value::String(x.to_string())),
        Node::Literal(Literal::Boolean(x)) => ValueRef::stack(Value::Boolean(*x)),
        Node::VariableDeclaration { name, value, .. } => {
            let value = if let Some(body) = value {
                evaluate_node(body, stack, global_environment)
            } else {
                ValueRef::stack(Value::Undefined)
            };

            stack.declare_variable(name, value);

            ValueRef::stack(Value::Variable(name.to_string()))
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
            let mut values: Vec<ValueRef> = vec![];
            while !node_stack.is_empty() {
                let n = node_stack.pop_front().unwrap();
                match n {
                    Node::ArrayInit { elements } => {
                        for e in elements {
                            node_stack.push_back(e)
                        }
                    }
                    _ => values.push(evaluate_node(n, stack, global_environment)),
                }
            }

            ValueRef::stack(Value::Array {
                arr: values,
                dimensions,
            })
        }
        Node::StructInitialization { name, fields } => {
            let mut field_values: HashMap<String, ValueRef> = HashMap::new();
            for (name, node) in fields {
                field_values.insert(
                    name.to_string(),
                    evaluate_node(node, stack, global_environment),
                );
            }
            ValueRef::stack(Value::StructInstance {
                name: name.to_string(),
                fields: field_values,
            })
        }
        Node::Access { nodes } => resolve_access(
            stack,
            global_environment,
            ValueRef::stack(Value::Undefined),
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
                    ValueRef::stack(Value::Undefined),
                    0,
                    nodes,
                );
                let new_value = evaluate_node(value.as_ref(), stack, global_environment);
                modifier.set(new_value);
                modifier.get()
            }
            _ => panic!("expected access to assignment node"),
        },
        Node::Access { nodes } => resolve_access(
            stack,
            global_environment,
            ValueRef::stack(Value::Undefined),
            0,
            nodes,
        )
        .get(),
        Node::FunctionCall { .. } => evaluate_function_call(node, stack, global_environment, None),
        Node::Block { statements } => {
            stack.create_frame_push("".to_string());
            // drop(stack_ref);
            let mut res = ValueRef::stack(Value::Undefined);
            for statement in statements {
                res = evaluate_node(statement, stack, global_environment);
                if res.returned() {
                    break;
                }
            }

            stack.frames.pop();
            // mem::drop(stack_ref);
            res
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
                    let i = dim_val
                        .map(|v| match v {
                            Value::Integer(i) => Some(*i),
                            _ => None,
                        })
                        .expect("expected integer in array size");
                    int_dimensions.push(i);
                    size = size * (i) as usize;
                }
                let mut arr = Vec::with_capacity(size);
                for _ in 0..size {
                    arr.push(evaluate_node(&arguments[0], stack, global_environment));
                }
                ValueRef::stack(Value::Array {
                    arr,
                    dimensions: int_dimensions.clone(),
                })
            }
            _ => panic!("unsupported static function call type"),
        },
        Node::If {
            condition,
            body,
            else_if_blocks,
            else_block,
        } => {
            let value_ref = evaluate_node(condition, stack, global_environment);

            let ok = value_ref
                .map(|v| match v {
                    Value::Boolean(ok) => Some(*ok),
                    _ => None,
                })
                .expect(format!("non boolean expression: {:?}", condition.as_ref()).as_str());
            if ok {
                for n in body {
                    let val = evaluate_node(n, stack, global_environment);
                    if val.returned() {
                        return val;
                    }
                }
                return ValueRef::stack(Value::Executed);
            }

            for n in else_if_blocks {
                let val = evaluate_node(n, stack, global_environment);
                if val.returned() {
                    return val;
                }
                if val.is_match(|v| match v {
                    Value::Executed => true,
                    _ => false,
                }) {
                    return val;
                }
            }

            if let Some(nodes) = else_block {
                for n in nodes {
                    let val = evaluate_node(n, stack, global_environment);
                    if val.returned() {
                        return val;
                    }
                    if val.is_match(|v| match v {
                        Value::Executed => true,
                        _ => false,
                    }) {
                        return val;
                    }
                }
            }

            ValueRef::stack(Value::Void)
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
                    let res = evaluate_node(cond.as_ref(), stack, global_environment);
                    res.map(|v| match v {
                        Value::Boolean(ok) => Some(*ok),
                        _ => None,
                    })
                    .expect("for condition in should be a boolean expression")
                })
                .unwrap_or(true)
            {
                for n in body {
                    let val = evaluate_node(n, stack, global_environment);
                    if val.returned() {
                        return val;
                    }
                }
                if let Some(n) = update {
                    evaluate_node(n, stack, global_environment);
                }
            }
            ValueRef::stack(Value::Void)
        }
        Node::UnaryOp { operator, operand } => {
            let operand_val = evaluate_node(operand, stack, global_environment);
            match operator {
                ast::UnaryOperator::Plus => operand_val, // unary `+` doesn't change the value
                ast::UnaryOperator::Minus => operand_val.map_value(|v| match v {
                    Value::Integer(val) => Value::Integer(-val),
                    _ => panic!("Unary minus is only supported for integers"),
                }),
                _ => panic!("unsupported unary operator: {:?}", operator),
            }
        }
        /*Node::BinaryOp {
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
                (Value::String(s1), Value::String(s2)) => match operator {
                    Operator::Add => Rc::new(RefCell::new(Value::String(format!("{}{}", s1, s2)))),
                    _ => panic!("Unsupported operator for string concatenation"),
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
        }*/
        Node::Print(value) => {
            let val = evaluate_node(value, stack, global_environment);
            println!("{:?}", val);
            ValueRef::stack(Value::Void)
        }
        Node::Identifier(name) => {
            let v = stack
                .get_variable(name)
                .expect(format!("variable '{}' does not exist", name).as_str());
            // let res = Rc::clone(&v);
            // drop(stack_ref);
            v.clone()
        }
        Node::EOI => ValueRef::stack(Value::Void),
        Node::Return(body_opt) => {
            if let Some(body) = body_opt {
                let val = evaluate_node(body, stack, global_environment);
                // Rc::new(RefCell::new(Return(Rc::clone(&val))))
                val.to_returned()
            } else {
                ValueRef::stack(Value::Void)
            }
        }
        _ => panic!("Unexpected node type: {:?}", node),
    }
}
//
// use crate::interpreter::Value::{Closure, StructInstance, Undefined, Void};
// use pest::Stack;
// use std::fmt::Octal;
// use std::ops::{Deref, DerefMut};

#[cfg(test)]
mod tests {
    use super::*;

    /*

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
            arr:int[] = createArray();
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
        fn test_array_len() {
            let source_code = r#"
            arr: int[3] = int[3]::new(1);
            arr.len;
            "#;
            let program = ast::parse(source_code);
            let res = evaluate(&program);
            let res_ref = res.borrow();
            assert_eq!(Value::Integer(3), *res_ref.deref());
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
        assert_eq!(Value::Integer(10), *res_ref.deref());
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
        assert_eq!(Value::Integer(1212), *res_ref.deref());
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
            list:List = List{size: 3, capacity: 2};
            list.grow();
            list.capacity;
        "#;
        let program = ast::parse(source_code);
        let res = evaluate(&program);
        assert_eq!(Value::Integer(4), *res.borrow().deref());
    }

    #[test]
    fn test_lambda_recursive() {
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
    fn test_lambda_pass_as_arg() {
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
        let res = evaluate(&program);
        assert_eq!(Value::Integer(47), *res.borrow().deref());
    }

    #[test]
    fn test_anonymous_function_with_params() {
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
            };
        }

        f()(47);
        "#;
        let program = ast::parse(source_code);
        let res = evaluate(&program);
        assert_eq!(Value::Integer(47), *res.borrow().deref());
    }

    #[test]
    fn test_closure() {
        let source_code = r#"
        function f(): () -> int {
            c: int = 0;
            return function (): int {
                c = c + 1;
                return c;
            };
        }

        counter: () -> int = f();
        counter();
        counter();
        "#;
        let program = ast::parse(source_code);
        let res = evaluate(&program);
        assert_eq!(Value::Integer(2), *res.borrow().deref());
    }

    #[test]
    fn test_closure_capture_outer() {
        let source_code = r#"
            count: int = 0;
            task: () -> int = function(): int {
                if(count == 3) {
                    return 3;
                }
                count = count + 1;
                return task();
            }
            task();
        "#;
        let program = ast::parse(source_code);
        let res = evaluate(&program);
        assert_eq!(Value::Integer(3), *res.borrow().deref());
    }

    #[test]
    fn test_closure_outer() {
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
        let res = evaluate(&program);
        assert_eq!(Value::Integer(3), *res.borrow().deref());
    }

    #[test]
    fn test_nested_closures() {
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
        a()();
        "#;

        let program = ast::parse(source_code);
        let res = evaluate(&program);
        assert_eq!(Value::Integer(6), *res.borrow().deref());
    }

    #[test]
    fn test_struct_lambda() {
        let source_code = r#"
        struct Foo {
            function f(self): (int) -> int {
                return function(i:int): int {
                    return i;
                };
            }
        }
        foo:Foo = Foo{};
        foo.f()(1);
        "#;
        let program = ast::parse(source_code);
        let res = evaluate(&program);
        assert_eq!(Value::Integer(1), *res.borrow().deref());
    }

         */
}
