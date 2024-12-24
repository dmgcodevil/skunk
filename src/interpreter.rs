use rustc_hash::FxHashMap;
use std::any::Any;
use std::collections::LinkedList;
use sysinfo::{Pid, System};

use crate::ast;
use crate::interpreter::Value::StructInstance;
use ast::Literal;
use ast::Node;
use ast::Operator;
use ast::Type;
use std::cell::RefCell;
use std::fmt;
use std::io::BufRead;
use std::ops::{Deref, DerefMut};
use std::rc::Rc;

#[derive(PartialEq, Clone)]
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
        fields: FxHashMap<String, ValueRef>,
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
    Executed, // indicates that a code path has been executed
    Void,
    Undefined,
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Integer(val) => write!(f, "Integer({})", val),
            Value::String(val) => write!(f, "String(\"{}\")", val),
            Value::Boolean(val) => write!(f, "Boolean({})", val),
            Value::Variable(name) => write!(f, "Variable(\"{}\")", name),
            Value::Array { arr, dimensions } => write!(
                f,
                "Array {{ dimensions: {:?}, values: {:?} }}",
                dimensions, arr
            ),
            Value::StructInstance { name, fields } => write!(
                f,
                "StructInstance {{ name: {}, fields: {:?} }}",
                name, fields
            ),
            Value::Struct { name, fields } => {
                write!(f, "Struct {{ name: {}, fields: {:?} }}", name, fields)
            }
            Value::Function {
                name,
                parameters,
                return_type,
                body,
            } => write!(
                f,
                "Function {{ name: {}, parameters: {:?}, return_type: {:?}, body: {:?} }}",
                name, parameters, return_type, body
            ),
            Value::Closure { function, env } => {
                write!(f, "Closure {{{:?}}}", function)
            }
            Value::Executed => write!(f, "Executed"),
            Value::Void => write!(f, "Void"),
            Value::Undefined => write!(f, "Undefined"),
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum ValueRef {
    Stack {
        value: Value,
        returned: bool,
    },
    Heap {
        value: Rc<RefCell<Value>>,
        returned: bool,
    },
}

impl ValueRef {
    fn stack(value: Value) -> Self {
        ValueRef::Stack {
            value,
            returned: false,
        }
    }

    fn heap(value: Value) -> Self {
        ValueRef::Heap {
            value: Rc::new(RefCell::new(value)),
            returned: false,
        }
    }
    fn returned(&self) -> bool {
        match self {
            ValueRef::Stack { returned, .. } => *returned,
            ValueRef::Heap { returned, .. } => *returned,
        }
    }

    fn get_value(&self) -> Value {
        match self {
            ValueRef::Stack { value, .. } => value.clone(),
            ValueRef::Heap { value, .. } => value.borrow().clone(),
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

    fn map_match<T, F>(&self, f: F) -> Option<T>
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

    fn map<F>(&self, f: F) -> ValueRef
    where
        F: Fn(&Value) -> ValueRef,
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

    fn unwrap<F>(&self, mut f: F)
    where
        F: FnMut(&Value),
    {
        self.is_match(|a| {
            f(a);
            true
        });
    }

    fn is_match<F>(&self, mut f: F) -> bool
    where
        F: FnMut(&Value) -> bool,
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

    fn is_struct(&self) -> bool {
        self.is_match(|v| match v {
            Value::StructInstance { .. } => true,
            _ => false,
        })
    }

    fn is_array(&self) -> bool {
        self.is_match(|v| match v {
            Value::Array { .. } => true,
            _ => false,
        })
    }
    fn is_closure(&self) -> bool {
        self.is_match(|v| match v {
            Value::Closure { .. } => true,
            _ => false,
        })
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
    functions: FxHashMap<String, FunctionDefinition>,
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
    structs: FxHashMap<String, StructDefinition>,
    functions: FxHashMap<String, FunctionDefinition>,
}

impl GlobalEnvironment {
    fn new() -> Self {
        GlobalEnvironment {
            structs: FxHashMap::default(),
            functions: FxHashMap::default(),
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
            .map_match(|v| match v {
                Value::StructInstance { name, .. } => Some(self.structs.get(name).unwrap()),
                _ => panic!("expected struct instance. given: {:?}", v),
            })
            .unwrap()
    }
}

const SMALL_VAR_STORAGE_MAX_SIZE: usize = 10;

#[derive(Debug, Clone, PartialEq)]
enum VariableStorage {
    Small(Vec<(String, ValueRef)>),
    Large(FxHashMap<String, ValueRef>),
}

impl VariableStorage {
    fn small() -> VariableStorage {
        VariableStorage::Small(Vec::with_capacity(SMALL_VAR_STORAGE_MAX_SIZE))
    }
    fn large() -> VariableStorage {
        VariableStorage::Large(FxHashMap::default())
    }

    fn get(&self, name: &str) -> Option<&ValueRef> {
        match self {
            VariableStorage::Small(vars) => {
                for v in vars.iter() {
                    if v.0 == name {
                        return Some(&v.1);
                    }
                }
                None
            }
            VariableStorage::Large(vars) => vars.get(name),
        }
    }
    fn set_or_add(&mut self, name: &str, value: ValueRef) {
        // println!("VariableStorage set {}={:?}", name, value);
        match self {
            VariableStorage::Small(vars) => {
                for index in 0..vars.len() {
                    if vars[index].0 == name {
                        vars[index].1 = value;
                        return;
                    }
                }
                vars.push((name.to_string(), value));
            }
            VariableStorage::Large(vars) => {
                vars.insert(name.to_string(), value);
            }
        }
    }
    fn exists(&self, name: &str) -> bool {
        match self {
            VariableStorage::Small(vars) => {
                for v in vars.iter() {
                    if v.0 == name {
                        return true;
                    }
                }
                false
            }
            VariableStorage::Large(vars) => vars.contains_key(name),
        }
    }

    fn size(&self) -> usize {
        match self {
            VariableStorage::Small(vars) => vars.len(),
            VariableStorage::Large(vars) => vars.len(),
        }
    }

    fn is_small(&self) -> bool {
        match self {
            VariableStorage::Small(_) => true,
            VariableStorage::Large(_) => false,
        }
    }
}

pub struct VariableStorageIter<'a> {
    small_iter: Option<std::slice::Iter<'a, (String, ValueRef)>>,
    large_iter: Option<std::collections::hash_map::Iter<'a, String, ValueRef>>,
}

impl<'a> Iterator for VariableStorageIter<'a> {
    type Item = (&'a String, &'a ValueRef);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(iter) = &mut self.small_iter {
            if let Some((name, value)) = iter.next() {
                return Some((name, value));
            }
        }
        if let Some(iter) = &mut self.large_iter {
            return iter.next();
        }
        None
    }
}

impl VariableStorage {
    fn iter(&self) -> VariableStorageIter {
        match self {
            VariableStorage::Small(vec) => VariableStorageIter {
                small_iter: Some(vec.iter()),
                large_iter: None,
            },
            VariableStorage::Large(map) => VariableStorageIter {
                small_iter: None,
                large_iter: Some(map.iter()),
            },
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
struct Environment {
    id: String,
    parent: Option<Rc<RefCell<Environment>>>,
    variables: VariableStorage,
}

impl Environment {
    fn new() -> Self {
        Environment {
            id: "".to_string(), //uuid::Uuid::new_v4().to_string(), // uncomment for debugging
            parent: None,
            variables: VariableStorage::small(),
        }
    }

    fn get_parent_id(&self) -> String {
        match &self.parent {
            Some(p) => p.borrow().id.clone(),
            None => "None".to_string(),
        }
    }

    fn get_var(&self, name: &str) -> Option<ValueRef> {
        match self.variables.get(name) {
            Some(v) => return Some(v.clone()),
            None => {}
        }

        if let Some(parent) = &self.parent {
            let parent_ref = parent.borrow();
            parent_ref.get_var(name)
        } else {
            None
        }
    }

    fn assign_variable(&mut self, name: &str, value: ValueRef) -> bool {
        if self.variables.exists(name) {
            // println!("assign_variable env id={} assign var {} = {:?}", self.id, name, value);
            self.variables.set_or_add(name, value);
            true
        } else if let Some(parent) = &self.parent {
            let mut parent_ref = parent.borrow_mut();
            parent_ref.assign_variable(name, value)
        } else {
            false
        }
    }

    fn declare_variable(&mut self, name: &str, value: ValueRef) {
        // println!("declare_variable env id={} assign var {} = {:?}", self.id, name, value);
        self.variables.set_or_add(name, value);
        if self.variables.is_small() && self.variables.size() > SMALL_VAR_STORAGE_MAX_SIZE {
            let mut variables = VariableStorage::large();
            match &mut self.variables {
                VariableStorage::Small(ref mut vars) => {
                    while let Some(var) = vars.pop() {
                        let name = var.0.to_string();
                        variables.set_or_add(&name, var.1)
                    }
                }
                _ => unreachable!(),
            }
            self.variables = variables;
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

struct Profiling {}

impl Profiling {
    fn sk_debug_stack(stack: &CallStack) {
        println!("--- Debugging Call Stack ---");
        for (i, frame) in stack.frames.iter().enumerate() {
            println!("Frame {}: {}", i, frame.name);
            let env = frame.env.borrow();
            println!(
                "  Environment id: '{}', parent={}",
                &env.id,
                &env.get_parent_id()
            );
            match &env.variables {
                VariableStorage::Small(vars) => {
                    for (var_name, var_value) in vars {
                        println!("      Variable '{}' -> {:?}", var_name, var_value);
                    }
                }
                VariableStorage::Large(vars) => {
                    for (var_name, var_value) in vars {
                        println!("      Variable '{}' -> {:?}", var_name, var_value);
                    }
                }
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
                    + std::mem::size_of::<FxHashMap<String, Rc<RefCell<Value>>>>()
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
                .iter()
                .map(|val| Profiling::estimate_value_ref_size(val.1))
                .sum();
            println!("    Memory Used: {} bytes", frame_mem);
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CallFrame {
    name: String,
    env: Rc<RefCell<Environment>>,
    closure_cache: Vec<(String, Option<ValueRef>)>,
}

impl CallFrame {
    fn new(name: String) -> Self {
        CallFrame {
            name: name.to_string(),
            env: Rc::new(RefCell::new(Environment::new())),
            closure_cache: Vec::new(),
        }
    }

    fn set_parent(&mut self, parent: Rc<RefCell<Environment>>) {
        // println!("CallFrame-{}:: set_parent", self.name);
        let mut env = self.env.borrow_mut();
        if let Some(old_parent) = &env.parent {
            let old_parent = old_parent.borrow();
            // assert_ne!(*old_parent.deref(), *parent.borrow().deref());
        }
        env.parent = Some(parent);
        drop(env)
    }

    fn assign_variable(&mut self, name: &str, value: ValueRef) -> bool {
        // println!("CallFrame-{}::has_variable {}", self.name, name);
        let mut env_ref = self.env.borrow_mut();
        env_ref.assign_variable(name, value)
    }

    fn declare_variable(&mut self, name: &str, value: ValueRef) {
        // println!("CallFrame-{}::declare_variable {}", self.name, name);
        self.env.borrow_mut().declare_variable(name, value);
    }

    fn get_variable(&self, name: &str) -> Option<ValueRef> {
        let env_ref = self.env.borrow();
        let v = env_ref.get_var(name);
        // println!("CallFrame-{}::get_variable {} = {:?}", self.name, name, v);
        v
    }

    fn get_closure_mem(&mut self, name: &str) -> Option<ValueRef> {
        for v in self.closure_cache.iter() {
            if v.0 == name {
                return v.1.clone();
            }
        }

        let var_opt = self.env.borrow().get_var(name);
        let c = match var_opt {
            Some(var) => {
                if var.is_closure() {
                    Some(var)
                } else {
                    None
                }
            }
            None => None,
        };
        self.closure_cache.push((name.to_string(), c.clone()));
        c
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
        let mut frame = CallFrame::new(name.to_string());
        if let Some(curr) = self.frames.last() {
            frame.set_parent(curr.env.clone());
        }
        // println!("create frame: {}, env_id:{}", frame.name, frame.env.borrow().id);
        frame
    }

    fn create_frame_push(&mut self, name: String) {
        let frame = self.create_frame(name);
        self.frames.push(frame);
    }

    fn get_closure(&mut self, name: &str) -> Option<ValueRef> {
        for x in self.current_frame().closure_cache.iter() {
            if x.0 == name {
                return x.1.clone();
            }
        }

        let mut c = None;

        for frame in self.frames.iter_mut().rev() {
            c = frame.get_closure_mem(name);
            if c.is_some() {
                break;
            }
        }
        self.current_frame_mut()
            .closure_cache
            .push((name.to_string(), c.clone()));
        c
    }

    fn get_variable(&self, name: &str) -> Option<ValueRef> {
        for frame in self.frames.iter().rev() {
            if let Some(v) = frame.get_variable(name) {
                return Some(v);
            }
        }
        None
    }

    fn assign_variable(&mut self, name: &str, value: ValueRef) {
        // println!("assign_variable {}", name);
        let ok = self.current_frame_mut().assign_variable(name, value);
        assert_eq!(ok, true);
    }

    fn declare_variable(&mut self, name: &str, value: ValueRef) {
        // println!("declare variable {}={:?}", name, value);
        self.current_frame_mut().declare_variable(name, value);
    }

    fn current_frame(&self) -> &CallFrame {
        self.frames.last().unwrap()
        //.expect("call stack underflow: no active frames")
    }

    fn current_frame_mut(&mut self) -> &mut CallFrame {
        self.frames.last_mut().unwrap()
        //.expect("call stack underflow: no active frames")
    }
}

pub fn evaluate(node: &Node) -> ValueRef {
    let stack = Rc::new(RefCell::new(CallStack::new()));
    let ge = Rc::new(RefCell::new(GlobalEnvironment::new()));
    stack.borrow_mut().create_frame_push("main".to_string());
    evaluate_node(node, &stack, &ge)
}

fn assert_value_is_struct(v: &Value) {
    if !matches!(v, Value::StructInstance { .. }) {
        panic!("expected struct instance, found {:?}", v);
    }
}

fn get_struct_field(instance: ValueRef, field: &str) -> ValueRef {
    match instance {
        ValueRef::Stack { value, .. } => ValueRef::stack(value.clone()),
        ValueRef::Heap { value, .. } => {
            let value_ref = value.borrow();
            if let Value::StructInstance { name, fields } = value_ref.deref() {
                fields.get(field).expect("Field not found").clone()
            } else {
                panic!("expected a struct instance");
            }
        }
    }
}
fn set_struct_field(
    stack: &Rc<RefCell<CallStack>>,
    instance: ValueRef,
    field: &str,
    new_value: ValueRef,
) {
    match instance {
        ValueRef::Stack { .. } => {
            panic!("structs cannot be on stack")
            // if let Value::StructInstance {
            //     name,
            //     ref mut fields,
            // } = value
            // {
            //     fields.insert(field.to_string(), new_value);
            //     stack.borrow_mut().assign_variable(
            //         name,
            //         ValueRef::stack(StructInstance {
            //             name: name.to_string(),
            //             fields: mem::take(fields),
            //         }),
            //     );
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
                fields.insert(field.to_string(), new_value);
            } else {
                panic!("expected a struct instance");
            }
        }
    }
}

fn get_or_set_struct_field(
    stack: &Rc<RefCell<CallStack>>,
    instance: ValueRef,
    field: &str,
    new_value: Option<&ValueRef>,
) -> ValueRef {
    match new_value {
        None => get_struct_field(instance, field),
        Some(v) => {
            set_struct_field(stack, instance, field, v.clone());
            ValueRef::stack(Value::Undefined)
        }
    }
}

fn set_array_ref_element(arr_ref: &ValueRef, coordinates: &Vec<i64>, new_value: ValueRef) {
    match arr_ref {
        ValueRef::Stack { .. } => {
            panic!("arrays cannot be allocated on stack")
            //set_array_element(value, new_value, coordinates);
            //stack.assign_variable(name.as_str(), ValueRef::stack(value.clone()));
            //panic!("cannot mutate array allocated on stack")
        }
        ValueRef::Heap { value, .. } => {
            let mut value_ref = value.borrow_mut();
            set_array_element(value_ref.deref_mut(), new_value, coordinates)
        }
    }
}

fn get_array_ref_element(arr_ref: &ValueRef, coordinates: &Vec<i64>) -> ValueRef {
    match arr_ref {
        ValueRef::Stack { .. } => {
            panic!("arrays cannot be allocated on stack")
            //set_array_element(value, new_value, coordinates);
            //stack.assign_variable(name.as_str(), ValueRef::stack(value.clone()));
            //panic!("cannot mutate array allocated on stack")
        }
        ValueRef::Heap { value, .. } => {
            let mut value_ref = value.borrow();
            get_array_element(&value_ref, coordinates)
        }
    }
}

fn resolve_member_access(
    stack: &Rc<RefCell<CallStack>>,
    global_environment: &Rc<RefCell<GlobalEnvironment>>,
    current_value: ValueRef,
    node: &Node,
    new_value: Option<&ValueRef>,
) -> ValueRef {
    if let Node::MemberAccess { member, .. } = node {
        let member_ref = member.as_ref();
        match member_ref {
            Node::Identifier(name) => {
                if current_value.is_struct() {
                    get_or_set_struct_field(stack, current_value, name, new_value)
                } else if current_value.is_array() {
                    current_value.map(|v| match v {
                        Value::Array { arr, .. } => match name.as_str() {
                            "len" => ValueRef::stack(Value::Integer(arr.len() as i64)),
                            _ => {
                                unreachable!("{}", format!("unsupported array member: `{}`", name))
                            }
                        },
                        _ => panic!("expected an array"),
                    })
                } else {
                    unreachable!("{}", format!("unsupported member: `{}`", name));
                }
            }
            Node::FunctionCall {
                name, arguments, ..
            } => {
                let global_environment_ref = global_environment.borrow();
                let struct_def = global_environment_ref.to_struct_def(&current_value);
                let fun_def = struct_def.functions.get(name).unwrap();
                assert_eq!(fun_def.parameters.first().unwrap().sk_type, Type::SkSelf);

                stack
                    .borrow_mut()
                    .create_frame_push(format!("{}.{}", struct_def.name, name));
                stack
                    .borrow_mut()
                    .declare_variable("self", current_value.clone());

                Profiling::sk_debug_stack(stack.borrow().deref());

                let res =
                    evaluate_function_call(member_ref, stack, global_environment, Some(fun_def));
                stack.borrow_mut().pop();
                drop(global_environment_ref);
                res
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
    arr_ref: ValueRef,
    node: &Node,
    new_value: Option<&ValueRef>,
) -> ValueRef {
    if let Node::ArrayAccess { coordinates } = node {
        let _coordinates: Vec<i64> = coordinates
            .iter()
            .map(|n| {
                evaluate_node(n, stack, global_environment)
                    .map_match(|v| match v {
                        Value::Integer(dim) => Some(*dim),
                        _ => panic!("expected integer index for array access"),
                    })
                    .unwrap()
            })
            .collect();

        match new_value {
            None => get_array_ref_element(&arr_ref, &_coordinates),
            Some(v) => {
                set_array_ref_element(&arr_ref, &_coordinates, v.clone());
                ValueRef::stack(Value::Undefined)
            }
        }
    } else {
        panic!("expected array access, found {:?}", node)
    }
}

fn resolve_access(
    stack: &Rc<RefCell<CallStack>>,
    global_environment: &Rc<RefCell<GlobalEnvironment>>,
    current_value: ValueRef,
    access_nodes: &Vec<Node>,
    new_value: Option<&ValueRef>,
) -> ValueRef {
    let mut res = current_value;

    let n = access_nodes.len();

    for i in 0..n {
        let access_node = &access_nodes[i];
        let to_set = if i == n - 1 {
            // last node, we can modify it
            new_value.clone()
        } else {
            None
        };

        res = match access_node {
            Node::MemberAccess { .. } => {
                // println!("resolve_access-{}, current_value={:?}, member={:?}, to_set={:?}",
                //          i,
                //          res,
                //          access_node,
                //          to_set);
                resolve_member_access(stack, global_environment, res, access_node, to_set)
            }

            Node::ArrayAccess { .. } => {
                resolve_array_access(stack, global_environment, res, access_node, to_set)
            }
            Node::Identifier(name) => match to_set {
                None => stack.borrow().get_variable(name).unwrap().clone(),
                Some(v) => {
                    stack.borrow_mut().assign_variable(name, v.clone());
                    ValueRef::stack(Value::Undefined)
                }
            },
            _ => panic!("unexpected access node: {:?}", access_node),
        };
    }

    res
}

fn evaluate_function(
    stack: &Rc<RefCell<CallStack>>,
    arguments: &Vec<Node>,
    parameters: &Vec<Parameter>,
    body: &Vec<Node>,
    global_environment: &Rc<RefCell<GlobalEnvironment>>,
) -> ValueRef {
    let mut parameters = &parameters[0..];
    if parameters.len() > 0 && parameters[0].sk_type == Type::SkSelf {
        parameters = &parameters[1..];
    }
    let n = parameters.len();
    for i in 0..n {
        let param = &parameters[i];
        let v = evaluate_node(&arguments[i], stack, global_environment);
        stack.borrow_mut().declare_variable(&param.name, v);
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
    stack: &Rc<RefCell<CallStack>>,
    global_environment: &Rc<RefCell<GlobalEnvironment>>,
) -> ValueRef {
    let mut curr = value;
    let mut result = ValueRef::stack(Value::Undefined);
    for arg in arguments {
        let new_result = match curr {
            ValueRef::Stack { value, .. } => match value {
                Value::Closure { function, env } => {
                    let mut frame = CallFrame::new(function.name.clone());
                    frame.set_parent(env.clone());
                    stack.borrow_mut().push(frame);
                    let r = evaluate_function(
                        stack,
                        arg,
                        &function.parameters,
                        &function.body,
                        global_environment,
                    );
                    stack.borrow_mut().pop();
                    r
                }
                _ => panic!("expected closure"),
            },
            ValueRef::Heap { value, .. } => {
                let value_ref = value.borrow();
                match value_ref.deref() {
                    Value::Closure { function, env } => {
                        let mut frame = CallFrame::new(function.name.clone());
                        frame.set_parent(env.clone());
                        stack.borrow_mut().push(frame);
                        let r = evaluate_function(
                            stack,
                            arg,
                            &function.parameters,
                            &function.body,
                            global_environment,
                        );
                        stack.borrow_mut().pop();
                        r
                    }
                    _ => panic!("expected closure"),
                }
            }
        };
        result = new_result;
        curr = &result;
    }
    result
}

fn evaluate_function_call(
    node: &Node,
    stack: &Rc<RefCell<CallStack>>,
    global_environment: &Rc<RefCell<GlobalEnvironment>>,
    fun: Option<&FunctionDefinition>,
) -> ValueRef {
    match node {
        Node::FunctionCall {
            name, arguments, ..
        } => {
            if name == "sk_debug_mem" {
                Profiling::sk_debug_mem_sysinfo();
                Profiling::sk_debug_mem_programmatic(stack.borrow().deref());
                return ValueRef::stack(Value::Undefined);
            }

            if name == "sk_debug_stack" {
                Profiling::sk_debug_stack(stack.borrow().deref());
                return ValueRef::stack(Value::Undefined);
            }

            let value_opt = { stack.borrow_mut().get_closure(name) };
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
                    .unwrap();

                stack.borrow_mut().create_frame_push(name.to_string());
                let r = evaluate_function(
                    stack,
                    arguments.first().unwrap(),
                    &fun.parameters,
                    &fun.body,
                    global_environment,
                );
                drop(global_environment_ref);
                stack.borrow_mut().pop();

                if arguments.len() > 1 {
                    // func chain
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
    stack: &Rc<RefCell<CallStack>>,
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
            let mut function_defs: FxHashMap<String, FunctionDefinition> = FxHashMap::default();
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
                    .insert(name.clone(), func_def.clone());
            }

            if *lambda {
                ValueRef::heap(Value::Closure {
                    function: func_def,
                    env: stack.borrow().current_frame().env.clone(),
                })
            } else {
                ValueRef::stack(
                    (Value::Function {
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
                    }),
                )
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

            stack.borrow_mut().declare_variable(name, value);
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

            ValueRef::heap(Value::Array {
                arr: values,
                dimensions,
            })
        }
        Node::StructInitialization { name, fields } => {
            let mut field_values: FxHashMap<String, ValueRef> = FxHashMap::default();
            for (name, node) in fields {
                field_values.insert(
                    name.to_string(),
                    evaluate_node(node, stack, global_environment),
                );
            }
            ValueRef::heap(Value::StructInstance {
                name: name.to_string(),
                fields: field_values,
            })
        }
        Node::Access { nodes } => {
            // println!("resolve access: {:?}", nodes);
            resolve_access(
                stack,
                global_environment,
                ValueRef::stack(Value::Undefined),
                nodes,
                None,
            )
        }
        Node::Assignment { var, value, .. } => match var.as_ref() {
            Node::Access { nodes } => {
                let new_value = evaluate_node(value.as_ref(), stack, global_environment);
                resolve_access(
                    stack,
                    global_environment,
                    ValueRef::stack(Value::Undefined),
                    nodes,
                    Some(&new_value),
                )
            }
            _ => panic!("expected access to assignment node"),
        },
        Node::Access { nodes } => resolve_access(
            stack,
            global_environment,
            ValueRef::stack(Value::Undefined),
            nodes,
            None,
        ),
        Node::FunctionCall { .. } => evaluate_function_call(node, stack, global_environment, None),
        Node::Block { statements } => {
            stack.borrow_mut().create_frame_push("".to_string());
            let mut res = ValueRef::stack(Value::Undefined);
            for statement in statements {
                res = evaluate_node(statement, stack, global_environment);
                if res.returned() {
                    break;
                }
            }

            stack.borrow_mut().frames.pop();
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
                        .map_match(|v| match v {
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
                ValueRef::heap(Value::Array {
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
                .map_match(|v| match v {
                    Value::Boolean(ok) => Some(*ok),
                    _ => None,
                })
                .unwrap();
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
                    res.map_match(|v| match v {
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
        Node::BinaryOp {
            left,
            operator,
            right,
        } => {
            let left_val = evaluate_node(left, stack, global_environment);
            let right_val = evaluate_node(right, stack, global_environment);
            let mut res = Value::Undefined;
            left_val.unwrap(|left_inner| {
                right_val.unwrap(|right_inner| {
                    res = match (left_inner, right_inner) {
                        (Value::Integer(l), Value::Integer(r)) => match operator {
                            Operator::Add => Value::Integer(l + r),
                            Operator::Subtract => Value::Integer(l - r),
                            Operator::Multiply => Value::Integer(l * r),
                            Operator::Divide => Value::Integer(l / r),
                            Operator::Mod => Value::Integer(l % r),
                            Operator::Equals => Value::Boolean(l == r),
                            Operator::LessThanOrEqual => Value::Boolean(l <= r),
                            Operator::LessThan => Value::Boolean(l < r),
                            Operator::GreaterThanOrEqual => Value::Boolean(l >= r),
                            Operator::GreaterThan => Value::Boolean(l > r),
                            _ => panic!("Unsupported operator"),
                        },
                        (Value::Boolean(a), Value::Boolean(b)) => match operator {
                            Operator::And => Value::Boolean(*a && *b),
                            Operator::Or => Value::Boolean(*a || *b),
                            _ => panic!("Unsupported operator"),
                        },
                        (Value::String(s1), Value::String(s2)) => match operator {
                            Operator::Add => Value::String(format!("{}{}", s1, s2)),
                            _ => panic!("Unsupported operator for string concatenation"),
                        },
                        (Value::String(s), Value::Integer(i)) => match operator {
                            Operator::Add => Value::String(format!("{}{}", s, i)),
                            _ => panic!("Unsupported operator for string concatenation"),
                        },
                        (Value::Integer(i), Value::String(s)) => match operator {
                            Operator::Add => Value::String(format!("{}{}", i, s)),
                            _ => panic!("Unsupported operator for string concatenation"),
                        },
                        (Value::Boolean(i), Value::String(s)) => match operator {
                            Operator::Add => Value::String(format!("{}{}", i, s)),
                            _ => panic!("Unsupported operator for string concatenation"),
                        },
                        (Value::String(s), Value::Boolean(i)) => match operator {
                            Operator::Add => Value::String(format!("{}{}", s, i)),
                            _ => panic!("Unsupported operator for string concatenation"),
                        },
                        _ => panic!(
                            "Unsupported binary operation={:?}, left={:?}, right={:?}",
                            operator, left_val, right_val
                        ),
                        _ => panic!("Unsupported operator"),
                    }
                })
            });
            ValueRef::stack(res)
        }

        Node::Print(value) => {
            let val = evaluate_node(value, stack, global_environment);
            println!("{:?}", val);
            ValueRef::stack(Value::Void)
        }
        Node::Identifier(name) => stack.borrow().get_variable(name).unwrap().clone(),
        Node::EOI => ValueRef::stack(Value::Void),
        Node::Return(body_opt) => {
            if let Some(body) = body_opt {
                evaluate_node(body, stack, global_environment).to_returned()
            } else {
                ValueRef::stack(Value::Void)
            }
        }
        _ => panic!("Unexpected node type: {:?}", node),
    }
}

#[cfg(test)]
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
        assert_eq!(Value::Integer(2), res.get_value());
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
        assert_eq!(Value::Integer(2), res.get_value());
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
        println!("{:?}", res);
        assert_eq!(
            Value::StructInstance {
                name: "Point".to_string(),
                fields: FxHashMap::from_iter([
                    ("x".to_string(), ValueRef::stack(Value::Integer(3))),
                    ("y".to_string(), ValueRef::stack(Value::Integer(4)))
                ])
            },
            res.get_value()
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
        assert_eq!(
            StructInstance {
                name: "Point".to_string(),
                fields: FxHashMap::from_iter([(
                    "c".to_string(),
                    ValueRef::heap(Value::Array {
                        arr: [
                            ValueRef::stack(Value::Integer(1)),
                            ValueRef::stack(Value::Integer(2))
                        ]
                        .to_vec(),
                        dimensions: [2].to_vec()
                    })
                )])
            },
            res.get_value()
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
        assert_eq!(Value::Integer(3), res.get_value());
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
            assert_eq!(Value::Integer(3), res.get_value());
        }
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
        assert_eq!(Value::Integer(5), res.get_value());
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
        assert_eq!(Value::Integer(8), res.get_value());
    }

    #[test]
    fn test_nested_for_loops_2d_array() {
        let source_code = r#"
                             arr: int[2][2] = int[2][2]::new(0);
                             for (i:int = 0; i < 2; i = i + 1) {
                             sk_debug_stack();
                                 for (j:int = 0; j < 2; j = j + 1) {
                                    sk_debug_stack();
                                     arr[i][j] = i + j;
                                 }
                             }
                             arr[1][1];
                         "#;
        let program = ast::parse(source_code);
        let res = evaluate(&program);
        assert_eq!(Value::Integer(2), res.get_value());
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
        assert_eq!(Value::Integer(2), res.get_value());
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
        assert_eq!(Value::Integer(1), res.get_value());
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
        assert_eq!(Value::Integer(10), res.get_value());
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
        assert_eq!(Value::Integer(20), res.get_value());
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
        assert_eq!(Value::Integer(3), res.get_value());
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
        assert_eq!(Value::Integer(4), res.get_value());
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
        assert_eq!(Value::Integer(10), res.get_value());
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
        assert_eq!(Value::Boolean(false), res.get_value());
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
        assert_eq!(Value::Integer(0), res.get_value());
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
        assert_eq!(Value::Integer(2), res.get_value());
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
        assert_eq!(Value::Integer(30 + 2), res.get_value());
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
        assert_eq!(Value::Integer(6), res.get_value());
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
        assert_eq!(Value::Integer(23), res.get_value()); // i=2, j=3 gives 6
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
        assert_eq!(Value::Integer(10), res.get_value());
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
        assert_eq!(Value::Integer(72), res.get_value());
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
        assert_eq!(Value::Integer(30), res.get_value());
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
        assert_eq!(Value::Integer(1212), res.get_value());
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
        assert_eq!(Value::Integer(10), res.get_value());
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
        assert_eq!(Value::Integer(42), res.get_value());
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
        assert_eq!(Value::Integer(20), res.get_value());
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
        assert_eq!(Value::Integer(5), res.get_value());
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

        match res.get_value() {
            Value::Array { arr, .. } => {
                let mut n = 1;
                for e in arr.iter() {
                    let e_ref = e.get_value();
                    if let Value::Integer(i) = e_ref {
                        assert_eq!(n + 10, i);
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
                        // sk_debug_stack();
                         p.set_x(3);
                         p.get_x();
                         "#;
        let program = ast::parse(source_code);
        println!("{:#?}", program);
        let res = evaluate(&program);
        assert_eq!(Value::Integer(3), res.get_value());
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
        assert_eq!(Value::Integer(1), res.get_value());
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
        assert_eq!(Value::Integer(1), res.get_value());
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
        assert_eq!(Value::Integer(4), res.get_value());
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
        assert_eq!(Value::Integer(6), res.get_value());
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
        assert_eq!(Value::Integer(1), res.get_value());
    }

    #[test]
    fn test_anonymous_function() {
        let source_code = r#"
                         function f(g: () -> int): int {
                             //sk_debug_stack();
                             return 47;
                             //return g();
                         }

                         f(function ():int {
                             return 47;
                         });
                         "#;

        let program = ast::parse(source_code);
        let res = evaluate(&program);
        assert_eq!(Value::Integer(47), res.get_value());
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
        assert_eq!(Value::Integer(47), res.get_value());
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
        assert_eq!(Value::Integer(47), res.get_value());
    }

    #[test]
    fn test_closure() {
        let source_code = r#"
                         function f(): () -> int {
                             counter: int = 0;
                             return function (): int {
                                 counter = counter + 1;
                                 return counter;
                             };
                         }
                         counter: () -> int = f();
                         counter();
                         counter();
                         "#;
        let program = ast::parse(source_code);
        let res = evaluate(&program);
        assert_eq!(Value::Integer(2), res.get_value());
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
        assert_eq!(Value::Integer(3), res.get_value());
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
        assert_eq!(Value::Integer(3), res.get_value());
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
        assert_eq!(Value::Integer(6), res.get_value());
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
        assert_eq!(Value::Integer(1), res.get_value());
    }
}
