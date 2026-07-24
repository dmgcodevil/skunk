//! Multi-file source loading for the syntax phase.
//!
//! Every file is parsed into the canonical syntax AST. Imports are expanded in
//! dependency order and module-private names are made unique before lexical
//! resolution sees the merged compilation unit.

use super::ast::*;
use crate::source_map::SourceMap;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path as FsPath, PathBuf};

#[derive(Debug)]
pub struct LoadedProgram {
    pub module: Module,
    pub sources: SourceMap,
}

pub fn load_program(entry_path: &FsPath) -> Result<LoadedProgram, String> {
    let entry_path = fs::canonicalize(entry_path)
        .map_err(|error| format!("failed to resolve `{}`: {error}", entry_path.display()))?;
    let module_root = entry_path
        .parent()
        .ok_or_else(|| format!("`{}` has no parent directory", entry_path.display()))?
        .to_path_buf();
    let mut loader = ProgramLoader::new(module_root);
    let root = loader.load_file(&entry_path, None, true)?;
    Ok(LoadedProgram {
        module: Module {
            id: root.id,
            span: root.span,
            name: root.name,
            entries: root.entries,
        },
        sources: loader.sources,
    })
}

struct ProgramLoader {
    module_root: PathBuf,
    sources: SourceMap,
    visited: HashSet<PathBuf>,
    loading: HashSet<PathBuf>,
}

impl ProgramLoader {
    fn new(module_root: PathBuf) -> Self {
        Self {
            module_root,
            sources: SourceMap::default(),
            visited: HashSet::new(),
            loading: HashSet::new(),
        }
    }

    fn load_file(
        &mut self,
        file_path: &FsPath,
        expected_module: Option<&str>,
        is_entry: bool,
    ) -> Result<Module, String> {
        let file_path = fs::canonicalize(file_path)
            .map_err(|error| format!("failed to resolve `{}`: {error}", file_path.display()))?;

        if self.visited.contains(&file_path) {
            let file = crate::ids::FileId::new(0);
            return Ok(Module {
                id: crate::ids::NodeId::in_file(file, 0),
                span: crate::source_map::Span::empty(file),
                name: None,
                entries: Vec::new(),
            });
        }
        if !self.loading.insert(file_path.clone()) {
            return Err(format!(
                "cyclic import detected while loading `{}`",
                file_path.display()
            ));
        }

        let result = self.load_new_file(&file_path, expected_module, is_entry);
        self.loading.remove(&file_path);
        if result.is_ok() {
            self.visited.insert(file_path);
        }
        result
    }

    fn load_new_file(
        &mut self,
        file_path: &FsPath,
        expected_module: Option<&str>,
        is_entry: bool,
    ) -> Result<Module, String> {
        let contents = fs::read_to_string(file_path)
            .map_err(|error| format!("failed to read `{}`: {error}", file_path.display()))?;
        let file = self
            .sources
            .add_file(file_path, contents)
            .map_err(|error| error.to_string())?;
        let module = super::parser::parse_module(&self.sources, file).map_err(|diagnostics| {
            format!(
                "failed to parse `{}`: {}",
                file_path.display(),
                render_diagnostics(diagnostics, &self.sources)
            )
        })?;
        let mut module = super::normalize::normalize(module)
            .map_err(|diagnostics| render_diagnostics(diagnostics, &self.sources))?;

        let declared_module = module.name.as_ref().map(Path::qualified_name);
        if let Some(expected) = expected_module {
            match declared_module.as_deref() {
                Some(actual) if actual == expected => {}
                Some(actual) => {
                    return Err(format!(
                        "module declaration mismatch in `{}`: expected `{expected}`, found `{actual}`",
                        file_path.display()
                    ));
                }
                None => {
                    return Err(format!(
                        "imported file `{}` must declare `module {expected};`",
                        file_path.display()
                    ));
                }
            }
        } else if !is_entry && declared_module.is_none() {
            return Err(format!(
                "imported file `{}` is missing a module declaration",
                file_path.display()
            ));
        }

        ModuleRenamer::new(declared_module.as_deref(), !is_entry, &module.entries)?
            .rename(&mut module);

        let mut entries = Vec::new();
        for entry in module.entries {
            if let TopLevelKind::Import(import) = &entry.kind {
                let name = import.module.qualified_name();
                let import_path = if crate::sdk::is_std_module(&name) {
                    crate::sdk::std_module_path(&name)?
                } else {
                    self.module_path(&name)
                };
                let imported = self.load_file(&import_path, Some(&name), false)?;
                entries.extend(imported.entries);
            } else {
                entries.push(entry);
            }
        }
        module.entries = entries;
        Ok(module)
    }

    fn module_path(&self, module_name: &str) -> PathBuf {
        let mut path = self.module_root.clone();
        for segment in module_name.split('.') {
            path.push(segment);
        }
        path.set_extension("skunk");
        path
    }
}

fn render_diagnostics(
    diagnostics: Vec<crate::diagnostic::Diagnostic>,
    sources: &SourceMap,
) -> String {
    diagnostics
        .into_iter()
        .map(|diagnostic| diagnostic.render(sources))
        .collect::<Vec<_>>()
        .join("\n")
}

struct ModuleRenamer {
    values: HashMap<String, String>,
    types: HashMap<String, String>,
}

impl ModuleRenamer {
    fn new(
        module_name: Option<&str>,
        rename_private: bool,
        entries: &[TopLevel],
    ) -> Result<Self, String> {
        let mut values = HashMap::new();
        let mut types = HashMap::new();
        let has_exports = entries
            .iter()
            .any(|entry| entry.visibility == Visibility::Public);
        if rename_private && has_exports {
            let module_name = module_name
                .ok_or_else(|| "cannot rename an imported module without a name".to_string())?;
            for entry in entries {
                if entry.visibility == Visibility::Public {
                    continue;
                }
                match declaration_name(&entry.kind) {
                    Some((Namespace::Value, name)) => {
                        values.insert(name.to_string(), mangle(module_name, name));
                    }
                    Some((Namespace::Type, name)) => {
                        types.insert(name.to_string(), mangle(module_name, name));
                    }
                    None => {}
                }
            }
        }
        Ok(Self { values, types })
    }

    fn rename(&self, module: &mut Module) {
        for entry in &mut module.entries {
            self.top_level(
                entry,
                true,
                &mut vec![HashSet::new()],
                &mut vec![HashSet::new()],
            );
        }
    }

    fn top_level(
        &self,
        entry: &mut TopLevel,
        module_scope: bool,
        value_scopes: &mut Vec<HashSet<String>>,
        type_scopes: &mut Vec<HashSet<String>>,
    ) {
        let exported = entry.visibility == Visibility::Public;
        match &mut entry.kind {
            TopLevelKind::Import(_) => {}
            TopLevelKind::TypeAlias(alias) => {
                if module_scope && !exported {
                    self.rename_type_declaration(&mut alias.name);
                }
                self.with_generics(
                    &mut alias.generic_parameters,
                    value_scopes,
                    type_scopes,
                    |this, values, types| this.ty(&mut alias.target, values, types),
                );
            }
            TopLevelKind::Struct(declaration) => {
                if module_scope && !exported {
                    self.rename_type_declaration(&mut declaration.name);
                }
                self.with_generics(
                    &mut declaration.generic_parameters,
                    value_scopes,
                    type_scopes,
                    |this, values, types| {
                        for field in &mut declaration.fields {
                            this.ty(&mut field.ty, values, types);
                        }
                        for method in &mut declaration.methods {
                            this.function(method, false, values, types);
                        }
                    },
                );
            }
            TopLevelKind::Enum(declaration) => {
                if module_scope && !exported {
                    self.rename_type_declaration(&mut declaration.name);
                }
                self.with_generics(
                    &mut declaration.generic_parameters,
                    value_scopes,
                    type_scopes,
                    |this, values, types| {
                        for variant in &mut declaration.variants {
                            for payload in &mut variant.payload {
                                this.ty(payload, values, types);
                            }
                        }
                        for method in &mut declaration.methods {
                            this.function(method, false, values, types);
                        }
                    },
                );
            }
            TopLevelKind::Trait(declaration) => {
                if module_scope && !exported {
                    self.rename_type_declaration(&mut declaration.name);
                }
                self.with_generics(
                    &mut declaration.generic_parameters,
                    value_scopes,
                    type_scopes,
                    |this, values, types| {
                        for supertrait in &mut declaration.supertraits {
                            this.type_path(supertrait, types);
                        }
                        for method in &mut declaration.methods {
                            this.trait_method(method, values, types);
                        }
                    },
                );
            }
            TopLevelKind::Shape(declaration) => {
                if module_scope && !exported {
                    self.rename_type_declaration(&mut declaration.name);
                }
                for method in &mut declaration.methods {
                    self.trait_method(method, value_scopes, type_scopes);
                }
            }
            TopLevelKind::Attach(declaration) => {
                self.with_generics(
                    &mut declaration.generic_parameters,
                    value_scopes,
                    type_scopes,
                    |this, values, types| {
                        this.ty(&mut declaration.target, values, types);
                        for method in &mut declaration.methods {
                            this.function(method, false, values, types);
                        }
                    },
                );
            }
            TopLevelKind::Conformance(declaration) => {
                self.with_generics(
                    &mut declaration.generic_parameters,
                    value_scopes,
                    type_scopes,
                    |this, values, types| {
                        for trait_type in &mut declaration.traits {
                            this.ty(trait_type, values, types);
                        }
                        this.ty(&mut declaration.target, values, types);
                        for method in &mut declaration.methods {
                            this.function(method, false, values, types);
                        }
                    },
                );
            }
            TopLevelKind::Implementation(declaration) => {
                self.with_generics(
                    &mut declaration.generic_parameters,
                    value_scopes,
                    type_scopes,
                    |this, values, types| {
                        for trait_type in &mut declaration.traits {
                            this.ty(trait_type, values, types);
                        }
                        this.ty(&mut declaration.target, values, types);
                    },
                );
            }
            TopLevelKind::Function(function) => {
                if module_scope && !exported {
                    self.rename_value_declaration(&mut function.name);
                }
                self.function(function, false, value_scopes, type_scopes);
            }
            TopLevelKind::ExternFunction(function) => {
                for parameter in &mut function.parameters {
                    self.parameter_type(parameter, value_scopes, type_scopes);
                }
                self.ty(&mut function.return_type, value_scopes, type_scopes);
            }
            TopLevelKind::Global(global) => {
                if module_scope && !exported {
                    self.rename_value_declaration(&mut global.name);
                }
                self.ty(&mut global.ty, value_scopes, type_scopes);
                if let Some(initializer) = &mut global.initializer {
                    self.expr(initializer, value_scopes, type_scopes);
                }
            }
            TopLevelKind::Test(test) => self.block(&mut test.body, value_scopes, type_scopes),
            TopLevelKind::Statement(statement) => self.stmt(statement, value_scopes, type_scopes),
        }
    }

    fn with_generics(
        &self,
        parameters: &mut [GenericParameter],
        value_scopes: &mut Vec<HashSet<String>>,
        type_scopes: &mut Vec<HashSet<String>>,
        body: impl FnOnce(&Self, &mut Vec<HashSet<String>>, &mut Vec<HashSet<String>>),
    ) {
        type_scopes.push(
            parameters
                .iter()
                .map(|parameter| parameter.name.clone())
                .collect(),
        );
        for parameter in parameters {
            for capability in &mut parameter.capabilities {
                self.type_path(capability, type_scopes);
            }
            if let Some(lower) = &mut parameter.lower_bound {
                self.ty(lower, value_scopes, type_scopes);
            }
            if let Some(upper) = &mut parameter.upper_bound {
                self.ty(upper, value_scopes, type_scopes);
            }
        }
        body(self, value_scopes, type_scopes);
        type_scopes.pop();
    }

    fn function(
        &self,
        function: &mut FunctionDecl,
        rename_name: bool,
        value_scopes: &mut Vec<HashSet<String>>,
        type_scopes: &mut Vec<HashSet<String>>,
    ) {
        if rename_name {
            self.rename_value_declaration(&mut function.name);
        }
        self.with_generics(
            &mut function.generic_parameters,
            value_scopes,
            type_scopes,
            |this, values, types| {
                let mut locals = HashSet::new();
                for parameter in &mut function.parameters {
                    this.parameter_type(parameter, values, types);
                    match &parameter.kind {
                        ParameterKind::Named { name, .. } => {
                            locals.insert(name.clone());
                        }
                        ParameterKind::Receiver { .. } => {
                            locals.insert("self".to_string());
                        }
                    }
                }
                this.ty(&mut function.return_type, values, types);
                values.push(locals);
                this.block_contents(&mut function.body, values, types);
                values.pop();
            },
        );
    }

    fn trait_method(
        &self,
        method: &mut TraitMethod,
        value_scopes: &mut Vec<HashSet<String>>,
        type_scopes: &mut Vec<HashSet<String>>,
    ) {
        let mut locals = HashSet::new();
        for parameter in &mut method.parameters {
            self.parameter_type(parameter, value_scopes, type_scopes);
            match &parameter.kind {
                ParameterKind::Named { name, .. } => {
                    locals.insert(name.clone());
                }
                ParameterKind::Receiver { .. } => {
                    locals.insert("self".to_string());
                }
            }
        }
        self.ty(&mut method.return_type, value_scopes, type_scopes);
        if let Some(body) = &mut method.default_body {
            value_scopes.push(locals);
            self.block_contents(body, value_scopes, type_scopes);
            value_scopes.pop();
        }
    }

    fn parameter_type(
        &self,
        parameter: &mut Parameter,
        value_scopes: &mut Vec<HashSet<String>>,
        type_scopes: &mut Vec<HashSet<String>>,
    ) {
        if let ParameterKind::Named { ty, .. } = &mut parameter.kind {
            self.ty(ty, value_scopes, type_scopes);
        }
    }

    fn block(
        &self,
        block: &mut Block,
        value_scopes: &mut Vec<HashSet<String>>,
        type_scopes: &mut Vec<HashSet<String>>,
    ) {
        value_scopes.push(HashSet::new());
        self.block_contents(block, value_scopes, type_scopes);
        value_scopes.pop();
    }

    fn block_contents(
        &self,
        block: &mut Block,
        value_scopes: &mut Vec<HashSet<String>>,
        type_scopes: &mut Vec<HashSet<String>>,
    ) {
        for statement in &mut block.statements {
            self.stmt(statement, value_scopes, type_scopes);
        }
    }

    fn stmt(
        &self,
        statement: &mut Stmt,
        value_scopes: &mut Vec<HashSet<String>>,
        type_scopes: &mut Vec<HashSet<String>>,
    ) {
        match &mut statement.kind {
            StmtKind::Local(local) => {
                self.ty(&mut local.ty, value_scopes, type_scopes);
                if let Some(initializer) = &mut local.initializer {
                    self.expr(initializer, value_scopes, type_scopes);
                }
                value_scopes
                    .last_mut()
                    .expect("local scope exists")
                    .insert(local.name.clone());
            }
            StmtKind::StructDestructure(pattern) => {
                self.ty(&mut pattern.ty, value_scopes, type_scopes);
                self.expr(&mut pattern.value, value_scopes, type_scopes);
                let scope = value_scopes.last_mut().expect("local scope exists");
                scope.extend(pattern.fields.iter().map(|field| field.binding.clone()));
            }
            StmtKind::Assignment { target, value } => {
                self.expr(target, value_scopes, type_scopes);
                self.expr(value, value_scopes, type_scopes);
            }
            StmtKind::Expression(expression)
            | StmtKind::Defer(expression)
            | StmtKind::Print(expression) => self.expr(expression, value_scopes, type_scopes),
            StmtKind::Return(expression) => {
                if let Some(expression) = expression {
                    self.expr(expression, value_scopes, type_scopes);
                }
            }
            StmtKind::Input => {}
            StmtKind::Declaration(declaration) => {
                self.top_level(declaration, false, value_scopes, type_scopes)
            }
            StmtKind::Block(block) | StmtKind::Unsafe(block) => {
                self.block(block, value_scopes, type_scopes)
            }
            StmtKind::If(branch) => {
                self.expr(&mut branch.condition, value_scopes, type_scopes);
                self.block(&mut branch.then_block, value_scopes, type_scopes);
                for (condition, block) in &mut branch.else_if {
                    self.expr(condition, value_scopes, type_scopes);
                    self.block(block, value_scopes, type_scopes);
                }
                if let Some(block) = &mut branch.else_block {
                    self.block(block, value_scopes, type_scopes);
                }
            }
            StmtKind::Match(branch) => {
                self.expr(&mut branch.value, value_scopes, type_scopes);
                for case in &mut branch.cases {
                    self.pattern(&mut case.pattern, value_scopes, type_scopes);
                    value_scopes.push(pattern_bindings(&case.pattern));
                    self.block_contents(&mut case.body, value_scopes, type_scopes);
                    value_scopes.pop();
                }
            }
            StmtKind::For(loop_statement) => {
                value_scopes.push(HashSet::new());
                if let Some(initializer) = &mut loop_statement.initializer {
                    self.stmt(initializer, value_scopes, type_scopes);
                }
                if let Some(condition) = &mut loop_statement.condition {
                    self.expr(condition, value_scopes, type_scopes);
                }
                if let Some(update) = &mut loop_statement.update {
                    self.stmt(update, value_scopes, type_scopes);
                }
                self.block(&mut loop_statement.body, value_scopes, type_scopes);
                value_scopes.pop();
            }
        }
    }

    fn pattern(
        &self,
        pattern: &mut Pattern,
        value_scopes: &mut Vec<HashSet<String>>,
        type_scopes: &mut Vec<HashSet<String>>,
    ) {
        match pattern {
            Pattern::EnumVariant { enum_type, .. } => {
                if let Some(enum_type) = enum_type {
                    self.ty(enum_type, value_scopes, type_scopes);
                }
            }
            Pattern::Struct { ty, .. } => self.ty(ty, value_scopes, type_scopes),
        }
    }

    fn expr(
        &self,
        expression: &mut Expr,
        value_scopes: &mut Vec<HashSet<String>>,
        type_scopes: &mut Vec<HashSet<String>>,
    ) {
        match &mut expression.kind {
            ExprKind::Literal(_) => {}
            ExprKind::Name(path) => self.value_path(path, value_scopes),
            ExprKind::Unary { operand, .. } => self.expr(operand, value_scopes, type_scopes),
            ExprKind::Binary { left, right, .. } => {
                self.expr(left, value_scopes, type_scopes);
                self.expr(right, value_scopes, type_scopes);
            }
            ExprKind::Call {
                callee,
                type_arguments,
                argument_groups,
            } => {
                self.expr(callee, value_scopes, type_scopes);
                for argument in type_arguments {
                    self.ty(argument, value_scopes, type_scopes);
                }
                for group in argument_groups {
                    for argument in group {
                        self.expr(argument, value_scopes, type_scopes);
                    }
                }
            }
            ExprKind::Field { receiver, .. } => self.expr(receiver, value_scopes, type_scopes),
            ExprKind::Index {
                receiver,
                coordinates,
            } => {
                self.expr(receiver, value_scopes, type_scopes);
                for coordinate in coordinates {
                    self.expr(coordinate, value_scopes, type_scopes);
                }
            }
            ExprKind::Slice {
                receiver,
                start,
                end,
            } => {
                self.expr(receiver, value_scopes, type_scopes);
                if let Some(start) = start {
                    self.expr(start, value_scopes, type_scopes);
                }
                if let Some(end) = end {
                    self.expr(end, value_scopes, type_scopes);
                }
            }
            ExprKind::StructInit { ty, fields } => {
                self.ty(ty, value_scopes, type_scopes);
                for (_, value) in fields {
                    self.expr(value, value_scopes, type_scopes);
                }
            }
            ExprKind::StaticCall { ty, arguments, .. } => {
                self.ty(ty, value_scopes, type_scopes);
                for argument in arguments {
                    self.expr(argument, value_scopes, type_scopes);
                }
            }
            ExprKind::Array(elements) => {
                for element in elements {
                    self.expr(element, value_scopes, type_scopes);
                }
            }
            ExprKind::Lambda(lambda) => {
                let mut locals = HashSet::new();
                for parameter in &mut lambda.parameters {
                    self.parameter_type(parameter, value_scopes, type_scopes);
                    match &parameter.kind {
                        ParameterKind::Named { name, .. } => {
                            locals.insert(name.clone());
                        }
                        ParameterKind::Receiver { .. } => {
                            locals.insert("self".to_string());
                        }
                    }
                }
                self.ty(&mut lambda.return_type, value_scopes, type_scopes);
                value_scopes.push(locals);
                self.block_contents(&mut lambda.body, value_scopes, type_scopes);
                value_scopes.pop();
            }
            ExprKind::Block(block) => self.block(block, value_scopes, type_scopes),
        }
    }

    fn ty(
        &self,
        ty: &mut TypeSyntax,
        value_scopes: &mut Vec<HashSet<String>>,
        type_scopes: &mut Vec<HashSet<String>>,
    ) {
        match &mut ty.kind {
            TypeSyntaxKind::Builtin(_) | TypeSyntaxKind::SelfType { .. } => {}
            TypeSyntaxKind::Named { path, arguments } => {
                self.type_path(path, type_scopes);
                for argument in arguments {
                    self.ty(argument, value_scopes, type_scopes);
                }
            }
            TypeSyntaxKind::Const(inner)
            | TypeSyntaxKind::Pointer(inner)
            | TypeSyntaxKind::Slice(inner) => self.ty(inner, value_scopes, type_scopes),
            TypeSyntaxKind::Array {
                element,
                dimensions,
            } => {
                self.ty(element, value_scopes, type_scopes);
                for dimension in dimensions {
                    self.expr(dimension, value_scopes, type_scopes);
                }
            }
            TypeSyntaxKind::Reference { target, .. } => self.ty(target, value_scopes, type_scopes),
            TypeSyntaxKind::Union(members) | TypeSyntaxKind::Intersection(members) => {
                for member in members {
                    self.ty(member, value_scopes, type_scopes);
                }
            }
            TypeSyntaxKind::Function { parameters, result } => {
                for parameter in parameters {
                    self.ty(parameter, value_scopes, type_scopes);
                }
                self.ty(result, value_scopes, type_scopes);
            }
        }
    }

    fn rename_value_declaration(&self, name: &mut String) {
        if let Some(replacement) = self.values.get(name) {
            *name = replacement.clone();
        }
    }

    fn rename_type_declaration(&self, name: &mut String) {
        if let Some(replacement) = self.types.get(name) {
            *name = replacement.clone();
        }
    }

    fn value_path(&self, path: &mut Path, scopes: &[HashSet<String>]) {
        let name = path.qualified_name();
        if !is_shadowed(&name, scopes) {
            if let Some(replacement) = self.values.get(&name) {
                *path = Path::from_qualified(replacement);
            }
        }
    }

    fn type_path(&self, path: &mut Path, scopes: &[HashSet<String>]) {
        let name = path.qualified_name();
        if !is_shadowed(&name, scopes) {
            if let Some(replacement) = self.types.get(&name) {
                *path = Path::from_qualified(replacement);
            }
        }
    }
}

#[derive(Clone, Copy)]
enum Namespace {
    Value,
    Type,
}

fn declaration_name(kind: &TopLevelKind) -> Option<(Namespace, &str)> {
    match kind {
        TopLevelKind::Function(declaration) => Some((Namespace::Value, &declaration.name)),
        TopLevelKind::Global(declaration) => Some((Namespace::Value, &declaration.name)),
        TopLevelKind::TypeAlias(declaration) => Some((Namespace::Type, &declaration.name)),
        TopLevelKind::Struct(declaration) => Some((Namespace::Type, &declaration.name)),
        TopLevelKind::Enum(declaration) => Some((Namespace::Type, &declaration.name)),
        TopLevelKind::Trait(declaration) => Some((Namespace::Type, &declaration.name)),
        TopLevelKind::Shape(declaration) => Some((Namespace::Type, &declaration.name)),
        _ => None,
    }
}

fn pattern_bindings(pattern: &Pattern) -> HashSet<String> {
    match pattern {
        Pattern::EnumVariant { bindings, .. } => bindings.iter().cloned().collect(),
        Pattern::Struct { fields, .. } => {
            fields.iter().map(|field| field.binding.clone()).collect()
        }
    }
}

fn is_shadowed(name: &str, scopes: &[HashSet<String>]) -> bool {
    scopes.iter().rev().any(|scope| scope.contains(name))
}

fn mangle(module_name: &str, name: &str) -> String {
    let module = module_name
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character
            } else {
                '_'
            }
        })
        .collect::<String>();
    format!("__{module}_{name}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use uuid::Uuid;

    fn write_file(path: &FsPath, contents: &str) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, contents).unwrap();
    }

    #[test]
    fn loads_imports_and_mangles_private_names_in_syntax() {
        let root = env::temp_dir().join(format!("skunk_syntax_modules_{}", Uuid::new_v4()));
        let entry = root.join("main.skunk");
        let module = root.join("mylib").join("math.skunk");
        write_file(
            &module,
            r#"
                module mylib.math;
                function helper(n: int): int { return n + 1; }
                export function inc(n: int): int { return helper(n); }
            "#,
        );
        write_file(
            &entry,
            r#"
                import mylib.math;
                function main(): void { print(inc(6)); }
            "#,
        );

        let loaded = load_program(&entry).unwrap();
        let names = loaded
            .module
            .entries
            .iter()
            .filter_map(|entry| match &entry.kind {
                TopLevelKind::Function(function) => Some(function.name.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert!(names.contains(&"inc"));
        assert!(names.contains(&"__mylib_math_helper"));
        assert!(names.contains(&"main"));
        assert_ne!(loaded.module.entries[0].id, loaded.module.entries[2].id);

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn rejects_module_name_mismatch() {
        let root = env::temp_dir().join(format!("skunk_syntax_modules_{}", Uuid::new_v4()));
        let entry = root.join("main.skunk");
        let module = root.join("mylib").join("math.skunk");
        write_file(
            &module,
            "module mylib.wrong; function inc(): int { return 1; }",
        );
        write_file(&entry, "import mylib.math; function main(): void {}");

        let error = load_program(&entry).unwrap_err();
        assert!(error.contains("module declaration mismatch"));
        fs::remove_dir_all(root).unwrap();
    }
}
