//! Direct Pest-to-syntax-AST parser.
//!
//! Pest's generated `Rule` values never escape this module. Every successful
//! parse produces source-oriented syntax nodes with stable IDs and byte spans.

use super::ast::*;
use super::pest::{Rule, SkunkParser};
use crate::diagnostic::Diagnostic;
use crate::ids::{FileId, NodeId};
use crate::source_map::{SourceMap, Span};
use pest::iterators::Pair;
use pest::pratt_parser::{Assoc, Op, PrattParser};
use pest::Parser as _;
use std::cell::Cell;

lazy_static::lazy_static! {
    static ref PRATT: PrattParser<Rule> = {
        use Assoc::*;
        use Rule::*;
        PrattParser::new()
            .op(Op::infix(eq, Left))
            .op(Op::infix(not_eq, Left))
            .op(Op::infix(lt, Left))
            .op(Op::infix(lte, Left))
            .op(Op::infix(gt, Left))
            .op(Op::infix(gte, Left))
            .op(Op::infix(add, Left) | Op::infix(subtract, Left))
            .op(Op::infix(multiply, Left) | Op::infix(divide, Left) | Op::infix(modulus, Left))
            .op(Op::infix(power, Right))
            .op(Op::infix(or, Left))
            .op(Op::infix(and, Left))
    };
}

pub fn parse_module(sources: &SourceMap, file: FileId) -> Result<Module, Vec<Diagnostic>> {
    let source = sources.source(file).ok_or_else(|| {
        vec![
            Diagnostic::error(format!("unknown source file id {}", file.index()))
                .with_code("E0002"),
        ]
    })?;
    let mut pairs = SkunkParser::parse(Rule::program, source).map_err(|error| {
        vec![Diagnostic::error(format!("parser failed: {error}")).with_code("E1000")]
    })?;
    let program = pairs.next().ok_or_else(|| {
        vec![Diagnostic::error("parser produced no program node").with_code("E1001")]
    })?;
    DirectParser::new(file, source.len())
        .module(program)
        .map_err(|diagnostic| vec![diagnostic])
}

#[cfg(test)]
pub(crate) fn parse_test_module(source: &str) -> Module {
    let mut sources = SourceMap::default();
    let file = sources.add_file("<test>", source).unwrap();
    let module = parse_module(&sources, file).unwrap();
    super::normalize::normalize(module).unwrap()
}

struct DirectParser {
    file: FileId,
    source_len: usize,
    next_id: Cell<u32>,
}

impl DirectParser {
    fn new(file: FileId, source_len: usize) -> Self {
        Self {
            file,
            source_len,
            next_id: Cell::new(0),
        }
    }

    fn id(&self) -> Result<NodeId, Diagnostic> {
        let value = self.next_id.get();
        self.next_id.set(value.checked_add(1).ok_or_else(|| {
            Diagnostic::error("source contains too many syntax nodes").with_code("E1002")
        })?);
        Ok(NodeId::in_file(self.file, value))
    }

    fn pair_span(&self, pair: &Pair<'_, Rule>) -> Result<Span, Diagnostic> {
        let pest_span = pair.as_span();
        Span::new(self.file, pest_span.start(), pest_span.end()).ok_or_else(|| {
            Diagnostic::error("source span is too large to represent").with_code("E0001")
        })
    }

    fn module(&self, pair: Pair<'_, Rule>) -> Result<Module, Diagnostic> {
        let span = Span::new(self.file, 0, self.source_len).ok_or_else(|| {
            Diagnostic::error("source file is too large to represent").with_code("E0001")
        })?;
        let mut name = None;
        let mut entries = Vec::new();
        for statement in pair.into_inner() {
            if statement.as_rule() == Rule::EOI {
                continue;
            }
            let statement = self.unwrap(statement, Rule::statement, "statement")?;
            match statement.as_rule() {
                Rule::module => {
                    let module_span = self.pair_span(&statement)?;
                    let qualified = statement.into_inner().next().ok_or_else(|| {
                        self.error(module_span, "module declaration is missing its name")
                    })?;
                    name = Some(self.path(qualified));
                }
                Rule::export_decl => {
                    let export_span = self.pair_span(&statement)?;
                    let declaration = statement.into_inner().next().ok_or_else(|| {
                        self.error(export_span, "export is missing a declaration")
                    })?;
                    entries.push(self.top_level(declaration, Visibility::Public)?);
                }
                other if self.is_top_level_rule(other) => {
                    entries.push(self.top_level(statement, Visibility::Private)?);
                }
                _ => entries.push(self.statement_top_level(statement)?),
            }
        }
        Ok(Module {
            id: self.id()?,
            span,
            name,
            entries,
        })
    }

    fn is_top_level_rule(&self, rule: Rule) -> bool {
        matches!(
            rule,
            Rule::import
                | Rule::type_alias_decl
                | Rule::struct_decl
                | Rule::enum_decl
                | Rule::trait_decl
                | Rule::shape_decl
                | Rule::attach_decl
                | Rule::conform_decl
                | Rule::func_decl
                | Rule::extern_func_decl
                | Rule::test_decl
                | Rule::var_decl_stmt
        )
    }

    fn top_level(
        &self,
        pair: Pair<'_, Rule>,
        visibility: Visibility,
    ) -> Result<TopLevel, Diagnostic> {
        let span = self.pair_span(&pair)?;
        let kind = match pair.as_rule() {
            Rule::import => {
                let module = pair
                    .into_inner()
                    .next()
                    .ok_or_else(|| self.error(span, "import declaration is missing its module"))?;
                TopLevelKind::Import(ImportDecl {
                    module: self.path(module),
                })
            }
            Rule::type_alias_decl => TopLevelKind::TypeAlias(self.type_alias(pair)?),
            Rule::struct_decl => TopLevelKind::Struct(self.struct_decl(pair)?),
            Rule::enum_decl => TopLevelKind::Enum(self.enum_decl(pair)?),
            Rule::trait_decl => TopLevelKind::Trait(self.trait_decl(pair)?),
            Rule::shape_decl => TopLevelKind::Shape(self.shape_decl(pair)?),
            Rule::attach_decl => TopLevelKind::Attach(self.attach_decl(pair)?),
            Rule::conform_decl => TopLevelKind::Conformance(self.conformance_decl(pair)?),
            Rule::func_decl => TopLevelKind::Function(self.function(pair)?),
            Rule::extern_func_decl => TopLevelKind::ExternFunction(self.extern_function(pair)?),
            Rule::test_decl => TopLevelKind::Test(self.test_decl(pair)?),
            Rule::var_decl_stmt => {
                let declaration = pair.into_inner().next().ok_or_else(|| {
                    self.error(span, "global declaration is missing its variable")
                })?;
                TopLevelKind::Global(self.local(declaration)?)
            }
            rule => {
                return Err(self.error(span, format!("{rule:?} is not a top-level declaration")))
            }
        };
        Ok(TopLevel {
            id: self.id()?,
            span,
            visibility,
            kind,
        })
    }

    fn statement_top_level(&self, pair: Pair<'_, Rule>) -> Result<TopLevel, Diagnostic> {
        let span = self.pair_span(&pair)?;
        Ok(TopLevel {
            id: self.id()?,
            span,
            visibility: Visibility::Private,
            kind: TopLevelKind::Statement(self.statement(pair)?),
        })
    }

    fn type_alias(&self, pair: Pair<'_, Rule>) -> Result<TypeAliasDecl, Diagnostic> {
        let span = self.pair_span(&pair)?;
        let mut inner = pair.into_inner().peekable();
        let name = self.identifier(
            inner
                .next()
                .ok_or_else(|| self.error(span, "type alias is missing its name"))?,
        )?;
        let mut parameters = if inner
            .peek()
            .is_some_and(|pair| pair.as_rule() == Rule::generic_params)
        {
            self.generic_parameters(self.required(
                inner.next(),
                span,
                "type alias generic parameters",
            )?)?
        } else {
            Vec::new()
        };
        let target = self.ty(inner
            .next()
            .ok_or_else(|| self.error(span, "type alias is missing its target type"))?)?;
        self.ensure_no_extra(inner.next(), span, "type alias")?;
        self.apply_where(&mut parameters, None)?;
        Ok(TypeAliasDecl {
            name,
            generic_parameters: parameters,
            target,
        })
    }

    fn struct_decl(&self, pair: Pair<'_, Rule>) -> Result<StructDecl, Diagnostic> {
        let span = self.pair_span(&pair)?;
        let mut inner = pair.into_inner().peekable();
        let name = self.identifier(
            inner
                .next()
                .ok_or_else(|| self.error(span, "struct is missing its name"))?,
        )?;
        let mut generic_parameters = self.optional_generics(&mut inner)?;
        if inner
            .peek()
            .is_some_and(|pair| pair.as_rule() == Rule::where_clause)
        {
            let clause = self.required(inner.next(), span, "struct where clause")?;
            self.apply_where(&mut generic_parameters, Some(clause))?;
        }
        let fields = inner
            .map(|field| self.struct_field(field))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(StructDecl {
            name,
            generic_parameters,
            fields,
            methods: Vec::new(),
        })
    }

    fn struct_field(&self, pair: Pair<'_, Rule>) -> Result<StructField, Diagnostic> {
        let span = self.pair_span(&pair)?;
        if pair.as_rule() != Rule::struct_field_decl {
            return Err(self.error(span, "expected struct field declaration"));
        }
        let mut inner = pair.into_inner().peekable();
        let is_const = inner
            .peek()
            .is_some_and(|pair| pair.as_rule() == Rule::const_kw);
        if is_const {
            inner.next();
        }
        let name = self.identifier(
            inner
                .next()
                .ok_or_else(|| self.error(span, "struct field is missing its name"))?,
        )?;
        let ty = self.ty(inner
            .next()
            .ok_or_else(|| self.error(span, "struct field is missing its type"))?)?;
        Ok(StructField {
            id: self.id()?,
            span,
            name,
            is_const,
            ty,
        })
    }

    fn enum_decl(&self, pair: Pair<'_, Rule>) -> Result<EnumDecl, Diagnostic> {
        let span = self.pair_span(&pair)?;
        let mut inner = pair.into_inner().peekable();
        let name = self.identifier(
            inner
                .next()
                .ok_or_else(|| self.error(span, "enum is missing its name"))?,
        )?;
        let mut generic_parameters = self.optional_generics(&mut inner)?;
        if inner
            .peek()
            .is_some_and(|pair| pair.as_rule() == Rule::where_clause)
        {
            let clause = self.required(inner.next(), span, "enum where clause")?;
            self.apply_where(&mut generic_parameters, Some(clause))?;
        }
        let variants = inner
            .map(|variant| self.enum_variant(variant))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(EnumDecl {
            name,
            generic_parameters,
            variants,
            methods: Vec::new(),
        })
    }

    fn enum_variant(&self, pair: Pair<'_, Rule>) -> Result<EnumVariant, Diagnostic> {
        let span = self.pair_span(&pair)?;
        let mut inner = pair.into_inner();
        let name = self.identifier(
            inner
                .next()
                .ok_or_else(|| self.error(span, "enum variant is missing its name"))?,
        )?;
        let payload = inner.map(|ty| self.ty(ty)).collect::<Result<Vec<_>, _>>()?;
        Ok(EnumVariant {
            id: self.id()?,
            span,
            name,
            payload,
        })
    }

    fn trait_decl(&self, pair: Pair<'_, Rule>) -> Result<TraitDecl, Diagnostic> {
        let span = self.pair_span(&pair)?;
        let mut inner = pair.into_inner().peekable();
        let name = self.identifier(
            inner
                .next()
                .ok_or_else(|| self.error(span, "trait is missing its name"))?,
        )?;
        let mut generic_parameters = self.optional_generics(&mut inner)?;
        let supertraits = if inner
            .peek()
            .is_some_and(|pair| pair.as_rule() == Rule::supertrait_bounds)
        {
            self.required(inner.next(), span, "supertrait bounds")?
                .into_inner()
                .map(|bound| Path::single(bound.as_str()))
                .collect()
        } else {
            Vec::new()
        };
        if inner
            .peek()
            .is_some_and(|pair| pair.as_rule() == Rule::where_clause)
        {
            let clause = self.required(inner.next(), span, "trait where clause")?;
            self.apply_where(&mut generic_parameters, Some(clause))?;
        }
        let methods = inner
            .map(|method| self.trait_method(method, true))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(TraitDecl {
            name,
            generic_parameters,
            supertraits,
            methods,
        })
    }

    fn shape_decl(&self, pair: Pair<'_, Rule>) -> Result<ShapeDecl, Diagnostic> {
        let span = self.pair_span(&pair)?;
        let mut inner = pair.into_inner();
        let name = self.identifier(
            inner
                .next()
                .ok_or_else(|| self.error(span, "shape is missing its name"))?,
        )?;
        let methods = inner
            .map(|method| self.trait_method(method, false))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(ShapeDecl { name, methods })
    }

    fn trait_method(
        &self,
        pair: Pair<'_, Rule>,
        allow_body: bool,
    ) -> Result<TraitMethod, Diagnostic> {
        let span = self.pair_span(&pair)?;
        let mut inner = pair.into_inner().peekable();
        let name = self.identifier(
            inner
                .next()
                .ok_or_else(|| self.error(span, "trait method is missing its name"))?,
        )?;
        let parameters = self.parameters(
            inner
                .next()
                .ok_or_else(|| self.error(span, "trait method is missing its parameter list"))?,
        )?;
        let return_type = if inner
            .peek()
            .is_some_and(|pair| pair.as_rule() == Rule::return_type)
        {
            self.return_type(self.required(inner.next(), span, "trait method return type")?)?
        } else {
            self.void_type(span)?
        };
        let default_body = if allow_body
            && inner
                .peek()
                .is_some_and(|pair| pair.as_rule() == Rule::trait_method_body)
        {
            Some(self.block_like(self.required(inner.next(), span, "trait method body")?)?)
        } else {
            None
        };
        Ok(TraitMethod {
            id: self.id()?,
            span,
            name,
            parameters,
            return_type,
            default_body,
        })
    }

    fn attach_decl(&self, pair: Pair<'_, Rule>) -> Result<AttachDecl, Diagnostic> {
        let span = self.pair_span(&pair)?;
        let mut inner = pair.into_inner().peekable();
        let mut generic_parameters = self.optional_generics(&mut inner)?;
        let target = self.ty(inner
            .next()
            .ok_or_else(|| self.error(span, "attach declaration is missing its target"))?)?;
        if inner
            .peek()
            .is_some_and(|pair| pair.as_rule() == Rule::where_clause)
        {
            let clause = self.required(inner.next(), span, "attach where clause")?;
            self.apply_where(&mut generic_parameters, Some(clause))?;
        }
        let methods = inner
            .map(|function| self.function(function))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(AttachDecl {
            generic_parameters,
            target,
            methods,
        })
    }

    fn conformance_decl(&self, pair: Pair<'_, Rule>) -> Result<ConformanceDecl, Diagnostic> {
        let span = self.pair_span(&pair)?;
        let mut inner = pair.into_inner().peekable();
        let mut generic_parameters = self.optional_generics(&mut inner)?;
        let trait_type = self.ty(inner
            .next()
            .ok_or_else(|| self.error(span, "conformance declaration is missing its trait"))?)?;
        let target = self.ty(inner
            .next()
            .ok_or_else(|| self.error(span, "conformance declaration is missing its target"))?)?;
        if inner
            .peek()
            .is_some_and(|pair| pair.as_rule() == Rule::where_clause)
        {
            let clause = self.required(inner.next(), span, "conformance where clause")?;
            self.apply_where(&mut generic_parameters, Some(clause))?;
        }
        let methods = inner
            .map(|function| self.function(function))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(ConformanceDecl {
            generic_parameters,
            traits: vec![trait_type],
            target,
            methods,
        })
    }

    fn function(&self, pair: Pair<'_, Rule>) -> Result<FunctionDecl, Diagnostic> {
        let span = self.pair_span(&pair)?;
        let is_lambda = pair.as_rule() == Rule::lambda_expr;
        let mut inner = pair.into_inner().peekable();
        let name = if is_lambda {
            "<lambda>".to_string()
        } else {
            self.identifier(
                inner
                    .next()
                    .ok_or_else(|| self.error(span, "function is missing its name"))?,
            )?
        };
        let mut generic_parameters = self.optional_generics(&mut inner)?;
        let parameters = self.parameters(
            inner
                .next()
                .ok_or_else(|| self.error(span, "function is missing its parameter list"))?,
        )?;
        let return_type = if inner
            .peek()
            .is_some_and(|pair| pair.as_rule() == Rule::return_type)
        {
            self.return_type(self.required(inner.next(), span, "function return type")?)?
        } else {
            self.void_type(span)?
        };
        if inner
            .peek()
            .is_some_and(|pair| pair.as_rule() == Rule::where_clause)
        {
            let clause = self.required(inner.next(), span, "function where clause")?;
            self.apply_where(&mut generic_parameters, Some(clause))?;
        }
        let body = self.block_from_statements(span, inner)?;
        Ok(FunctionDecl {
            name,
            generic_parameters,
            parameters,
            return_type,
            body,
        })
    }

    fn extern_function(&self, pair: Pair<'_, Rule>) -> Result<ExternFunctionDecl, Diagnostic> {
        let span = self.pair_span(&pair)?;
        let mut inner = pair.into_inner();
        let abi_pair = inner
            .next()
            .ok_or_else(|| self.error(span, "extern function is missing its ABI"))?;
        let abi = parse_string(abi_pair.as_str()).map_err(|message| self.error(span, message))?;
        let name = self.identifier(
            inner
                .next()
                .ok_or_else(|| self.error(span, "extern function is missing its name"))?,
        )?;
        let parameters =
            self.parameters(inner.next().ok_or_else(|| {
                self.error(span, "extern function is missing its parameter list")
            })?)?;
        let return_type = if let Some(return_type) = inner.next() {
            self.return_type(return_type)?
        } else {
            self.void_type(span)?
        };
        Ok(ExternFunctionDecl {
            abi,
            name,
            parameters,
            return_type,
        })
    }

    fn test_decl(&self, pair: Pair<'_, Rule>) -> Result<TestDecl, Diagnostic> {
        let span = self.pair_span(&pair)?;
        let mut inner = pair.into_inner();
        let name_pair = inner
            .next()
            .ok_or_else(|| self.error(span, "test declaration is missing its name"))?;
        let name = parse_string(name_pair.as_str()).map_err(|message| self.error(span, message))?;
        Ok(TestDecl {
            name,
            body: self.block_from_statements(span, inner)?,
        })
    }

    fn optional_generics<'a>(
        &self,
        inner: &mut std::iter::Peekable<pest::iterators::Pairs<'a, Rule>>,
    ) -> Result<Vec<GenericParameter>, Diagnostic> {
        let Some(pair) = inner
            .peek()
            .filter(|pair| pair.as_rule() == Rule::generic_params)
        else {
            return Ok(Vec::new());
        };
        let span = self.pair_span(pair)?;
        self.generic_parameters(self.required(inner.next(), span, "generic parameters")?)
    }

    fn generic_parameters(
        &self,
        pair: Pair<'_, Rule>,
    ) -> Result<Vec<GenericParameter>, Diagnostic> {
        pair.into_inner()
            .map(|parameter| {
                let span = self.pair_span(&parameter)?;
                let mut inner = parameter.into_inner();
                let name =
                    self.identifier(inner.next().ok_or_else(|| {
                        self.error(span, "generic parameter is missing its name")
                    })?)?;
                let mut generic = GenericParameter {
                    id: self.id()?,
                    span,
                    name,
                    capabilities: Vec::new(),
                    lower_bound: None,
                    upper_bound: None,
                };
                for bound in inner {
                    self.apply_generic_bound(&mut generic, bound)?;
                }
                Ok(generic)
            })
            .collect()
    }

    fn apply_where(
        &self,
        parameters: &mut [GenericParameter],
        clause: Option<Pair<'_, Rule>>,
    ) -> Result<(), Diagnostic> {
        let Some(clause) = clause else {
            return Ok(());
        };
        for predicate in clause.into_inner() {
            let span = self.pair_span(&predicate)?;
            let mut inner = predicate.into_inner();
            let name =
                self.identifier(inner.next().ok_or_else(|| {
                    self.error(span, "where predicate is missing its parameter")
                })?)?;
            let bound = inner
                .next()
                .ok_or_else(|| self.error(span, "where predicate is missing its bound"))?;
            let parameter = parameters
                .iter_mut()
                .find(|parameter| parameter.name == name)
                .ok_or_else(|| {
                    self.error(span, format!("where predicate references unknown `{name}`"))
                        .with_code("E1010")
                })?;
            self.apply_generic_bound(parameter, bound)?;
        }
        Ok(())
    }

    fn apply_generic_bound(
        &self,
        parameter: &mut GenericParameter,
        pair: Pair<'_, Rule>,
    ) -> Result<(), Diagnostic> {
        match pair.as_rule() {
            Rule::capability_bounds => {
                for capability in pair.into_inner().map(|bound| Path::single(bound.as_str())) {
                    if !parameter.capabilities.contains(&capability) {
                        parameter.capabilities.push(capability);
                    }
                }
            }
            Rule::subtype_bounds => {
                for bound in pair.into_inner() {
                    let span = self.pair_span(&bound)?;
                    let rule = bound.as_rule();
                    let ty = self
                        .ty(bound.into_inner().next().ok_or_else(|| {
                            self.error(span, "subtype bound is missing its type")
                        })?)?;
                    match rule {
                        Rule::lower_type_bound => parameter.lower_bound = Some(ty),
                        Rule::upper_type_bound => parameter.upper_bound = Some(ty),
                        _ => return Err(self.error(span, "unexpected subtype bound")),
                    }
                }
            }
            rule => {
                return Err(self.error(
                    self.pair_span(&pair)?,
                    format!("unexpected generic bound {rule:?}"),
                ))
            }
        }
        Ok(())
    }

    fn parameters(&self, pair: Pair<'_, Rule>) -> Result<Vec<Parameter>, Diagnostic> {
        let span = self.pair_span(&pair)?;
        match pair.as_rule() {
            Rule::member_func_params | Rule::static_func_params => {
                let mut parameters = Vec::new();
                for part in pair.into_inner() {
                    match part.as_rule() {
                        Rule::_self => parameters.push(self.receiver_parameter(part)?),
                        Rule::param_list => {
                            parameters.extend(self.named_parameters(part)?);
                        }
                        Rule::empty_params => {}
                        rule => {
                            return Err(self.error(
                                span,
                                format!("unexpected parameter-list component {rule:?}"),
                            ))
                        }
                    }
                }
                Ok(parameters)
            }
            _ => Err(self.error(span, "expected function parameter list")),
        }
    }

    fn receiver_parameter(&self, pair: Pair<'_, Rule>) -> Result<Parameter, Diagnostic> {
        let span = self.pair_span(&pair)?;
        let qualifier = pair.into_inner().next().map(|pair| pair.as_rule());
        Ok(Parameter {
            id: self.id()?,
            span,
            kind: ParameterKind::Receiver {
                mutable: qualifier == Some(Rule::mut_kw),
                is_const: qualifier == Some(Rule::const_kw),
            },
        })
    }

    fn named_parameters(&self, pair: Pair<'_, Rule>) -> Result<Vec<Parameter>, Diagnostic> {
        pair.into_inner()
            .map(|parameter| {
                let span = self.pair_span(&parameter)?;
                let mut inner = parameter.into_inner().peekable();
                let is_const = inner
                    .peek()
                    .is_some_and(|pair| pair.as_rule() == Rule::const_kw);
                if is_const {
                    inner.next();
                }
                let name = self.identifier(
                    inner
                        .next()
                        .ok_or_else(|| self.error(span, "parameter is missing its name"))?,
                )?;
                let ty = self.ty(inner
                    .next()
                    .ok_or_else(|| self.error(span, "parameter is missing its type"))?)?;
                Ok(Parameter {
                    id: self.id()?,
                    span,
                    kind: ParameterKind::Named { name, is_const, ty },
                })
            })
            .collect()
    }

    fn return_type(&self, pair: Pair<'_, Rule>) -> Result<TypeSyntax, Diagnostic> {
        let span = self.pair_span(&pair)?;
        self.ty(pair
            .into_inner()
            .next()
            .ok_or_else(|| self.error(span, "return type is missing its type"))?)
    }

    fn void_type(&self, span: Span) -> Result<TypeSyntax, Diagnostic> {
        Ok(TypeSyntax {
            id: self.id()?,
            span,
            kind: TypeSyntaxKind::Builtin(BuiltinType::Void),
        })
    }

    fn ty(&self, pair: Pair<'_, Rule>) -> Result<TypeSyntax, Diagnostic> {
        let span = self.pair_span(&pair)?;
        let kind = match pair.as_rule() {
            Rule::_type | Rule::base_type => {
                let inner = pair
                    .into_inner()
                    .next()
                    .ok_or_else(|| self.error(span, "empty type expression"))?;
                return self.ty(inner);
            }
            Rule::builtin_type => TypeSyntaxKind::Builtin(match pair.as_str() {
                "void" => BuiltinType::Void,
                "byte" => BuiltinType::Byte,
                "short" => BuiltinType::Short,
                "int" => BuiltinType::Int,
                "long" => BuiltinType::Long,
                "float" => BuiltinType::Float,
                "double" => BuiltinType::Double,
                "string" => BuiltinType::String,
                "boolean" | "bool" => BuiltinType::Boolean,
                "char" => BuiltinType::Char,
                "Allocator" => BuiltinType::Allocator,
                "Arena" => BuiltinType::Arena,
                name => return Err(self.error(span, format!("unknown builtin type `{name}`"))),
            }),
            Rule::nominal_type => {
                let mut inner = pair.into_inner();
                let name = inner
                    .next()
                    .ok_or_else(|| self.error(span, "nominal type is missing its name"))?;
                TypeSyntaxKind::Named {
                    path: Path::single(name.as_str()),
                    arguments: inner
                        .map(|argument| self.ty(argument))
                        .collect::<Result<_, _>>()?,
                }
            }
            Rule::function_type => {
                let mut inner = pair.into_inner().peekable();
                let parameters = if inner
                    .peek()
                    .is_some_and(|pair| pair.as_rule() == Rule::param_type_list)
                {
                    self.required(inner.next(), span, "function parameter type list")?
                        .into_inner()
                        .map(|parameter| self.ty(parameter))
                        .collect::<Result<_, _>>()?
                } else {
                    Vec::new()
                };
                let result = self.ty(inner
                    .next()
                    .ok_or_else(|| self.error(span, "function type is missing its result"))?)?;
                TypeSyntaxKind::Function {
                    parameters,
                    result: Box::new(result),
                }
            }
            Rule::prefixed_type => return self.prefixed_type(pair),
            Rule::legacy_array_type => {
                let mut inner = pair.into_inner();
                let element = self.ty(inner
                    .next()
                    .ok_or_else(|| self.error(span, "array type is missing its element type"))?)?;
                let dimensions = inner
                    .map(|dimension| {
                        let dimension_span = self.pair_span(&dimension)?;
                        self.expression(dimension.into_inner().next().ok_or_else(|| {
                            self.error(dimension_span, "array dimension is empty")
                        })?)
                    })
                    .collect::<Result<_, _>>()?;
                TypeSyntaxKind::Array {
                    element: Box::new(element),
                    dimensions,
                }
            }
            Rule::legacy_slice_type => {
                let source = pair.as_str().to_string();
                let element = self
                    .ty(pair.into_inner().next().ok_or_else(|| {
                        self.error(span, "slice type is missing its element type")
                    })?)?;
                let mut result = element;
                for _ in 0..source.matches("[]").count() {
                    result = TypeSyntax {
                        id: self.id()?,
                        span,
                        kind: TypeSyntaxKind::Slice(Box::new(result)),
                    };
                }
                return Ok(result);
            }
            Rule::union_type => TypeSyntaxKind::Union(
                pair.into_inner()
                    .map(|member| self.ty(member))
                    .collect::<Result<_, _>>()?,
            ),
            Rule::intersection_type => TypeSyntaxKind::Intersection(
                pair.into_inner()
                    .map(|member| self.ty(member))
                    .collect::<Result<_, _>>()?,
            ),
            rule => return Err(self.error(span, format!("unsupported type syntax {rule:?}"))),
        };
        Ok(TypeSyntax {
            id: self.id()?,
            span,
            kind,
        })
    }

    fn prefixed_type(&self, pair: Pair<'_, Rule>) -> Result<TypeSyntax, Diagnostic> {
        let span = self.pair_span(&pair)?;
        let mut parts = pair.into_inner().collect::<Vec<_>>();
        let base_pair = parts
            .pop()
            .ok_or_else(|| self.error(span, "prefixed type is missing its base type"))?;
        let base_is_const = parts
            .last()
            .is_some_and(|part| part.as_rule() == Rule::const_kw);
        if base_is_const {
            parts.pop();
        }
        let mut current = self.ty(base_pair)?;
        if base_is_const {
            current = TypeSyntax {
                id: self.id()?,
                span,
                kind: TypeSyntaxKind::Const(Box::new(current)),
            };
        }
        for prefix in parts.into_iter().rev() {
            let prefix_span = self.pair_span(&prefix)?;
            let inner = prefix.into_inner().next();
            current = match inner {
                Some(part) if part.as_rule() == Rule::pointer_prefix => TypeSyntax {
                    id: self.id()?,
                    span: prefix_span,
                    kind: TypeSyntaxKind::Pointer(Box::new(current)),
                },
                Some(part) if part.as_rule() == Rule::reference_prefix => {
                    let mutable = part.into_inner().next().is_some();
                    TypeSyntax {
                        id: self.id()?,
                        span: prefix_span,
                        kind: TypeSyntaxKind::Reference {
                            target: Box::new(current),
                            mutable,
                        },
                    }
                }
                Some(part) if part.as_rule() == Rule::expression => {
                    let dimension = self.expression(part)?;
                    match current {
                        TypeSyntax {
                            id,
                            kind:
                                TypeSyntaxKind::Array {
                                    element,
                                    mut dimensions,
                                },
                            ..
                        } => {
                            dimensions.insert(0, dimension);
                            TypeSyntax {
                                id,
                                span: prefix_span,
                                kind: TypeSyntaxKind::Array {
                                    element,
                                    dimensions,
                                },
                            }
                        }
                        element => TypeSyntax {
                            id: self.id()?,
                            span: prefix_span,
                            kind: TypeSyntaxKind::Array {
                                element: Box::new(element),
                                dimensions: vec![dimension],
                            },
                        },
                    }
                }
                None => TypeSyntax {
                    id: self.id()?,
                    span: prefix_span,
                    kind: TypeSyntaxKind::Slice(Box::new(current)),
                },
                Some(part) => {
                    return Err(self.error(
                        prefix_span,
                        format!("unsupported type prefix {:?}", part.as_rule()),
                    ))
                }
            };
        }
        Ok(current)
    }

    fn block(&self, pair: Pair<'_, Rule>) -> Result<Block, Diagnostic> {
        let span = self.pair_span(&pair)?;
        self.block_from_statements(span, pair.into_inner())
    }

    fn block_like(&self, pair: Pair<'_, Rule>) -> Result<Block, Diagnostic> {
        let span = self.pair_span(&pair)?;
        self.block_from_statements(span, pair.into_inner())
    }

    fn block_from_statements<'a>(
        &self,
        span: Span,
        statements: impl Iterator<Item = Pair<'a, Rule>>,
    ) -> Result<Block, Diagnostic> {
        Ok(Block {
            id: self.id()?,
            span,
            statements: statements
                .map(|statement| self.statement(statement))
                .collect::<Result<_, _>>()?,
        })
    }

    fn statement(&self, pair: Pair<'_, Rule>) -> Result<Stmt, Diagnostic> {
        let pair = if pair.as_rule() == Rule::statement {
            self.unwrap(pair, Rule::statement, "statement")?
        } else {
            pair
        };
        let span = self.pair_span(&pair)?;
        let kind =
            match pair.as_rule() {
                Rule::var_decl_stmt => {
                    let declaration = pair.into_inner().next().ok_or_else(|| {
                        self.error(span, "local declaration is missing its variable")
                    })?;
                    StmtKind::Local(self.local(declaration)?)
                }
                Rule::var_decl => StmtKind::Local(self.local(pair)?),
                Rule::struct_destructure_stmt => {
                    let destructure = pair.into_inner().next().ok_or_else(|| {
                        self.error(span, "destructure statement is missing its pattern")
                    })?;
                    StmtKind::StructDestructure(self.struct_destructure(destructure)?)
                }
                Rule::assignment => {
                    let mut inner = pair.into_inner();
                    let target =
                        self.expression(inner.next().ok_or_else(|| {
                            self.error(span, "assignment is missing its target")
                        })?)?;
                    let value = self.expression(
                        inner
                            .next()
                            .ok_or_else(|| self.error(span, "assignment is missing its value"))?,
                    )?;
                    StmtKind::Assignment { target, value }
                }
                Rule::block => StmtKind::Block(self.block(pair)?),
                Rule::unsafe_block => StmtKind::Unsafe(self.block_like(pair)?),
                Rule::control_flow => {
                    let control = pair
                        .into_inner()
                        .next()
                        .ok_or_else(|| self.error(span, "control-flow statement is empty"))?;
                    match control.as_rule() {
                        Rule::if_expr => StmtKind::If(self.if_expr(control)?),
                        Rule::match_expr => StmtKind::Match(self.match_expr(control)?),
                        Rule::for_expr => StmtKind::For(self.for_stmt(control)?),
                        rule => {
                            return Err(
                                self.error(span, format!("unsupported control flow {rule:?}"))
                            )
                        }
                    }
                }
                Rule::sk_return => {
                    let value = pair
                        .into_inner()
                        .next()
                        .map(|value| self.expression(value))
                        .transpose()?;
                    StmtKind::Return(value)
                }
                Rule::defer_stmt => {
                    let value = pair.into_inner().next().ok_or_else(|| {
                        self.error(span, "defer statement is missing its expression")
                    })?;
                    StmtKind::Defer(self.expression(value)?)
                }
                Rule::io => {
                    let io = pair
                        .into_inner()
                        .next()
                        .ok_or_else(|| self.error(span, "I/O statement is empty"))?;
                    match io.as_rule() {
                        Rule::print => {
                            let value = io.into_inner().next().ok_or_else(|| {
                                self.error(span, "print is missing its expression")
                            })?;
                            StmtKind::Print(self.expression(value)?)
                        }
                        Rule::input => StmtKind::Input,
                        rule => return Err(self.error(span, format!("unsupported I/O {rule:?}"))),
                    }
                }
                rule if self.is_top_level_rule(rule) => {
                    StmtKind::Declaration(Box::new(self.top_level(pair, Visibility::Private)?))
                }
                _ => StmtKind::Expression(self.expression(pair)?),
            };
        Ok(Stmt {
            id: self.id()?,
            span,
            kind,
        })
    }

    fn local(&self, pair: Pair<'_, Rule>) -> Result<LocalDecl, Diagnostic> {
        let span = self.pair_span(&pair)?;
        let mut inner = pair.into_inner().peekable();
        let is_const = inner
            .peek()
            .is_some_and(|pair| pair.as_rule() == Rule::const_kw);
        if is_const {
            inner.next();
        }
        let name = self.identifier(
            inner
                .next()
                .ok_or_else(|| self.error(span, "variable is missing its name"))?,
        )?;
        let ty = self.ty(inner
            .next()
            .ok_or_else(|| self.error(span, "variable is missing its type"))?)?;
        let initializer = inner
            .next()
            .map(|value| self.expression(value))
            .transpose()?;
        Ok(LocalDecl {
            name,
            is_const,
            ty,
            initializer,
        })
    }

    fn struct_destructure(&self, pair: Pair<'_, Rule>) -> Result<StructDestructure, Diagnostic> {
        let span = self.pair_span(&pair)?;
        let mut inner = pair.into_inner();
        let ty = self.ty(inner
            .next()
            .ok_or_else(|| self.error(span, "struct destructure is missing its type"))?)?;
        let fields = self.pattern_fields(
            inner
                .next()
                .ok_or_else(|| self.error(span, "struct destructure is missing its fields"))?,
        )?;
        let value = self.expression(
            inner
                .next()
                .ok_or_else(|| self.error(span, "struct destructure is missing its value"))?,
        )?;
        Ok(StructDestructure { ty, fields, value })
    }

    fn if_expr(&self, pair: Pair<'_, Rule>) -> Result<IfExpr, Diagnostic> {
        let span = self.pair_span(&pair)?;
        let mut inner = pair.into_inner().peekable();
        let condition = self.expression(
            inner
                .next()
                .ok_or_else(|| self.error(span, "if is missing its condition"))?,
        )?;
        let mut body_statements = Vec::new();
        while inner
            .peek()
            .is_some_and(|pair| pair.as_rule() == Rule::statement)
        {
            body_statements.push(self.required(inner.next(), span, "if body statement")?);
        }
        let then_block = self.block_from_statements(span, body_statements.into_iter())?;
        let mut else_if = Vec::new();
        while inner
            .peek()
            .is_some_and(|pair| pair.as_rule() == Rule::else_if_expr)
        {
            let branch = self.required(inner.next(), span, "else-if branch")?;
            let branch_span = self.pair_span(&branch)?;
            let mut branch_inner = branch.into_inner();
            let branch_condition = self.expression(
                branch_inner
                    .next()
                    .ok_or_else(|| self.error(branch_span, "else-if is missing its condition"))?,
            )?;
            let branch_body = self.block_from_statements(branch_span, branch_inner)?;
            else_if.push((branch_condition, branch_body));
        }
        let else_block = inner
            .next()
            .map(|branch| self.block_like(branch))
            .transpose()?;
        Ok(IfExpr {
            condition,
            then_block,
            else_if,
            else_block,
        })
    }

    fn match_expr(&self, pair: Pair<'_, Rule>) -> Result<MatchExpr, Diagnostic> {
        let span = self.pair_span(&pair)?;
        let mut inner = pair.into_inner();
        let value = self.expression(
            inner
                .next()
                .ok_or_else(|| self.error(span, "match is missing its value"))?,
        )?;
        let cases = inner
            .map(|case| self.match_case(case))
            .collect::<Result<_, _>>()?;
        Ok(MatchExpr { value, cases })
    }

    fn match_case(&self, pair: Pair<'_, Rule>) -> Result<MatchCase, Diagnostic> {
        let span = self.pair_span(&pair)?;
        let mut inner = pair.into_inner();
        let pattern = self.pattern(
            inner
                .next()
                .ok_or_else(|| self.error(span, "match case is missing its pattern"))?,
        )?;
        Ok(MatchCase {
            id: self.id()?,
            span,
            pattern,
            body: self.block_from_statements(span, inner)?,
        })
    }

    fn pattern(&self, pair: Pair<'_, Rule>) -> Result<Pattern, Diagnostic> {
        let span = self.pair_span(&pair)?;
        let pattern = self.unwrap(pair, Rule::match_pattern, "match pattern")?;
        match pattern.as_rule() {
            Rule::enum_match_pattern => {
                let mut inner = pattern.into_inner().peekable();
                let enum_type = if inner
                    .peek()
                    .is_some_and(|pair| pair.as_rule() == Rule::nominal_type)
                {
                    Some(self.ty(self.required(inner.next(), span, "enum pattern type")?)?)
                } else {
                    None
                };
                let variant = self.identifier(
                    inner
                        .next()
                        .ok_or_else(|| self.error(span, "enum pattern is missing its variant"))?,
                )?;
                let bindings = inner
                    .map(|binding| self.identifier(binding))
                    .collect::<Result<_, _>>()?;
                Ok(Pattern::EnumVariant {
                    enum_type,
                    variant,
                    bindings,
                })
            }
            Rule::struct_match_pattern => {
                let mut inner = pattern.into_inner();
                let ty = self.ty(inner
                    .next()
                    .ok_or_else(|| self.error(span, "struct pattern is missing its type"))?)?;
                let fields =
                    self.pattern_fields(inner.next().ok_or_else(|| {
                        self.error(span, "struct pattern is missing its fields")
                    })?)?;
                Ok(Pattern::Struct { ty, fields })
            }
            rule => Err(self.error(span, format!("unsupported match pattern {rule:?}"))),
        }
    }

    fn pattern_fields(&self, pair: Pair<'_, Rule>) -> Result<Vec<PatternField>, Diagnostic> {
        pair.into_inner()
            .map(|field| {
                let span = self.pair_span(&field)?;
                let mut inner = field.into_inner();
                let name = self.identifier(
                    inner
                        .next()
                        .ok_or_else(|| self.error(span, "pattern field is missing its name"))?,
                )?;
                let binding = inner
                    .next()
                    .map(|binding| self.identifier(binding))
                    .transpose()?
                    .unwrap_or_else(|| name.clone());
                Ok(PatternField { name, binding })
            })
            .collect()
    }

    fn for_stmt(&self, pair: Pair<'_, Rule>) -> Result<ForStmt, Diagnostic> {
        let span = self.pair_span(&pair)?;
        let mut inner = pair.into_inner();
        let header = inner
            .next()
            .ok_or_else(|| self.error(span, "for loop is missing its header"))?;
        let (initializer, condition, update) = match header.as_rule() {
            Rule::for_classic => {
                let mut parts = header.into_inner();
                let initializer = self.optional_for_statement(parts.next(), span, "initializer")?;
                let condition = self.optional_for_expression(parts.next(), span, "condition")?;
                let update = self.optional_for_statement(parts.next(), span, "update")?;
                (initializer.map(Box::new), condition, update.map(Box::new))
            }
            Rule::for_infinite => {
                let condition = header
                    .into_inner()
                    .next()
                    .map(|condition| self.expression(condition))
                    .transpose()?;
                (None, condition, None)
            }
            Rule::for_in => {
                return Err(self
                    .error(span, "for-in loops are not implemented")
                    .with_code("E1011"))
            }
            rule => return Err(self.error(span, format!("unsupported for header {rule:?}"))),
        };
        Ok(ForStmt {
            initializer,
            condition,
            update,
            body: self.block_from_statements(span, inner)?,
        })
    }

    fn optional_for_statement(
        &self,
        pair: Option<Pair<'_, Rule>>,
        span: Span,
        part: &str,
    ) -> Result<Option<Stmt>, Diagnostic> {
        let pair =
            pair.ok_or_else(|| self.error(span, format!("for loop is missing its {part}")))?;
        let Some(inner) = pair.into_inner().next() else {
            return Ok(None);
        };
        self.statement(inner).map(Some)
    }

    fn optional_for_expression(
        &self,
        pair: Option<Pair<'_, Rule>>,
        span: Span,
        part: &str,
    ) -> Result<Option<Expr>, Diagnostic> {
        let pair =
            pair.ok_or_else(|| self.error(span, format!("for loop is missing its {part}")))?;
        let Some(inner) = pair.into_inner().next() else {
            return Ok(None);
        };
        self.expression(inner).map(Some)
    }

    fn expression(&self, pair: Pair<'_, Rule>) -> Result<Expr, Diagnostic> {
        let span = self.pair_span(&pair)?;
        match pair.as_rule() {
            Rule::expression => PRATT
                .map_primary(|primary| self.primary(primary))
                .map_infix(|left, operator, right| {
                    Ok(Expr {
                        id: self.id()?,
                        span: Span::new(
                            self.file,
                            left.as_ref().map_or(span.start, |expr| expr.span.start) as usize,
                            right.as_ref().map_or(span.end, |expr| expr.span.end) as usize,
                        )
                        .unwrap_or(span),
                        kind: ExprKind::Binary {
                            left: Box::new(left?),
                            operator: self.binary_operator(operator)?,
                            right: Box::new(right?),
                        },
                    })
                })
                .parse(pair.into_inner()),
            Rule::primary => self.primary(pair),
            Rule::lambda_expr => self.lambda(pair),
            Rule::literal => self.literal(pair),
            Rule::static_func_call => self.static_call(pair),
            Rule::func_call => self.function_call(pair),
            Rule::struct_init => self.struct_init(pair),
            Rule::inline_array_init => self.array_literal(pair),
            Rule::access => self.access(pair),
            Rule::block => Ok(Expr {
                id: self.id()?,
                span,
                kind: ExprKind::Block(self.block(pair)?),
            }),
            Rule::IDENTIFIER => self.name_expr(pair.as_str(), span),
            rule => Err(self.error(span, format!("expected expression, found {rule:?}"))),
        }
    }

    fn primary(&self, pair: Pair<'_, Rule>) -> Result<Expr, Diagnostic> {
        let span = self.pair_span(&pair)?;
        let inner = if pair.as_rule() == Rule::primary {
            pair.into_inner()
                .next()
                .ok_or_else(|| self.error(span, "empty primary expression"))?
        } else {
            pair
        };
        if inner.as_rule() == Rule::unary_op {
            return self.unary(inner);
        }
        self.expression(inner)
    }

    fn unary(&self, pair: Pair<'_, Rule>) -> Result<Expr, Diagnostic> {
        let span = self.pair_span(&pair)?;
        let mut inner = pair.into_inner();
        let operator = inner
            .next()
            .ok_or_else(|| self.error(span, "unary expression is missing its operator"))?;
        let operator = match operator.as_rule() {
            Rule::unary_plus => UnaryOperator::Plus,
            Rule::unary_minus => UnaryOperator::Minus,
            Rule::negate => UnaryOperator::Not,
            Rule::address_of => UnaryOperator::AddressOf,
            Rule::address_of_mut => UnaryOperator::AddressOfMut,
            rule => return Err(self.error(span, format!("unsupported unary operator {rule:?}"))),
        };
        let operand = self.primary(
            inner
                .next()
                .ok_or_else(|| self.error(span, "unary expression is missing its operand"))?,
        )?;
        Ok(Expr {
            id: self.id()?,
            span,
            kind: ExprKind::Unary {
                operator,
                operand: Box::new(operand),
            },
        })
    }

    fn literal(&self, pair: Pair<'_, Rule>) -> Result<Expr, Diagnostic> {
        let span = self.pair_span(&pair)?;
        let token = if matches!(pair.as_rule(), Rule::literal | Rule::size) {
            pair.into_inner()
                .next()
                .ok_or_else(|| self.error(span, "empty literal"))?
        } else {
            pair
        };
        let text = token.as_str();
        let literal = match token.as_rule() {
            Rule::INTEGER => Literal::Integer(text.parse::<i64>().map_err(|_| {
                self.error(
                    span,
                    format!("integer literal `{text}` is outside the supported 64-bit range"),
                )
            })?),
            Rule::LONG_LITERAL => {
                Literal::Long(text[..text.len() - 1].parse::<i64>().map_err(|_| {
                    self.error(
                        span,
                        format!("long literal `{text}` is outside the supported 64-bit range"),
                    )
                })?)
            }
            Rule::FLOAT_LITERAL => {
                let value = text[..text.len() - 1]
                    .parse::<f32>()
                    .map_err(|_| self.error(span, format!("invalid float literal `{text}`")))?;
                if !value.is_finite() {
                    return Err(self.error(
                        span,
                        format!("float literal `{text}` is outside the supported range"),
                    ));
                }
                Literal::Float(value)
            }
            Rule::DOUBLE_LITERAL => {
                let value = text
                    .parse::<f64>()
                    .map_err(|_| self.error(span, format!("invalid double literal `{text}`")))?;
                if !value.is_finite() {
                    return Err(self.error(
                        span,
                        format!("double literal `{text}` is outside the supported range"),
                    ));
                }
                Literal::Double(value)
            }
            Rule::STRING_LITERAL => Literal::String(text.to_string()),
            Rule::BOOLEAN_LITERAL => Literal::Boolean(text == "true"),
            Rule::CHAR_LITERAL => {
                Literal::Char(parse_char(text).map_err(|message| self.error(span, message))?)
            }
            rule => return Err(self.error(span, format!("unsupported literal {rule:?}"))),
        };
        Ok(Expr {
            id: self.id()?,
            span,
            kind: ExprKind::Literal(literal),
        })
    }

    fn function_call(&self, pair: Pair<'_, Rule>) -> Result<Expr, Diagnostic> {
        let span = self.pair_span(&pair)?;
        let mut inner = pair.into_inner().peekable();
        let name = self.identifier(
            inner
                .next()
                .ok_or_else(|| self.error(span, "function call is missing its callee"))?,
        )?;
        let type_arguments = if inner
            .peek()
            .is_some_and(|pair| pair.as_rule() == Rule::type_arg_list)
        {
            self.required(inner.next(), span, "function-call type arguments")?
                .into_inner()
                .map(|argument| self.ty(argument))
                .collect::<Result<_, _>>()?
        } else {
            Vec::new()
        };
        let argument_groups = inner
            .map(|arguments| {
                arguments
                    .into_inner()
                    .map(|argument| self.expression(argument))
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<_, _>>()?;
        Ok(Expr {
            id: self.id()?,
            span,
            kind: ExprKind::Call {
                callee: Box::new(self.name_expr(&name, span)?),
                type_arguments,
                argument_groups,
            },
        })
    }

    fn static_call(&self, pair: Pair<'_, Rule>) -> Result<Expr, Diagnostic> {
        let span = self.pair_span(&pair)?;
        let mut inner = pair.into_inner();
        let ty = self.ty(inner
            .next()
            .ok_or_else(|| self.error(span, "static call is missing its owner type"))?)?;
        let name = self.identifier(
            inner
                .next()
                .ok_or_else(|| self.error(span, "static call is missing its method"))?,
        )?;
        let arguments = inner
            .map(|argument| self.expression(argument))
            .collect::<Result<_, _>>()?;
        Ok(Expr {
            id: self.id()?,
            span,
            kind: ExprKind::StaticCall {
                ty,
                name,
                arguments,
            },
        })
    }

    fn struct_init(&self, pair: Pair<'_, Rule>) -> Result<Expr, Diagnostic> {
        let span = self.pair_span(&pair)?;
        let mut inner = pair.into_inner();
        let ty = self.ty(inner
            .next()
            .ok_or_else(|| self.error(span, "struct literal is missing its type"))?)?;
        let fields = if let Some(fields) = inner.next() {
            fields
                .into_inner()
                .map(|field| {
                    let field_span = self.pair_span(&field)?;
                    let mut field_inner = field.into_inner();
                    let name = self.identifier(field_inner.next().ok_or_else(|| {
                        self.error(field_span, "struct literal field is missing its name")
                    })?)?;
                    let value = field_inner
                        .next()
                        .map(|value| self.expression(value))
                        .transpose()?
                        .unwrap_or(self.name_expr(&name, field_span)?);
                    Ok((name, value))
                })
                .collect::<Result<_, Diagnostic>>()?
        } else {
            Vec::new()
        };
        Ok(Expr {
            id: self.id()?,
            span,
            kind: ExprKind::StructInit { ty, fields },
        })
    }

    fn array_literal(&self, pair: Pair<'_, Rule>) -> Result<Expr, Diagnostic> {
        let span = self.pair_span(&pair)?;
        let elements = pair
            .into_inner()
            .map(|element| self.expression(element))
            .collect::<Result<_, _>>()?;
        Ok(Expr {
            id: self.id()?,
            span,
            kind: ExprKind::Array(elements),
        })
    }

    fn lambda(&self, pair: Pair<'_, Rule>) -> Result<Expr, Diagnostic> {
        let span = self.pair_span(&pair)?;
        let function = self.function(pair)?;
        Ok(Expr {
            id: self.id()?,
            span,
            kind: ExprKind::Lambda(LambdaExpr {
                parameters: function.parameters,
                return_type: function.return_type,
                body: function.body,
            }),
        })
    }

    fn access(&self, pair: Pair<'_, Rule>) -> Result<Expr, Diagnostic> {
        let span = self.pair_span(&pair)?;
        let first = pair
            .into_inner()
            .next()
            .ok_or_else(|| self.error(span, "access expression is missing its receiver"))?;
        if first.as_rule() == Rule::IDENTIFIER {
            return self.name_expr(first.as_str(), span);
        }
        let mut steps = first.into_inner();
        let base = steps
            .next()
            .ok_or_else(|| self.error(span, "access expression is missing its receiver"))?;
        let mut receiver = match base.as_rule() {
            Rule::IDENTIFIER => self.name_expr(base.as_str(), self.pair_span(&base)?)?,
            Rule::func_call => self.function_call(base)?,
            rule => return Err(self.error(span, format!("invalid access receiver {rule:?}"))),
        };
        for step in steps {
            receiver = self.access_step(receiver, step)?;
        }
        Ok(receiver)
    }

    fn access_step(&self, receiver: Expr, pair: Pair<'_, Rule>) -> Result<Expr, Diagnostic> {
        let span = self.pair_span(&pair)?;
        let step = if pair.as_rule() == Rule::access_step {
            pair.into_inner()
                .next()
                .ok_or_else(|| self.error(span, "empty access step"))?
        } else {
            pair
        };
        let kind = match step.as_rule() {
            Rule::member_access => {
                let member = step
                    .into_inner()
                    .next()
                    .ok_or_else(|| self.error(span, "member access is missing its member"))?;
                match member.as_rule() {
                    Rule::IDENTIFIER => ExprKind::Field {
                        receiver: Box::new(receiver),
                        name: member.as_str().to_string(),
                    },
                    Rule::func_call => {
                        let call = self.function_call(member)?;
                        let ExprKind::Call {
                            callee,
                            type_arguments,
                            argument_groups,
                        } = call.kind
                        else {
                            return Err(self.error(span, "member call did not produce a call"));
                        };
                        let ExprKind::Name(path) = callee.kind else {
                            return Err(self.error(span, "member call name is not an identifier"));
                        };
                        let name = path.qualified_name();
                        let field = Expr {
                            id: self.id()?,
                            span,
                            kind: ExprKind::Field {
                                receiver: Box::new(receiver),
                                name,
                            },
                        };
                        ExprKind::Call {
                            callee: Box::new(field),
                            type_arguments,
                            argument_groups,
                        }
                    }
                    rule => {
                        return Err(self.error(span, format!("unsupported member access {rule:?}")))
                    }
                }
            }
            Rule::array_access => ExprKind::Index {
                receiver: Box::new(receiver),
                coordinates: step
                    .into_inner()
                    .map(|coordinate| self.expression(coordinate))
                    .collect::<Result<_, _>>()?,
            },
            Rule::slice_access => {
                let mut bounds = step.into_inner();
                let start = self.slice_bound(bounds.next(), span)?;
                let end = self.slice_bound(bounds.next(), span)?;
                ExprKind::Slice {
                    receiver: Box::new(receiver),
                    start: start.map(Box::new),
                    end: end.map(Box::new),
                }
            }
            Rule::deref_access => ExprKind::Unary {
                operator: UnaryOperator::Dereference,
                operand: Box::new(receiver),
            },
            rule => return Err(self.error(span, format!("unsupported access step {rule:?}"))),
        };
        Ok(Expr {
            id: self.id()?,
            span,
            kind,
        })
    }

    fn slice_bound(
        &self,
        pair: Option<Pair<'_, Rule>>,
        span: Span,
    ) -> Result<Option<Expr>, Diagnostic> {
        let pair = pair.ok_or_else(|| self.error(span, "slice is missing one of its bounds"))?;
        pair.into_inner()
            .next()
            .map(|expression| self.expression(expression))
            .transpose()
    }

    fn name_expr(&self, name: &str, span: Span) -> Result<Expr, Diagnostic> {
        Ok(Expr {
            id: self.id()?,
            span,
            kind: ExprKind::Name(Path::single(name)),
        })
    }

    fn binary_operator(&self, pair: Pair<'_, Rule>) -> Result<BinaryOperator, Diagnostic> {
        let span = self.pair_span(&pair)?;
        match pair.as_rule() {
            Rule::add => Ok(BinaryOperator::Add),
            Rule::subtract => Ok(BinaryOperator::Subtract),
            Rule::multiply => Ok(BinaryOperator::Multiply),
            Rule::divide => Ok(BinaryOperator::Divide),
            Rule::modulus => Ok(BinaryOperator::Modulo),
            Rule::power => Ok(BinaryOperator::Power),
            Rule::eq => Ok(BinaryOperator::Equals),
            Rule::not_eq => Ok(BinaryOperator::NotEquals),
            Rule::lt => Ok(BinaryOperator::LessThan),
            Rule::gt => Ok(BinaryOperator::GreaterThan),
            Rule::lte => Ok(BinaryOperator::LessThanOrEqual),
            Rule::gte => Ok(BinaryOperator::GreaterThanOrEqual),
            Rule::and => Ok(BinaryOperator::And),
            Rule::or => Ok(BinaryOperator::Or),
            rule => Err(self.error(span, format!("unsupported binary operator {rule:?}"))),
        }
    }

    fn path(&self, pair: Pair<'_, Rule>) -> Path {
        Path {
            segments: pair
                .into_inner()
                .map(|segment| segment.as_str().to_string())
                .collect(),
        }
    }

    fn identifier(&self, pair: Pair<'_, Rule>) -> Result<String, Diagnostic> {
        let span = self.pair_span(&pair)?;
        if pair.as_rule() == Rule::IDENTIFIER {
            Ok(pair.as_str().to_string())
        } else {
            Err(self.error(
                span,
                format!("expected identifier, found {:?}", pair.as_rule()),
            ))
        }
    }

    fn unwrap<'a>(
        &self,
        pair: Pair<'a, Rule>,
        expected: Rule,
        description: &str,
    ) -> Result<Pair<'a, Rule>, Diagnostic> {
        let span = self.pair_span(&pair)?;
        if pair.as_rule() != expected {
            return Ok(pair);
        }
        pair.into_inner()
            .next()
            .ok_or_else(|| self.error(span, format!("empty {description}")))
    }

    fn ensure_no_extra(
        &self,
        extra: Option<Pair<'_, Rule>>,
        span: Span,
        description: &str,
    ) -> Result<(), Diagnostic> {
        if let Some(extra) = extra {
            Err(self.error(
                span,
                format!("unexpected {:?} in {description}", extra.as_rule()),
            ))
        } else {
            Ok(())
        }
    }

    fn required<'a>(
        &self,
        pair: Option<Pair<'a, Rule>>,
        span: Span,
        description: &str,
    ) -> Result<Pair<'a, Rule>, Diagnostic> {
        pair.ok_or_else(|| self.error(span, format!("parser omitted {description}")))
    }

    fn error(&self, span: Span, message: impl Into<String>) -> Diagnostic {
        Diagnostic::error(message).with_code("E1003").at(span)
    }
}

fn parse_string(literal: &str) -> Result<String, String> {
    if literal.len() < 2 || !literal.starts_with('"') || !literal.ends_with('"') {
        return Err("invalid string literal".to_string());
    }
    Ok(literal[1..literal.len() - 1].to_string())
}

fn parse_char(literal: &str) -> Result<char, String> {
    if literal.len() < 3 || !literal.starts_with('\'') || !literal.ends_with('\'') {
        return Err(format!("invalid character literal `{literal}`"));
    }
    let body = &literal[1..literal.len() - 1];
    let value = match body {
        "\\n" => '\n',
        "\\r" => '\r',
        "\\t" => '\t',
        "\\0" => '\0',
        "\\'" => '\'',
        "\\\\" => '\\',
        _ => {
            let mut chars = body.chars();
            let value = chars
                .next()
                .ok_or_else(|| format!("empty character literal `{literal}`"))?;
            if chars.next().is_some() {
                return Err(format!(
                    "character literal `{literal}` contains multiple characters"
                ));
            }
            value
        }
    };
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn produces_category_specific_module_nodes_with_precise_spans() {
        let source = r#"
            module geometry;
            export struct Point { x: int; y: int; }
            function sum(point: Point): int { return point.x + point.y; }
        "#;
        let mut sources = SourceMap::default();
        let file = sources.add_file("geometry.skunk", source).unwrap();
        let module = parse_module(&sources, file).unwrap();

        assert_eq!(module.name.as_ref().unwrap().qualified_name(), "geometry");
        assert!(matches!(
            module.entries[0],
            TopLevel {
                visibility: Visibility::Public,
                kind: TopLevelKind::Struct(_),
                ..
            }
        ));
        assert!(matches!(module.entries[1].kind, TopLevelKind::Function(_)));
        assert!(module.entries[0].span.end < module.span.end);
    }

    #[test]
    fn oversized_integer_is_a_diagnostic_not_a_panic() {
        let source = "function main(): void { print(999999999999999999999999999999); }";
        let mut sources = SourceMap::default();
        let file = sources.add_file("large.skunk", source).unwrap();
        let diagnostics = parse_module(&sources, file).unwrap_err();

        assert!(diagnostics[0]
            .message
            .contains("outside the supported 64-bit range"));
    }
}
