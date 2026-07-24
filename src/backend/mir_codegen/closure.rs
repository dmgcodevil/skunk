//! Closure environment construction and indirect function-value calls.

use super::*;

impl<'a> FunctionCompiler<'a> {
    pub(super) fn compile_mir_closure(
        &mut self,
        source: NodeId,
        capture_places: &[ir::Place],
        function_type: &LlvmType,
        locals: &[LocalVar],
        context: &CodegenContext<'_>,
    ) -> Result<ExprValue, String> {
        let LlvmType::Function {
            parameters,
            return_type,
        } = function_type
        else {
            return Err(format!(
                "MIR closure cannot initialize `{}`",
                function_type.ir()
            ));
        };
        let function = context
            .closure_functions
            .get(&source)
            .cloned()
            .ok_or_else(|| format!("missing MIR closure body {source:?}"))?;
        if function.parameters.len() != parameters.len() {
            return Err(format!(
                "MIR closure expected {} parameters, found {}",
                parameters.len(),
                function.parameters.len()
            ));
        }
        let captures = capture_places
            .iter()
            .map(|place| self.mir_place(place, locals, context))
            .collect::<Result<Vec<_>, _>>()?;
        if captures.len() != function.captures.len() {
            return Err(format!(
                "MIR closure expected {} captures, found {}",
                function.captures.len(),
                captures.len()
            ));
        }

        let lambda_id = *self.lambda_counter;
        *self.lambda_counter += 1;
        let symbol_name = format!(
            "skunk_mir_lambda_{}_{}",
            sanitize_name(self.function_name),
            lambda_id
        );
        let environment_type = format!("{}_env", symbol_name);
        self.extra_type_decls.push(format!(
            "%env.{} = type {{ {} }}",
            sanitize_name(&environment_type),
            vec!["ptr"; captures.len()].join(", ")
        ));

        let dependencies = FunctionCompilerDependencies {
            signatures: self.signatures,
            structs: self.structs,
            enums: self.enums,
            traits: self.traits,
            trait_vtables: self.trait_vtables,
            globals: self.globals,
            extra_type_decls: self.extra_type_decls,
            extra_function_irs: self.extra_function_irs,
            lambda_counter: self.lambda_counter,
        };
        let nested_compiler =
            FunctionCompiler::new(&symbol_name, return_type.as_ref().clone(), dependencies);
        let body_lines =
            nested_compiler.compile_mir_with_env(&function, context, Some(&environment_type))?;
        let param_defs = parameters
            .iter()
            .enumerate()
            .map(|(index, ty)| format!("{} %arg{}", ty.ir(), index + 1));
        let mut function_ir = String::new();
        let _ = writeln!(
            function_ir,
            "define {} @{}({}) {{",
            return_type.ir(),
            symbol_name,
            std::iter::once("ptr %env".to_string())
                .chain(param_defs)
                .collect::<Vec<_>>()
                .join(", ")
        );
        let _ = writeln!(function_ir, "entry:");
        for line in body_lines {
            let _ = writeln!(function_ir, "{}", line);
        }
        let _ = writeln!(function_ir, "}}");
        self.extra_function_irs.push(function_ir);

        let environment_ptr = if captures.is_empty() {
            "null".to_string()
        } else {
            let environment_ptr = self.next_temp();
            self.emit_line(format!(
                "{} = call ptr @malloc(i64 {})",
                environment_ptr,
                captures.len() * 8
            ));
            for (index, capture) in captures.iter().enumerate() {
                let field_ptr = self.next_temp();
                self.emit_line(format!(
                    "{} = getelementptr inbounds %env.{}, ptr {}, i32 0, i32 {}",
                    field_ptr,
                    sanitize_name(&environment_type),
                    environment_ptr,
                    index
                ));
                self.emit_line(format!(
                    "store ptr {}, ptr {}, align 8",
                    capture.ptr, field_ptr
                ));
            }
            environment_ptr
        };

        let with_function = self.next_temp();
        self.emit_line(format!(
            "{} = insertvalue {} zeroinitializer, ptr @{}, 0",
            with_function,
            function_type.ir(),
            symbol_name
        ));
        let full = self.next_temp();
        self.emit_line(format!(
            "{} = insertvalue {} {}, ptr {}, 1",
            full,
            function_type.ir(),
            with_function,
            environment_ptr
        ));
        Ok(ExprValue {
            llvm_type: function_type.clone(),
            value: full,
        })
    }

    pub(super) fn compile_mir_closure_call(
        &mut self,
        callee: ExprValue,
        arguments: Vec<ExprValue>,
    ) -> Result<ExprValue, String> {
        let (parameters, return_type) = match &callee.llvm_type {
            LlvmType::Function {
                parameters,
                return_type,
            } => (parameters.clone(), return_type.as_ref().clone()),
            other => {
                return Err(format!(
                    "direct MIR closure call requires a function value, found `{}`",
                    other.ir()
                ))
            }
        };
        if arguments.len() != parameters.len() {
            return Err(format!(
                "direct MIR closure call expected {} arguments, found {}",
                parameters.len(),
                arguments.len()
            ));
        }
        let function_ptr = self.next_temp();
        self.emit_line(format!(
            "{} = extractvalue {} {}, 0",
            function_ptr,
            callee.llvm_type.ir(),
            callee.value
        ));
        let environment_ptr = self.next_temp();
        self.emit_line(format!(
            "{} = extractvalue {} {}, 1",
            environment_ptr,
            callee.llvm_type.ir(),
            callee.value
        ));
        let mut rendered = vec![format!("ptr {environment_ptr}")];
        for (argument, expected) in arguments.into_iter().zip(&parameters) {
            let argument = self.coerce_expr(argument, expected, "MIR closure argument")?;
            rendered.push(format!("{} {}", argument.llvm_type.ir(), argument.value));
        }
        if return_type == LlvmType::Void {
            self.emit_line(format!(
                "call void {}({})",
                function_ptr,
                rendered.join(", ")
            ));
            Ok(ExprValue {
                llvm_type: LlvmType::Void,
                value: "void".to_string(),
            })
        } else {
            let result = self.next_temp();
            self.emit_line(format!(
                "{} = call {} {}({})",
                result,
                return_type.ir(),
                function_ptr,
                rendered.join(", ")
            ));
            Ok(ExprValue {
                llvm_type: return_type,
                value: result,
            })
        }
    }
}
