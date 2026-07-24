//! Low-level LLVM emission shared by MIR instruction selection.

use super::*;

impl<'a> FunctionCompiler<'a> {
    pub(super) fn new(
        function_name: &'a str,
        return_type: LlvmType,
        dependencies: FunctionCompilerDependencies<'a>,
    ) -> Self {
        Self {
            function_name,
            return_type,
            signatures: dependencies.signatures,
            structs: dependencies.structs,
            enums: dependencies.enums,
            traits: dependencies.traits,
            trait_vtables: dependencies.trait_vtables,
            globals: dependencies.globals,
            extra_type_decls: dependencies.extra_type_decls,
            extra_function_irs: dependencies.extra_function_irs,
            lambda_counter: dependencies.lambda_counter,
            lines: Vec::new(),
            temp_counter: 0,
            label_counter: 0,
        }
    }

    pub(super) fn compile_print_value(&mut self, value: ExprValue) -> Result<(), String> {
        match value.llvm_type.clone() {
            LlvmType::I8 | LlvmType::I16 | LlvmType::I32 => {
                let format = self.global_c_string("fmt_i32", "%d\n");
                let value = self.coerce_expr(value, &LlvmType::I32, "print")?;
                self.emit_line(format!(
                    "call i32 (ptr, ...) @printf(ptr {}, i32 {})",
                    self.string_ptr(&format),
                    value.value
                ));
                Ok(())
            }
            LlvmType::I64 => {
                let format = self.global_c_string("fmt_int", "%lld\n");
                self.emit_line(format!(
                    "call i32 (ptr, ...) @printf(ptr {}, i64 {})",
                    self.string_ptr(&format),
                    value.value
                ));
                Ok(())
            }
            LlvmType::I1 => {
                let format = self.global_c_string("fmt_str", "%s\n");
                let true_string = self.global_c_string("bool_true", "true");
                let false_string = self.global_c_string("bool_false", "false");
                let selected = self.next_temp();
                self.emit_line(format!(
                    "{} = select i1 {}, ptr {}, ptr {}",
                    selected,
                    value.value,
                    self.string_ptr(&true_string),
                    self.string_ptr(&false_string)
                ));
                self.emit_line(format!(
                    "call i32 (ptr, ...) @printf(ptr {}, ptr {})",
                    self.string_ptr(&format),
                    selected
                ));
                Ok(())
            }
            LlvmType::F32 => {
                let format = self.global_c_string("fmt_float", "%f\n");
                let value = self.coerce_expr(value, &LlvmType::F64, "print")?;
                self.emit_line(format!(
                    "call i32 (ptr, ...) @printf(ptr {}, double {})",
                    self.string_ptr(&format),
                    value.value
                ));
                Ok(())
            }
            LlvmType::F64 => {
                let format = self.global_c_string("fmt_float", "%f\n");
                self.emit_line(format!(
                    "call i32 (ptr, ...) @printf(ptr {}, double {})",
                    self.string_ptr(&format),
                    value.value
                ));
                Ok(())
            }
            LlvmType::Char16 => {
                let format = self.global_c_string("fmt_char", "%lc\n");
                let value = self.coerce_expr(value, &LlvmType::I32, "print")?;
                self.emit_line(format!(
                    "call i32 (ptr, ...) @printf(ptr {}, i32 {})",
                    self.string_ptr(&format),
                    value.value
                ));
                Ok(())
            }
            LlvmType::PtrI8 => {
                let format = self.global_c_string("fmt_str", "%s\n");
                self.emit_line(format!(
                    "call i32 (ptr, ...) @printf(ptr {}, ptr {})",
                    self.string_ptr(&format),
                    value.value
                ));
                Ok(())
            }
            LlvmType::Allocator
            | LlvmType::Arena
            | LlvmType::Window
            | LlvmType::TraitObject(_)
            | LlvmType::TraitIntersection(_)
            | LlvmType::Reference { .. }
            | LlvmType::Pointer { .. } => {
                Err("cannot print a pointer-like value directly".to_string())
            }
            LlvmType::Struct(_) => Err("cannot print a struct value directly".to_string()),
            LlvmType::Enum(_) => Err("cannot print an enum value directly".to_string()),
            LlvmType::Union(_) => Err("cannot print a union value directly".to_string()),
            LlvmType::Function { .. } => Err("cannot print a function value directly".to_string()),
            LlvmType::Slice { .. } => Err("cannot print a slice value directly".to_string()),
            LlvmType::Array { .. } => Err("cannot print an array value directly".to_string()),
            LlvmType::Void => Err("cannot print a void value".to_string()),
        }
    }

    pub(super) fn build_slice_from_array_ptr(
        &mut self,
        array_ptr: &str,
        array_type: &LlvmType,
    ) -> Result<ExprValue, String> {
        let (element_type, len) = match array_type {
            LlvmType::Array { elem_type, len } => (elem_type.as_ref(), *len),
            other => {
                return Err(format!(
                    "expected array storage for slice construction, found `{}`",
                    other.ir()
                ))
            }
        };
        let length = ExprValue {
            llvm_type: LlvmType::I32,
            value: len.to_string(),
        };
        let data_ptr = if len == 0 {
            "null".to_string()
        } else {
            let data_ptr = self.next_temp();
            self.emit_line(format!(
                "{} = getelementptr inbounds {}, ptr {}, i64 0, i64 0",
                data_ptr,
                array_type.ir(),
                array_ptr
            ));
            data_ptr
        };
        self.build_slice_header_from_values(
            &data_ptr,
            element_type,
            &length,
            ExprValue {
                llvm_type: LlvmType::I32,
                value: "0".to_string(),
            },
            length.clone(),
        )
    }

    pub(super) fn build_slice_header_from_values(
        &mut self,
        data_ptr: &str,
        element_type: &LlvmType,
        base_len: &ExprValue,
        start: ExprValue,
        end: ExprValue,
    ) -> Result<ExprValue, String> {
        self.emit_slice_range_bounds_check(&start, &end, base_len)?;
        let start_i64 = self.coerce_expr(start.clone(), &LlvmType::I64, "slice start")?;
        let offset_ptr = if data_ptr == "null" {
            "null".to_string()
        } else {
            let offset_ptr = self.next_temp();
            self.emit_line(format!(
                "{} = getelementptr inbounds {}, ptr {}, i64 {}",
                offset_ptr,
                element_type.ir(),
                data_ptr,
                start_i64.value
            ));
            offset_ptr
        };
        let slice_len = self.next_temp();
        self.emit_line(format!(
            "{} = sub i32 {}, {}",
            slice_len, end.value, start.value
        ));
        let slice_type = LlvmType::Slice {
            elem_type: Box::new(element_type.clone()),
        };
        let with_ptr = self.next_temp();
        self.emit_line(format!(
            "{} = insertvalue {} zeroinitializer, ptr {}, 0",
            with_ptr,
            slice_type.ir(),
            offset_ptr
        ));
        let complete = self.next_temp();
        self.emit_line(format!(
            "{} = insertvalue {} {}, i32 {}, 1",
            complete,
            slice_type.ir(),
            with_ptr,
            slice_len
        ));
        Ok(ExprValue {
            llvm_type: slice_type,
            value: complete,
        })
    }

    pub(super) fn load_from_ptr(
        &mut self,
        ptr: &str,
        llvm_type: &LlvmType,
    ) -> Result<ExprValue, String> {
        let value = self.next_temp();
        self.emit_line(format!(
            "{} = load {}, ptr {}, align {}",
            value,
            llvm_type.ir(),
            ptr,
            self.align_of(llvm_type)
        ));
        Ok(ExprValue {
            llvm_type: llvm_type.clone(),
            value,
        })
    }

    pub(super) fn extract_slice_data(&mut self, slice: &ExprValue) -> Result<String, String> {
        if !matches!(slice.llvm_type, LlvmType::Slice { .. }) {
            return Err(format!(
                "expected slice value, found `{}`",
                slice.llvm_type.ir()
            ));
        }
        let data_ptr = self.next_temp();
        self.emit_line(format!(
            "{} = extractvalue {} {}, 0",
            data_ptr,
            slice.llvm_type.ir(),
            slice.value
        ));
        Ok(data_ptr)
    }

    pub(super) fn extract_slice_len(&mut self, slice: &ExprValue) -> Result<ExprValue, String> {
        if !matches!(slice.llvm_type, LlvmType::Slice { .. }) {
            return Err(format!(
                "expected slice value, found `{}`",
                slice.llvm_type.ir()
            ));
        }
        let length = self.next_temp();
        self.emit_line(format!(
            "{} = extractvalue {} {}, 1",
            length,
            slice.llvm_type.ir(),
            slice.value
        ));
        Ok(ExprValue {
            llvm_type: LlvmType::I32,
            value: length,
        })
    }

    pub(super) fn emit_index_bounds_check(
        &mut self,
        index: &ExprValue,
        length: &ExprValue,
    ) -> Result<(), String> {
        let index = self.coerce_expr(index.clone(), &LlvmType::I64, "bounds-check index")?;
        let length = self.coerce_expr(length.clone(), &LlvmType::I64, "bounds-check length")?;
        let non_negative = self.next_temp();
        self.emit_line(format!(
            "{} = icmp sge i64 {}, 0",
            non_negative, index.value
        ));
        let below_length = self.next_temp();
        self.emit_line(format!(
            "{} = icmp slt i64 {}, {}",
            below_length, index.value, length.value
        ));
        let valid = self.next_temp();
        self.emit_line(format!(
            "{} = and i1 {}, {}",
            valid, non_negative, below_length
        ));
        let ok = self.next_label("bounds_ok");
        let panic = self.next_label("bounds_panic");
        self.emit_line(format!("br i1 {}, label %{}, label %{}", valid, ok, panic));
        self.emit_label(&panic);
        self.emit_line(format!(
            "call void @skunk_panic_index_out_of_bounds(i64 {}, i64 {})",
            index.value, length.value
        ));
        self.emit_line("unreachable".to_string());
        self.emit_label(&ok);
        Ok(())
    }

    fn emit_slice_range_bounds_check(
        &mut self,
        start: &ExprValue,
        end: &ExprValue,
        length: &ExprValue,
    ) -> Result<(), String> {
        let start = self.coerce_expr(start.clone(), &LlvmType::I64, "slice start")?;
        let end = self.coerce_expr(end.clone(), &LlvmType::I64, "slice end")?;
        let length = self.coerce_expr(length.clone(), &LlvmType::I64, "slice length")?;
        let non_negative = self.next_temp();
        self.emit_line(format!(
            "{} = icmp sge i64 {}, 0",
            non_negative, start.value
        ));
        let ordered = self.next_temp();
        self.emit_line(format!(
            "{} = icmp sle i64 {}, {}",
            ordered, start.value, end.value
        ));
        let in_bounds = self.next_temp();
        self.emit_line(format!(
            "{} = icmp sle i64 {}, {}",
            in_bounds, end.value, length.value
        ));
        let valid_start = self.next_temp();
        self.emit_line(format!(
            "{} = and i1 {}, {}",
            valid_start, non_negative, ordered
        ));
        let valid = self.next_temp();
        self.emit_line(format!("{} = and i1 {}, {}", valid, valid_start, in_bounds));
        let ok = self.next_label("slice_bounds_ok");
        let panic = self.next_label("slice_bounds_panic");
        self.emit_line(format!("br i1 {}, label %{}, label %{}", valid, ok, panic));
        self.emit_label(&panic);
        self.emit_line(format!(
            "call void @skunk_panic_slice_range_out_of_bounds(i64 {}, i64 {}, i64 {})",
            start.value, end.value, length.value
        ));
        self.emit_line("unreachable".to_string());
        self.emit_label(&ok);
        Ok(())
    }

    pub(super) fn default_value(&self, llvm_type: &LlvmType) -> ExprValue {
        let value = match llvm_type {
            LlvmType::F32 | LlvmType::F64 => "0.0",
            LlvmType::PtrI8
            | LlvmType::Allocator
            | LlvmType::Arena
            | LlvmType::Window
            | LlvmType::Reference { .. }
            | LlvmType::Pointer { .. } => "null",
            LlvmType::Struct(_)
            | LlvmType::Enum(_)
            | LlvmType::TraitObject(_)
            | LlvmType::TraitIntersection(_)
            | LlvmType::Union(_)
            | LlvmType::Function { .. }
            | LlvmType::Slice { .. }
            | LlvmType::Array { .. } => "zeroinitializer",
            LlvmType::Void => "void",
            _ => "0",
        };
        ExprValue {
            llvm_type: llvm_type.clone(),
            value: value.to_string(),
        }
    }

    pub(super) fn global_c_string(&mut self, prefix: &str, value: &str) -> String {
        let mut bytes = value.as_bytes().to_vec();
        bytes.push(0);
        if let Some(existing) = self.globals.iter().find(|global| global.bytes == bytes) {
            return existing.name.clone();
        }
        let name = format!("{}.{}", prefix, self.globals.len());
        self.globals.push(GlobalString {
            name: name.clone(),
            bytes,
        });
        name
    }

    pub(super) fn string_ptr(&self, global: &str) -> String {
        format!("getelementptr inbounds (i8, ptr @{}, i64 0)", global)
    }

    pub(super) fn emit_store(&mut self, ptr: &str, value: &ExprValue) {
        self.emit_line(format!(
            "store {} {}, ptr {}, align {}",
            value.llvm_type.ir(),
            value.value,
            ptr,
            self.align_of(&value.llvm_type)
        ));
    }

    pub(super) fn emit_heap_alloc(&mut self, llvm_type: LlvmType, hint: &str) -> String {
        let ptr = format!("%{}_{}", sanitize_name(hint), self.temp_counter);
        self.temp_counter += 1;
        self.emit_line(format!(
            "{} = call ptr @malloc(i64 {})",
            ptr,
            self.size_of(&llvm_type)
        ));
        ptr
    }

    pub(super) fn emit_label(&mut self, label: &str) {
        self.lines.push(format!("{}:", label));
    }

    pub(super) fn emit_line(&mut self, line: String) {
        self.lines.push(format!("  {}", line));
    }

    pub(super) fn next_temp(&mut self) -> String {
        let temp = format!("%t{}", self.temp_counter);
        self.temp_counter += 1;
        temp
    }

    fn next_label(&mut self, prefix: &str) -> String {
        let label = format!(
            "{}_{}_{}",
            sanitize_name(self.function_name),
            prefix,
            self.label_counter
        );
        self.label_counter += 1;
        label
    }

    pub(super) fn align_of(&self, llvm_type: &LlvmType) -> usize {
        match llvm_type {
            LlvmType::I8 | LlvmType::I1 => 1,
            LlvmType::I16 | LlvmType::Char16 => 2,
            LlvmType::I32 | LlvmType::F32 => 4,
            LlvmType::I64
            | LlvmType::F64
            | LlvmType::PtrI8
            | LlvmType::Allocator
            | LlvmType::Arena
            | LlvmType::Window
            | LlvmType::TraitObject(_)
            | LlvmType::TraitIntersection(_)
            | LlvmType::Union(_)
            | LlvmType::Reference { .. }
            | LlvmType::Pointer { .. }
            | LlvmType::Function { .. }
            | LlvmType::Slice { .. } => 8,
            LlvmType::Struct(name) => self
                .structs
                .get(name)
                .map(|layout| {
                    layout
                        .fields
                        .iter()
                        .map(|(_, field_type)| self.align_of(field_type))
                        .max()
                        .unwrap_or(1)
                })
                .unwrap_or(8),
            LlvmType::Enum(name) => self
                .enums
                .get(name)
                .map(|layout| {
                    layout
                        .variants
                        .iter()
                        .flat_map(|variant| &variant.payload_types)
                        .map(|payload_type| self.align_of(payload_type))
                        .max()
                        .unwrap_or(4)
                        .max(4)
                })
                .unwrap_or(8),
            LlvmType::Array { elem_type, .. } => self.align_of(elem_type),
            LlvmType::Void => 1,
        }
    }

    pub(super) fn size_of(&self, llvm_type: &LlvmType) -> usize {
        match llvm_type {
            LlvmType::I8 | LlvmType::I1 => 1,
            LlvmType::I16 | LlvmType::Char16 => 2,
            LlvmType::I32 | LlvmType::F32 => 4,
            LlvmType::I64
            | LlvmType::F64
            | LlvmType::PtrI8
            | LlvmType::Allocator
            | LlvmType::Arena
            | LlvmType::Window
            | LlvmType::Reference { .. }
            | LlvmType::Pointer { .. } => 8,
            LlvmType::TraitObject(_)
            | LlvmType::Union(_)
            | LlvmType::Function { .. }
            | LlvmType::Slice { .. } => 16,
            LlvmType::TraitIntersection(traits) => 8 * (traits.len() + 1),
            LlvmType::Array { elem_type, len } => self.size_of(elem_type) * len,
            LlvmType::Struct(name) => self
                .structs
                .get(name)
                .map(|layout| aggregate_size(layout.fields.iter().map(|(_, ty)| ty), self))
                .unwrap_or(8),
            LlvmType::Enum(name) => self
                .enums
                .get(name)
                .map(|layout| {
                    aggregate_size(
                        std::iter::once(&LlvmType::I32).chain(
                            layout
                                .variants
                                .iter()
                                .flat_map(|variant| &variant.payload_types),
                        ),
                        self,
                    )
                })
                .unwrap_or(8),
            LlvmType::Void => 1,
        }
    }
}

fn aggregate_size<'a>(
    fields: impl IntoIterator<Item = &'a LlvmType>,
    compiler: &FunctionCompiler<'_>,
) -> usize {
    let mut offset = 0;
    let mut max_align = 1;
    for field in fields {
        let align = compiler.align_of(field);
        max_align = max_align.max(align);
        offset = align_up(offset, align);
        offset += compiler.size_of(field);
    }
    align_up(offset, max_align)
}
