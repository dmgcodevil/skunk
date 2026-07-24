//! Runtime and layout intrinsics lowered from resolved MIR call targets.

use super::*;

impl<'a> FunctionCompiler<'a> {
    pub(super) fn compile_mir_static_intrinsic(
        &mut self,
        owner: TypeId,
        name: &str,
        arguments: Vec<ExprValue>,
        expected: &LlvmType,
        context: &CodegenContext<'_>,
    ) -> Result<ExprValue, String> {
        if matches!(name, "size_of" | "align_of") {
            if !arguments.is_empty() {
                return Err(format!("intrinsic `{name}` does not accept arguments"));
            }
            let measured = context.llvm_type(owner)?;
            let value = if name == "size_of" {
                self.size_of(&measured)
            } else {
                self.align_of(&measured)
            };
            return Ok(ExprValue {
                llvm_type: LlvmType::I32,
                value: value.to_string(),
            });
        }

        if let SemanticTypeKind::Intrinsic(intrinsic) = context.model.types.kind(owner) {
            return match (*intrinsic, name) {
                (IntrinsicType::Bounds, "check") => {
                    let [index, length] = take_arguments(arguments, "Bounds::check")?;
                    self.emit_index_bounds_check(&index, &length)?;
                    Ok(void_value())
                }
                (IntrinsicType::Memory, "copy") => {
                    let [destination, source, count] = take_arguments(arguments, "Memory::copy")?;
                    require_pointer(&destination, "Memory::copy destination")?;
                    require_pointer(&source, "Memory::copy source")?;
                    let count = self.coerce_expr(count, &LlvmType::I64, "Memory::copy count")?;
                    self.emit_line(format!(
                        "call ptr @memcpy(ptr {}, ptr {}, i64 {})",
                        destination.value, source.value, count.value
                    ));
                    Ok(void_value())
                }
                (IntrinsicType::Memory, "set") => {
                    let [destination, value, count] = take_arguments(arguments, "Memory::set")?;
                    require_pointer(&destination, "Memory::set destination")?;
                    let value = self.coerce_expr(value, &LlvmType::I32, "Memory::set value")?;
                    let count = self.coerce_expr(count, &LlvmType::I64, "Memory::set count")?;
                    self.emit_line(format!(
                        "call ptr @memset(ptr {}, i32 {}, i64 {})",
                        destination.value, value.value, count.value
                    ));
                    Ok(void_value())
                }
                (IntrinsicType::Color, "black" | "white" | "red" | "green" | "blue") => {
                    if !arguments.is_empty() {
                        return Err(format!("Color::{name} expects no arguments"));
                    }
                    let value = match name {
                        "black" => 0xFF000000u32,
                        "white" => 0xFFFFFFFFu32,
                        "red" => 0xFFFF0000u32,
                        "green" => 0xFF00FF00u32,
                        "blue" => 0xFF0000FFu32,
                        _ => return Err(format!("unknown Color intrinsic `{name}`")),
                    };
                    Ok(ExprValue {
                        llvm_type: LlvmType::I32,
                        value: value.to_string(),
                    })
                }
                (IntrinsicType::Color, "rgb" | "rgba") => self.compile_mir_color(name, arguments),
                (IntrinsicType::Window, "create") => {
                    let [width, height, title] = take_arguments(arguments, "Window::create")?;
                    let width = self.coerce_expr(width, &LlvmType::I32, "Window::create width")?;
                    let height =
                        self.coerce_expr(height, &LlvmType::I32, "Window::create height")?;
                    let title =
                        self.coerce_expr(title, &LlvmType::PtrI8, "Window::create title")?;
                    let result = self.next_temp();
                    self.emit_line(format!(
                        "{} = call ptr @skunk_window_create(i32 {}, i32 {}, ptr {})",
                        result, width.value, height.value, title.value
                    ));
                    Ok(ExprValue {
                        llvm_type: LlvmType::Window,
                        value: result,
                    })
                }
                (IntrinsicType::Keyboard, "is_down") => {
                    let [window, key] = take_arguments(arguments, "Keyboard::is_down")?;
                    let window =
                        self.coerce_expr(window, &LlvmType::Window, "Keyboard::is_down window")?;
                    let key = self.coerce_expr(key, &LlvmType::Char16, "Keyboard::is_down key")?;
                    let result = self.next_temp();
                    self.emit_line(format!(
                        "{} = call i1 @skunk_keyboard_is_down(ptr {}, i16 {})",
                        result, window.value, key.value
                    ));
                    Ok(ExprValue {
                        llvm_type: LlvmType::I1,
                        value: result,
                    })
                }
                (IntrinsicType::System, "allocator") => {
                    if !arguments.is_empty() {
                        return Err("System::allocator expects no arguments".to_string());
                    }
                    let result = self.next_temp();
                    self.emit_line(format!("{} = call ptr @skunk_system_allocator()", result));
                    Ok(ExprValue {
                        llvm_type: LlvmType::Allocator,
                        value: result,
                    })
                }
                _ => Err(format!(
                    "unsupported direct MIR intrinsic `{}::{name}`",
                    intrinsic.name()
                )),
            };
        }

        let owner_type = context.llvm_type(owner)?;
        match (&owner_type, name) {
            (LlvmType::Pointer { .. }, "cast") => {
                let [value] = take_arguments(arguments, "pointer cast")?;
                require_pointer(&value, "pointer cast")?;
                Ok(ExprValue {
                    llvm_type: expected.clone(),
                    value: value.value,
                })
            }
            (LlvmType::Pointer { target_type }, "offset") => {
                if target_type.as_ref() != &LlvmType::I8 {
                    return Err("pointer offset requires a byte pointer".to_string());
                }
                let [base, offset] = take_arguments(arguments, "pointer offset")?;
                require_pointer(&base, "pointer offset")?;
                let offset = self.coerce_expr(offset, &LlvmType::I64, "pointer offset")?;
                let result = self.next_temp();
                self.emit_line(format!(
                    "{} = getelementptr inbounds i8, ptr {}, i64 {}",
                    result, base.value, offset.value
                ));
                Ok(ExprValue {
                    llvm_type: owner_type,
                    value: result,
                })
            }
            (LlvmType::Arena, "init") => {
                let [allocator] = take_arguments(arguments, "Arena::init")?;
                let allocator = self.coerce_expr(allocator, &LlvmType::Allocator, "Arena::init")?;
                let result = self.next_temp();
                self.emit_line(format!(
                    "{} = call ptr @skunk_arena_init(ptr {})",
                    result, allocator.value
                ));
                Ok(ExprValue {
                    llvm_type: LlvmType::Arena,
                    value: result,
                })
            }
            (LlvmType::Slice { elem_type }, "alloc") => {
                let [allocator, length] = take_arguments(arguments, "slice alloc")?;
                let allocator =
                    self.coerce_expr(allocator, &LlvmType::Allocator, "slice alloc allocator")?;
                let length = self.coerce_expr(length, &LlvmType::I32, "slice alloc length")?;
                let data_ptr = self.next_temp();
                self.emit_line(format!(
                    "{} = call ptr @skunk_alloc_buffer(ptr {}, i64 {}, i32 {})",
                    data_ptr,
                    allocator.value,
                    self.size_of(elem_type),
                    length.value
                ));
                let with_ptr = self.next_temp();
                self.emit_line(format!(
                    "{} = insertvalue {} zeroinitializer, ptr {}, 0",
                    with_ptr,
                    owner_type.ir(),
                    data_ptr
                ));
                let result = self.next_temp();
                self.emit_line(format!(
                    "{} = insertvalue {} {}, i32 {}, 1",
                    result,
                    owner_type.ir(),
                    with_ptr,
                    length.value
                ));
                Ok(ExprValue {
                    llvm_type: owner_type,
                    value: result,
                })
            }
            (LlvmType::Array { .. }, "fill" | "new") => {
                let [fill] = take_arguments(arguments, "array fill")?;
                self.compile_mir_array_fill(&owner_type, &fill)
            }
            (_, "create") => {
                let [allocator] = take_arguments(arguments, "type create")?;
                let allocator = self.coerce_expr(allocator, &LlvmType::Allocator, "type create")?;
                let LlvmType::Pointer { target_type } = expected else {
                    return Err(format!(
                        "type create must return a pointer, found `{}`",
                        expected.ir()
                    ));
                };
                let result = self.next_temp();
                self.emit_line(format!(
                    "{} = call ptr @skunk_alloc_create(ptr {}, i64 {})",
                    result,
                    allocator.value,
                    self.size_of(target_type)
                ));
                Ok(ExprValue {
                    llvm_type: expected.clone(),
                    value: result,
                })
            }
            _ => Err(format!(
                "unsupported direct MIR intrinsic `{}`::`{name}`",
                owner_type.ir()
            )),
        }
    }

    pub(super) fn compile_mir_method_intrinsic(
        &mut self,
        receiver: ExprValue,
        name: &str,
        arguments: Vec<ExprValue>,
    ) -> Result<ExprValue, String> {
        match (&receiver.llvm_type, name) {
            (LlvmType::Allocator, "destroy") => {
                let [pointer] = take_arguments(arguments, "Allocator.destroy")?;
                if !matches!(pointer.llvm_type, LlvmType::Pointer { .. }) {
                    return Err("Allocator.destroy expects a pointer".to_string());
                }
                self.emit_line(format!(
                    "call void @skunk_alloc_destroy(ptr {}, ptr {})",
                    receiver.value, pointer.value
                ));
                Ok(void_value())
            }
            (LlvmType::Allocator, "free") => {
                let [slice] = take_arguments(arguments, "Allocator.free")?;
                if !matches!(slice.llvm_type, LlvmType::Slice { .. }) {
                    return Err("Allocator.free expects a slice".to_string());
                }
                let data_ptr = self.extract_slice_data(&slice)?;
                self.emit_line(format!(
                    "call void @skunk_alloc_free(ptr {}, ptr {})",
                    receiver.value, data_ptr
                ));
                Ok(void_value())
            }
            (LlvmType::Arena, "allocator") => {
                expect_no_arguments(arguments, "Arena.allocator")?;
                let result = self.next_temp();
                self.emit_line(format!(
                    "{} = call ptr @skunk_arena_allocator(ptr {})",
                    result, receiver.value
                ));
                Ok(ExprValue {
                    llvm_type: LlvmType::Allocator,
                    value: result,
                })
            }
            (LlvmType::Arena, "reset" | "deinit") => {
                expect_no_arguments(arguments, &format!("Arena.{name}"))?;
                let runtime_name = if name == "reset" {
                    "skunk_arena_reset"
                } else {
                    "skunk_arena_deinit"
                };
                self.emit_line(format!("call void @{runtime_name}(ptr {})", receiver.value));
                Ok(void_value())
            }
            (LlvmType::Window, _) => self.compile_mir_window_method(receiver, name, arguments),
            (other, _) => Err(format!(
                "unsupported direct MIR method intrinsic `{}`.`{name}`",
                other.ir()
            )),
        }
    }

    fn compile_mir_color(
        &mut self,
        name: &str,
        arguments: Vec<ExprValue>,
    ) -> Result<ExprValue, String> {
        let expected = if name == "rgb" { 3 } else { 4 };
        if arguments.len() != expected {
            return Err(format!("Color::{name} expects {expected} arguments"));
        }
        let mut channels = Vec::with_capacity(expected);
        for argument in arguments {
            let argument = self.coerce_expr(argument, &LlvmType::I32, "Color channel")?;
            let masked = self.next_temp();
            self.emit_line(format!("{} = and i32 {}, 255", masked, argument.value));
            channels.push(masked);
        }
        let alpha = if name == "rgb" {
            "255".to_string()
        } else {
            channels[3].clone()
        };
        let shifted_alpha = self.next_temp();
        self.emit_line(format!("{} = shl i32 {}, 24", shifted_alpha, alpha));
        let shifted_red = self.next_temp();
        self.emit_line(format!("{} = shl i32 {}, 16", shifted_red, channels[0]));
        let shifted_green = self.next_temp();
        self.emit_line(format!("{} = shl i32 {}, 8", shifted_green, channels[1]));
        let with_red = self.next_temp();
        self.emit_line(format!(
            "{} = or i32 {}, {}",
            with_red, shifted_alpha, shifted_red
        ));
        let with_green = self.next_temp();
        self.emit_line(format!(
            "{} = or i32 {}, {}",
            with_green, with_red, shifted_green
        ));
        let result = self.next_temp();
        self.emit_line(format!(
            "{} = or i32 {}, {}",
            result, with_green, channels[2]
        ));
        Ok(ExprValue {
            llvm_type: LlvmType::I32,
            value: result,
        })
    }

    fn compile_mir_array_fill(
        &mut self,
        expected: &LlvmType,
        fill: &ExprValue,
    ) -> Result<ExprValue, String> {
        let LlvmType::Array { elem_type, len } = expected else {
            return self.coerce_expr(fill.clone(), expected, "array fill");
        };
        let mut result = ExprValue {
            llvm_type: expected.clone(),
            value: "zeroinitializer".to_string(),
        };
        for index in 0..*len {
            let element = self.compile_mir_array_fill(elem_type, fill)?;
            let next = self.next_temp();
            self.emit_line(format!(
                "{} = insertvalue {} {}, {} {}, {}",
                next,
                expected.ir(),
                result.value,
                element.llvm_type.ir(),
                element.value,
                index
            ));
            result.value = next;
        }
        Ok(result)
    }

    fn compile_mir_window_method(
        &mut self,
        receiver: ExprValue,
        name: &str,
        arguments: Vec<ExprValue>,
    ) -> Result<ExprValue, String> {
        match name {
            "is_open" => {
                expect_no_arguments(arguments, "Window.is_open")?;
                let result = self.next_temp();
                self.emit_line(format!(
                    "{} = call i1 @skunk_window_is_open(ptr {})",
                    result, receiver.value
                ));
                Ok(ExprValue {
                    llvm_type: LlvmType::I1,
                    value: result,
                })
            }
            "poll" | "present" | "close" | "deinit" => {
                expect_no_arguments(arguments, &format!("Window.{name}"))?;
                let runtime_name = match name {
                    "poll" => "skunk_window_poll",
                    "present" => "skunk_window_present",
                    "close" => "skunk_window_close",
                    "deinit" => "skunk_window_deinit",
                    _ => return Err(format!("unknown Window method `{name}`")),
                };
                self.emit_line(format!("call void @{runtime_name}(ptr {})", receiver.value));
                Ok(void_value())
            }
            "clear" => {
                let [color] = take_arguments(arguments, "Window.clear")?;
                let color = self.coerce_expr(color, &LlvmType::I32, "Window.clear color")?;
                self.emit_line(format!(
                    "call void @skunk_window_clear(ptr {}, i32 {})",
                    receiver.value, color.value
                ));
                Ok(void_value())
            }
            "draw_rect" => {
                let [x, y, width, height, color] = take_arguments(arguments, "Window.draw_rect")?;
                let x = self.coerce_expr(x, &LlvmType::F64, "Window.draw_rect x")?;
                let y = self.coerce_expr(y, &LlvmType::F64, "Window.draw_rect y")?;
                let width = self.coerce_expr(width, &LlvmType::F64, "Window.draw_rect width")?;
                let height = self.coerce_expr(height, &LlvmType::F64, "Window.draw_rect height")?;
                let color = self.coerce_expr(color, &LlvmType::I32, "Window.draw_rect color")?;
                self.emit_line(format!(
                    "call void @skunk_window_draw_rect(ptr {}, double {}, double {}, double {}, double {}, i32 {})",
                    receiver.value,
                    x.value,
                    y.value,
                    width.value,
                    height.value,
                    color.value
                ));
                Ok(void_value())
            }
            "delta_time" => {
                expect_no_arguments(arguments, "Window.delta_time")?;
                let result = self.next_temp();
                self.emit_line(format!(
                    "{} = call double @skunk_window_delta_time(ptr {})",
                    result, receiver.value
                ));
                Ok(ExprValue {
                    llvm_type: LlvmType::F64,
                    value: result,
                })
            }
            _ => Err(format!("unknown Window method `{name}`")),
        }
    }
}

fn take_arguments<const N: usize>(
    arguments: Vec<ExprValue>,
    context: &str,
) -> Result<[ExprValue; N], String> {
    arguments.try_into().map_err(|arguments: Vec<ExprValue>| {
        format!("{context} expects {N} arguments, found {}", arguments.len())
    })
}

fn expect_no_arguments(arguments: Vec<ExprValue>, context: &str) -> Result<(), String> {
    let [] = take_arguments(arguments, context)?;
    Ok(())
}

fn require_pointer(value: &ExprValue, context: &str) -> Result<(), String> {
    if matches!(
        value.llvm_type,
        LlvmType::Pointer { .. } | LlvmType::Reference { .. }
    ) {
        Ok(())
    } else {
        Err(format!(
            "{context} expects a pointer, found `{}`",
            value.llvm_type.ir()
        ))
    }
}

fn void_value() -> ExprValue {
    ExprValue {
        llvm_type: LlvmType::Void,
        value: "void".to_string(),
    }
}
