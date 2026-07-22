use super::*;
use crate::ast;

fn prepared_statements(source: &str) -> Vec<Node> {
    let program = ast::parse(source);
    let Node::Program { statements } = prepare_program(&program).unwrap() else {
        panic!("expected prepared program");
    };
    statements
}

#[test]
fn monomorphizes_generic_wrap_program() {
    let statements = prepared_statements(
        r#"
        struct Box[T] {
            value: T;
        }

        function wrap[T](value: T): Box[T] {
            return Box[T] { value: value };
        }

        function main(): void {
            box: Box[int] = wrap(7);
            print(box.value);
        }
        "#,
    );

    assert!(statements.iter().any(|statement| matches!(
        statement,
        Node::StructDeclaration { name, .. } if name == "Box__int"
    )));
    assert!(statements.iter().any(|statement| matches!(
        statement,
        Node::FunctionDeclaration { name, body, .. }
            if name == "wrap__int" && matches!(body.first(), Some(Node::Return(Some(_))))
    )));
}

#[test]
fn monomorphizes_nested_generic_program() {
    let statements = prepared_statements(
        r#"
        struct Box[T] {
            value: T;
        }

        function wrap[T](value: T): Box[T] {
            return Box[T] { value: value };
        }

        function main(): void {
            inner: Box[int] = wrap(7);
            outer: Box[Box[int]] = wrap(inner);
            print(outer.value.value);
        }
        "#,
    );

    assert!(statements.iter().any(|statement| matches!(
        statement,
        Node::StructDeclaration { name, .. } if name == "Box__int"
    )));
    assert!(statements.iter().any(|statement| matches!(
        statement,
        Node::StructDeclaration { name, .. } if name == "Box__Box__int"
    )));
    assert!(statements.iter().any(|statement| matches!(
        statement,
        Node::FunctionDeclaration { name, body, .. }
            if name == "wrap__Box__int" && matches!(body.first(), Some(Node::Return(Some(_))))
    )));
}

#[test]
fn monomorphizes_generic_enum_program() {
    let statements = prepared_statements(
        r#"
        enum Option[T] {
            None;
            Some(T);
        }

        function wrap[T](value: T): Option[T] {
            return Option[T]::Some(value);
        }

        function main(): void {
            value: Option[int] = wrap(7);
            match (value) {
                case None: {
                    print(0);
                }
                case Some(v): {
                    print(v);
                }
            }
        }
        "#,
    );

    assert!(statements.iter().any(|statement| matches!(
        statement,
        Node::EnumDeclaration { name, variants, .. }
            if name == "Option__int"
                && variants
                    .iter()
                    .any(|variant| variant.name == "Some" && variant.payload_types == vec![Type::Int])
    )));
    assert!(statements.iter().any(|statement| matches!(
        statement,
        Node::FunctionDeclaration { name, .. } if name == "wrap__int"
    )));
}

#[test]
fn infers_generic_enum_constructor_arguments_from_context() {
    let statements = prepared_statements(
        r#"
        enum Outcome[T, E] {
            Ok(T);
            Err(E);
        }

        enum AppError {
            Failed;
        }

        enum Optional[T] {
            Some(T);
            None;
        }

        struct Holder {
            value: Outcome[int, AppError];
        }

        type AppResult = Outcome[int, AppError];

        function success(): AppResult {
            return Outcome::Ok(40);
        }

        function failure(): Outcome[int, AppError] {
            return Outcome::Err(AppError::Failed());
        }

        function no_value(): Optional[int] {
            return Optional::None();
        }

        function widened_value(): Outcome[long, AppError] {
            value: int = 7;
            return Outcome::Ok(value);
        }

        function consume(value: Outcome[int, AppError]): int {
            match (value) {
                case Ok(number): { return number; }
                case Err(error): { return 0; }
            }
        }

        function generic_success[T, E](value: T): Outcome[T, E] {
            return Outcome::Ok(value);
        }

        function main(): void {
            assigned: Outcome[int, AppError];
            assigned = Outcome::Ok(41);
            direct: Outcome[int, AppError] = Outcome::Err(AppError::Failed());
            inferred: Outcome[int, AppError] = generic_success(42);
            holder: Holder = Holder { value: Outcome::Ok(43) };
            values: [1]Outcome[int, AppError] = [Outcome::Ok(44)];
            print(consume(Outcome::Ok(43)));
        }
        "#,
    );

    assert!(statements.iter().any(|statement| matches!(
        statement,
        Node::EnumDeclaration { name, .. } if name == "Outcome__int__AppError"
    )));
    assert!(statements.iter().any(|statement| matches!(
        statement,
        Node::EnumDeclaration { name, .. } if name == "Optional__int"
    )));
    assert!(statements.iter().any(|statement| matches!(
        statement,
        Node::FunctionDeclaration { name, .. } if name == "generic_success__int__AppError"
    )));
}

#[test]
fn infers_generic_enum_constructor_arguments_from_payloads() {
    let statements = prepared_statements(
        r#"
        enum Pair[A, B] {
            Pair(A, B);
        }

        function main(): void {
            Pair::Pair(7, "seven");
        }
        "#,
    );

    assert!(statements.iter().any(|statement| matches!(
        statement,
        Node::EnumDeclaration { name, .. } if name == "Pair__int__string"
    )));
}

#[test]
fn reports_missing_generic_enum_constructor_arguments() {
    let program = ast::parse(
        r#"
        enum Outcome[T, E] {
            Ok(T);
            Err(E);
        }

        function main(): void {
            Outcome::Ok(7);
        }
        "#,
    );

    let error = prepare_program(&program).unwrap_err();
    assert!(error
        .contains("could not infer type argument `E` for generic enum constructor `Outcome::Ok`"));
    assert!(error.contains("write explicit type arguments"));
}

#[test]
fn rejects_conflicting_contextual_generic_enum_constructor_arguments() {
    let program = ast::parse(
        r#"
        enum Outcome[T, E] {
            Ok(T);
            Err(E);
        }

        enum AppError {
            Failed;
        }

        function invalid(): Outcome[int, AppError] {
            return Outcome::Ok("not an int");
        }

        function main(): void {}
        "#,
    );

    let error = prepare_program(&program).unwrap_err();
    assert!(error.contains("conflicting inferred types for `T`: `int` and `string`"));
}

#[test]
fn expands_concrete_and_generic_type_aliases() {
    let statements = prepared_statements(
        r#"
        type Value = string | int;
        type Either[T] = T | string;

        function use_value(value: Value): Either[int] {
            return value;
        }

        function main(): void {
            value: Value = 7;
            result: Either[int] = use_value(value);
        }
        "#,
    );
    let use_value = statements
        .iter()
        .find(|statement| {
            matches!(
                statement,
                Node::FunctionDeclaration { name, .. } if name == "use_value"
            )
        })
        .expect("prepared function");
    let Node::FunctionDeclaration {
        parameters,
        return_type,
        ..
    } = use_value
    else {
        unreachable!();
    };
    let expected = Type::Union(vec![Type::Int, Type::String]);
    assert_eq!(parameters[0].1, expected);
    assert_eq!(return_type, &expected);
    assert!(!statements
        .iter()
        .any(|statement| matches!(statement, Node::TypeAliasDeclaration { .. })));
}

#[test]
fn rejects_recursive_type_aliases() {
    let program = ast::parse(
        r#"
        type A = B;
        type B = A;
        function main(): void {}
        "#,
    );
    let error = prepare_program(&program).unwrap_err();
    assert!(error.contains("cyclic type alias detected: A -> B -> A"));
}

#[test]
fn rejects_type_alias_bound_violations() {
    let program = ast::parse(
        r#"
        trait Writer {
            function write(mut self, value: int): int;
        }

        type WriterValue[T: Writer] = T | string;

        function main(): void {
            value: WriterValue[int] = 1;
        }
        "#,
    );
    let error = prepare_program(&program).unwrap_err();
    assert!(error.contains("type alias `WriterValue` requires `T` to implement trait `Writer`"));
}

#[test]
fn rejects_non_trait_intersection_members() {
    let program = ast::parse(
        r#"
        type Invalid = int & string;

        function main(): void {
            value: Invalid;
        }
        "#,
    );
    let error = prepare_program(&program).unwrap_err();
    assert!(error.contains("intersection member `int` is not a trait"));
}

#[test]
fn accepts_upper_subtype_bound_for_conforming_type() {
    let statements = prepared_statements(
        r#"
        trait Animal {
            function sound(self): int;
        }

        struct Dog {}

        conform Animal for Dog {
            function sound(self): int { return 1; }
        }

        function identity[T <: Animal](value: T): T {
            return value;
        }

        function main(): void {
            dog: Dog = identity(Dog {});
        }
        "#,
    );

    assert!(statements.iter().any(|statement| matches!(
        statement,
        Node::FunctionDeclaration { name, .. } if name == "identity__Dog"
    )));
}

#[test]
fn accepts_trait_intersection_as_upper_bound() {
    let statements = prepared_statements(
        r#"
        trait Animal {}
        trait Serializable {}

        struct Dog {}
        conform Animal for Dog {}
        conform Serializable for Dog {}

        function identity[T <: Animal & Serializable](value: T): T {
            return value;
        }

        function main(): void {
            dog: Dog = identity(Dog {});
        }
        "#,
    );

    assert!(statements.iter().any(|statement| matches!(
        statement,
        Node::FunctionDeclaration { name, .. } if name == "identity__Dog"
    )));
}

#[test]
fn rejects_upper_subtype_bound_violation() {
    let program = ast::parse(
        r#"
        trait Animal {}

        function identity[T <: Animal](value: T): T {
            return value;
        }

        function main(): void {
            value: int = identity(1);
        }
        "#,
    );
    let error = prepare_program(&program).unwrap_err();
    assert!(error.contains(
        "generic function `identity` requires `T` to be a subtype of `Animal`, but found `int`"
    ));
}

#[test]
fn numeric_conversion_is_not_subtyping() {
    let program = ast::parse(
        r#"
        function identity[T <: long](value: T): T {
            return value;
        }

        function main(): void {
            value: int = identity(1);
        }
        "#,
    );
    let error = prepare_program(&program).unwrap_err();
    assert!(error.contains(
        "generic function `identity` requires `T` to be a subtype of `long`, but found `int`"
    ));
}

#[test]
fn checks_subtype_bounds_on_generic_type_aliases() {
    let program = ast::parse(
        r#"
        trait Animal {}
        type Pet[T <: Animal] = T | string;

        function main(): void {
            pet: Pet[int] = 1;
        }
        "#,
    );
    let error = prepare_program(&program).unwrap_err();
    assert!(error
        .contains("type alias `Pet` requires `T` to be a subtype of `Animal`, but found `int`"));
}

#[test]
fn rejects_lower_subtype_bound_violation_for_explicit_argument() {
    let program = ast::parse(
        r#"
        struct Dog {}
        struct Cat {}

        function keep[T >: Dog](value: T): T {
            return value;
        }

        function main(): void {
            cat: Cat = keep[Cat](Cat {});
        }
        "#,
    );
    let error = prepare_program(&program).unwrap_err();
    assert!(error.contains(
        "generic function `keep` requires `T` to be a supertype of `Dog`, but found `Cat`"
    ));
}

#[test]
fn widens_direct_generic_lower_constraints_to_union() {
    let statements = prepared_statements(
        r#"
        struct Dog {}
        struct Cat {}

        function choose[T](left: T, right: T, first: bool): T {
            if (first) {
                return left;
            }
            return right;
        }

        function main(): void {
            pet: Dog | Cat = choose(Dog {}, Cat {}, true);
        }
        "#,
    );

    assert!(statements.iter().any(|statement| matches!(
        statement,
        Node::FunctionDeclaration { name, .. }
            if name == "choose__union_Cat_or_Dog"
    )));
}

#[test]
fn inferred_lower_bound_can_reference_another_type_parameter() {
    let statements = prepared_statements(
        r#"
        struct Dog {}
        struct Cat {}

        function choose[A, B >: A](left: A, right: B, first: bool): B {
            if (first) {
                return left;
            }
            return right;
        }

        function main(): void {
            pet: Dog | Cat = choose(Dog {}, Cat {}, true);
        }
        "#,
    );

    assert!(statements.iter().any(|statement| matches!(
        statement,
        Node::FunctionDeclaration { name, .. }
            if name == "choose__Dog__union_Cat_or_Dog"
    )));
}
