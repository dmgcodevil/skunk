# skunk

## Lexical Grammar

```
IDENTIFIER      ::= [a-zA-Z_][a-zA-Z0-9_]*

INTEGER         ::= [0-9]+
STRING_LITERAL  ::= '"' [^"]* '"'
BOOLEAN_LITERAL ::= 'true' | 'false'

KEYWORD         ::= 'int' | 'string' | 'boolean' | 'struct' | 'if' | 'else' | 'for' 
                  | 'match' | 'function' | 'return'

OPERATOR        ::= '+' | '-' | '*' | '/' | '=' | '==' | '!=' | '<' | '>' | '<=' | '>=' 
                  | '&&' | '||' | '!'
                  
PUNCTUATION     ::= '{' | '}' | '(' | ')' | '[' | ']' | ';' | ',' | ':'
WHITESPACE      ::= [ \t\n\r]+
COMMENT         ::= '//' [^\n]* | '/*' .* '*/'

```

## Grammar

```
program        ::= { statement }
statement      ::= var_decl | func_decl | struct_decl | control_flow | io | expression

var_decl       ::= "int" IDENTIFIER "=" expression ";" 
                 | "string" IDENTIFIER "=" expression ";" 
                 | "boolean" IDENTIFIER "=" expression ";"

func_decl      ::= "function" IDENTIFIER "(" [ param_list ] ")" "{" { statement } "}"
param_list     ::= IDENTIFIER ":" type { "," IDENTIFIER ":" type }

struct_decl    ::= "struct" IDENTIFIER "{" { struct_field_decl } "}"
struct_field_decl ::= IDENTIFIER ":" type ";"

control_flow   ::= "if" "(" expression ")" "{" { statement } "}"
                 | "for" "(" var_decl ";" expression ";" expression ")" "{" { statement } "}"
                 | "match" "(" expression ")" "{" { match_case } "}"
match_case     ::= "case" expression ":" "{" { statement } "}"

io             ::= "print" "(" expression ")" ";" 
                 | "input" "(" ")" ";"

expression     ::= literal | IDENTIFIER | binary_op | func_call
literal        ::= INTEGER | STRING_LITERAL | BOOLEAN_LITERAL
binary_op      ::= expression OPERATOR expression
func_call      ::= IDENTIFIER "(" [ expression { "," expression } ] ")"

type           ::= "int" | "string" | "boolean" | IDENTIFIER


```
