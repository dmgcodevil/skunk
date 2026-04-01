"use strict";

const vscode = require("vscode");

const KEYWORDS = [
    "module",
    "import",
    "export",
    "struct",
    "enum",
    "trait",
    "shape",
    "attach",
    "conform",
    "for",
    "function",
    "return",
    "if",
    "else",
    "match",
    "case",
    "unsafe",
    "const",
    "mut",
    "in",
    "true",
    "false",
    "self",
];

const BUILTIN_TYPES = [
    "byte",
    "short",
    "int",
    "long",
    "float",
    "double",
    "string",
    "boolean",
    "bool",
    "char",
    "void",
    "Allocator",
    "Arena",
];

const BUILTIN_SYMBOLS = ["System", "Memory"];

function activate(context) {
    context.subscriptions.push(
        vscode.languages.registerCompletionItemProvider(
            { language: "skunk" },
            createCompletionProvider(),
            ".",
            ":"
        ),
        vscode.languages.registerDocumentFormattingEditProvider(
            { language: "skunk" },
            createFormattingProvider()
        )
    );
}

function deactivate() {}

function createCompletionProvider() {
    return {
        provideCompletionItems(document) {
            const items = [];
            const seen = new Set();

            for (const keyword of KEYWORDS) {
                pushUniqueItem(seen, items, createKeywordItem(keyword));
            }

            for (const builtinType of BUILTIN_TYPES) {
                pushUniqueItem(seen, items, createBuiltinTypeItem(builtinType));
            }

            for (const builtinSymbol of BUILTIN_SYMBOLS) {
                pushUniqueItem(seen, items, createBuiltinSymbolItem(builtinSymbol));
            }

            for (const item of createSnippetItems()) {
                pushUniqueItem(seen, items, item);
            }

            for (const item of collectDocumentSymbolItems(document.getText())) {
                pushUniqueItem(seen, items, item);
            }

            return items;
        },
    };
}

function createFormattingProvider() {
    return {
        provideDocumentFormattingEdits(document, options) {
            const original = document.getText();
            const eol = document.eol === vscode.EndOfLine.CRLF ? "\r\n" : "\n";
            const formatted = formatSkunkDocument(original, options, eol);
            if (formatted === original) {
                return [];
            }

            const fullRange = new vscode.Range(
                document.positionAt(0),
                document.positionAt(original.length)
            );
            return [vscode.TextEdit.replace(fullRange, formatted)];
        },
    };
}

function createKeywordItem(keyword) {
    const item = new vscode.CompletionItem(keyword, vscode.CompletionItemKind.Keyword);
    item.detail = "Skunk keyword";
    return item;
}

function createBuiltinTypeItem(name) {
    const item = new vscode.CompletionItem(name, vscode.CompletionItemKind.Struct);
    item.detail = "Skunk builtin type";
    return item;
}

function createBuiltinSymbolItem(name) {
    const item = new vscode.CompletionItem(name, vscode.CompletionItemKind.Class);
    item.detail = "Skunk builtin symbol";
    return item;
}

function createSnippetItems() {
    return [
        createSnippetItem(
            "function",
            "Create a function declaration",
            "function ${1:name}(${2}): ${3:void} {\n\t$0\n}"
        ),
        createSnippetItem(
            "main",
            "Create a main entry point",
            "function main(): void {\n\t$0\n}"
        ),
        createSnippetItem(
            "struct",
            "Create a data-only struct",
            "struct ${1:Name} {\n\t${2:field}: ${3:int};\n}"
        ),
        createSnippetItem(
            "trait",
            "Create a trait declaration",
            "trait ${1:Name} {\n\tfunction ${2:method}(${3:mut self}): ${4:void};\n}"
        ),
        createSnippetItem(
            "shape",
            "Create a shape declaration",
            "shape ${1:Name} {\n\tfunction ${2:method}(${3:mut self}): ${4:void};\n}"
        ),
        createSnippetItem(
            "attach",
            "Create an attach block",
            "attach ${1:Type} {\n\tfunction ${2:name}(${3:self}): ${4:void} {\n\t\t$0\n\t}\n}"
        ),
        createSnippetItem(
            "conform",
            "Create a conform block",
            "conform ${1:Trait} for ${2:Type} {\n\tfunction ${3:name}(${4:self}): ${5:void} {\n\t\t$0\n\t}\n}"
        ),
        createSnippetItem(
            "match",
            "Create a match expression",
            "match (${1:value}) {\n\tcase ${2:Pattern}: {\n\t\t$0\n\t}\n}"
        ),
    ];
}

function createSnippetItem(label, detail, snippet) {
    const item = new vscode.CompletionItem(label, vscode.CompletionItemKind.Snippet);
    item.detail = detail;
    item.insertText = new vscode.SnippetString(snippet);
    item.documentation = new vscode.MarkdownString("Skunk snippet");
    return item;
}

function collectDocumentSymbolItems(text) {
    const specs = [
        {
            regex: /\bstruct\s+([A-Za-z_][A-Za-z0-9_]*)/g,
            kind: vscode.CompletionItemKind.Struct,
            detail: "Skunk struct",
        },
        {
            regex: /\benum\s+([A-Za-z_][A-Za-z0-9_]*)/g,
            kind: vscode.CompletionItemKind.Enum,
            detail: "Skunk enum",
        },
        {
            regex: /\btrait\s+([A-Za-z_][A-Za-z0-9_]*)/g,
            kind: vscode.CompletionItemKind.Interface,
            detail: "Skunk trait",
        },
        {
            regex: /\bshape\s+([A-Za-z_][A-Za-z0-9_]*)/g,
            kind: vscode.CompletionItemKind.Interface,
            detail: "Skunk shape",
        },
        {
            regex: /\bfunction\s+([A-Za-z_][A-Za-z0-9_]*)/g,
            kind: vscode.CompletionItemKind.Function,
            detail: "Skunk function",
        },
    ];

    const items = [];
    const seen = new Set();

    for (const spec of specs) {
        let match;
        while ((match = spec.regex.exec(text)) !== null) {
            const label = match[1];
            const key = `${spec.detail}:${label}`;
            if (seen.has(key)) {
                continue;
            }
            seen.add(key);
            const item = new vscode.CompletionItem(label, spec.kind);
            item.detail = spec.detail;
            items.push(item);
        }
    }

    return items;
}

function pushUniqueItem(seen, items, item) {
    const key = `${item.kind}:${item.label}`;
    if (seen.has(key)) {
        return;
    }
    seen.add(key);
    items.push(item);
}

function formatSkunkDocument(text, options, eol) {
    const indentUnit = options.insertSpaces ? " ".repeat(options.tabSize) : "\t";
    const lines = text.replace(/\r\n/g, "\n").split("\n");
    const output = [];

    let indentLevel = 0;
    let state = {
        inBlockComment: false,
        inString: false,
        inChar: false,
    };

    for (const originalLine of lines) {
        const trimmed = originalLine.trim();
        if (trimmed.length === 0) {
            output.push("");
            continue;
        }

        const analysis = analyzeLine(trimmed, state);
        const currentIndent = Math.max(indentLevel - analysis.leadingClosers, 0);
        output.push(indentUnit.repeat(currentIndent) + trimmed);
        indentLevel = Math.max(indentLevel + analysis.opens - analysis.closes, 0);
        state = analysis.state;
    }

    return output.join(eol);
}

function analyzeLine(line, state) {
    const nextState = { ...state };
    let opens = 0;
    let closes = 0;
    let leadingClosers = 0;
    let sawCode = false;

    for (let i = 0; i < line.length; i += 1) {
        const ch = line[i];
        const next = i + 1 < line.length ? line[i + 1] : "";

        if (nextState.inBlockComment) {
            if (ch === "*" && next === "/") {
                nextState.inBlockComment = false;
                i += 1;
            }
            continue;
        }

        if (!nextState.inString && !nextState.inChar) {
            if (ch === "/" && next === "*") {
                nextState.inBlockComment = true;
                i += 1;
                continue;
            }
            if (ch === "/" && next === "/") {
                break;
            }
        }

        if (!nextState.inChar && ch === "\"" && !isEscaped(line, i)) {
            nextState.inString = !nextState.inString;
            continue;
        }
        if (!nextState.inString && ch === "'" && !isEscaped(line, i)) {
            nextState.inChar = !nextState.inChar;
            continue;
        }

        if (nextState.inString || nextState.inChar) {
            continue;
        }

        if (!/\s/.test(ch)) {
            if (!sawCode && ch === "}") {
                leadingClosers += 1;
            } else if (!sawCode) {
                sawCode = true;
            }
        }

        if (ch === "{") {
            opens += 1;
            sawCode = true;
        } else if (ch === "}") {
            closes += 1;
            sawCode = true;
        }
    }

    return {
        opens,
        closes,
        leadingClosers,
        state: nextState,
    };
}

function isEscaped(text, index) {
    let backslashes = 0;
    for (let i = index - 1; i >= 0 && text[i] === "\\"; i -= 1) {
        backslashes += 1;
    }
    return backslashes % 2 === 1;
}

module.exports = {
    activate,
    deactivate,
};
