//! Structured compiler diagnostics independent of any rendering frontend.

use crate::source_map::{SourceMap, Span};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Severity {
    Error,
    Warning,
    Note,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Label {
    pub span: Span,
    pub message: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Diagnostic {
    pub severity: Severity,
    pub code: Option<&'static str>,
    pub message: String,
    pub primary: Option<Box<Label>>,
    pub secondary: Vec<Label>,
    pub notes: Vec<String>,
}

impl Diagnostic {
    pub fn error(message: impl Into<String>) -> Self {
        Self {
            severity: Severity::Error,
            code: None,
            message: message.into(),
            primary: None,
            secondary: Vec::new(),
            notes: Vec::new(),
        }
    }

    pub fn at(mut self, span: Span) -> Self {
        self.primary = Some(Box::new(Label {
            span,
            message: None,
        }));
        self
    }

    pub fn with_code(mut self, code: &'static str) -> Self {
        self.code = Some(code);
        self
    }

    pub fn label(mut self, span: Span, message: impl Into<String>) -> Self {
        self.secondary.push(Label {
            span,
            message: Some(message.into()),
        });
        self
    }

    pub fn note(mut self, note: impl Into<String>) -> Self {
        self.notes.push(note.into());
        self
    }

    /// Renders this diagnostic with a stable file/line/column location when
    /// its source file is still available.
    pub fn render(&self, sources: &SourceMap) -> String {
        let severity = match self.severity {
            Severity::Error => "error",
            Severity::Warning => "warning",
            Severity::Note => "note",
        };
        let code = self
            .code
            .map(|code| format!("[{code}]"))
            .unwrap_or_default();
        let mut output = String::new();
        if let Some(primary) = &self.primary {
            if let Some(file) = sources.file(primary.span.file) {
                if let Some(location) = file.line_column(primary.span.start) {
                    output.push_str(&format!(
                        "{}:{}:{}: {severity}{code}: {}",
                        file.path().display(),
                        location.line,
                        location.column,
                        self.message
                    ));
                }
            }
        }
        if output.is_empty() {
            output.push_str(&format!("{severity}{code}: {}", self.message));
        }
        if let Some(message) = self
            .primary
            .as_ref()
            .and_then(|label| label.message.as_deref())
        {
            output.push_str(&format!("\n  = {message}"));
        }
        for label in &self.secondary {
            if let Some(file) = sources.file(label.span.file) {
                if let Some(location) = file.line_column(label.span.start) {
                    output.push_str(&format!(
                        "\n  = {}:{}:{}: {}",
                        file.path().display(),
                        location.line,
                        location.column,
                        label.message.as_deref().unwrap_or("related location")
                    ));
                }
            }
        }
        for note in &self.notes {
            output.push_str(&format!("\n  = note: {note}"));
        }
        output
    }
}

impl std::fmt::Display for Diagnostic {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(code) = self.code {
            write!(formatter, "[{code}] ")?;
        }
        write!(formatter, "{}", self.message)
    }
}

impl std::error::Error for Diagnostic {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_source_location_and_code() {
        let mut sources = SourceMap::default();
        let file = sources.add_file("sample.skunk", "one\ntwo\n").unwrap();
        let diagnostic = Diagnostic::error("bad value")
            .with_code("E1234")
            .at(Span::new(file, 5, 6).unwrap())
            .note("use another value");

        assert_eq!(
            diagnostic.render(&sources),
            "sample.skunk:2:2: error[E1234]: bad value\n  = note: use another value"
        );
    }
}
