//! Owns source text and translates compact spans into file/line locations.

use crate::ids::FileId;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Span {
    pub file: FileId,
    pub start: u32,
    pub end: u32,
}

impl Span {
    pub fn new(file: FileId, start: usize, end: usize) -> Option<Self> {
        if start > end {
            return None;
        }
        Some(Self {
            file,
            start: u32::try_from(start).ok()?,
            end: u32::try_from(end).ok()?,
        })
    }
}

#[derive(Debug)]
pub struct SourceFile {
    path: PathBuf,
    text: String,
    line_starts: Vec<u32>,
}

impl SourceFile {
    fn new(path: PathBuf, text: String) -> Result<Self, SourceMapError> {
        if text.len() > u32::MAX as usize {
            return Err(SourceMapError::FileTooLarge(path));
        }

        let mut line_starts = vec![0];
        for (offset, byte) in text.bytes().enumerate() {
            if byte == b'\n' {
                line_starts.push((offset + 1) as u32);
            }
        }

        Ok(Self {
            path,
            text,
            line_starts,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn line_column(&self, offset: u32) -> Option<LineColumn> {
        if offset as usize > self.text.len() {
            return None;
        }

        let line_index = self.line_starts.partition_point(|start| *start <= offset) - 1;
        let line_start = self.line_starts[line_index];
        Some(LineColumn {
            line: line_index + 1,
            column: (offset - line_start) as usize + 1,
        })
    }
}

#[derive(Debug, Default)]
pub struct SourceMap {
    files: Vec<SourceFile>,
}

impl SourceMap {
    pub fn add_file(
        &mut self,
        path: impl Into<PathBuf>,
        text: impl Into<String>,
    ) -> Result<FileId, SourceMapError> {
        let path = path.into();
        let raw_id = u32::try_from(self.files.len()).map_err(|_| SourceMapError::TooManyFiles)?;
        let file = SourceFile::new(path, text.into())?;
        self.files.push(file);
        Ok(FileId::new(raw_id))
    }

    pub fn file(&self, id: FileId) -> Option<&SourceFile> {
        self.files.get(id.index())
    }

    pub fn source(&self, id: FileId) -> Option<&str> {
        self.file(id).map(SourceFile::text)
    }

    pub fn span_text(&self, span: Span) -> Option<&str> {
        let file = self.file(span.file)?;
        file.text().get(span.start as usize..span.end as usize)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LineColumn {
    pub line: usize,
    pub column: usize,
}

#[derive(Debug)]
pub enum SourceMapError {
    FileTooLarge(PathBuf),
    TooManyFiles,
}

impl std::fmt::Display for SourceMapError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FileTooLarge(path) => write!(
                formatter,
                "source file `{}` exceeds Skunk's 4 GiB source limit",
                path.display()
            ),
            Self::TooManyFiles => write!(formatter, "program contains too many source files"),
        }
    }
}

impl std::error::Error for SourceMapError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_offsets_to_one_based_lines_and_columns() {
        let mut sources = SourceMap::default();
        let file = sources.add_file("geometry.skunk", "one\ntwo\n").unwrap();
        let source = sources.file(file).unwrap();

        assert_eq!(
            source.line_column(5),
            Some(LineColumn { line: 2, column: 2 })
        );
    }
}
