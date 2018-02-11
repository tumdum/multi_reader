use std::io::{BufReader,Error,Read};
use std;
use super::*;

#[derive(Debug,Clone)]
pub enum Line {
    Boundary { start: u64, len: u32 },
    Copy(Box<Vec<u8>>)
}

#[derive(Debug)]
pub enum LineError {
    OutOfBounds(usize),
    IoError(Error)
}

pub type LineResult<T> = std::result::Result<T, LineError>;

pub fn find_lines_boundaries<R: Read>(reader: &mut R, offset: u64) -> LineResult<Vec<Line>> {
    // TODO: is unicode important here?
    // TODO: run on threads for each chunk in multireader
    let mut position = offset;
    let mut start = offset;
    let mut boundaries = vec![];
    let mut in_break = true;
    let reader = BufReader::new(reader);
    for b in reader.bytes() {
        match b {
            Ok(b'\n') | Ok(b'\r') => {
                if !in_break {
                    boundaries.push(Line::Boundary{start: start, len: (position-start) as u32});
                    in_break = true;
                }
            }
            Err(e) => return Err(LineError::IoError(e)),
            Ok(_) => {
                if in_break {
                    in_break = false;
                    start = position;
                }
            }
        };
        position += 1;
    }
    if position > offset && !in_break {
        boundaries.push(Line::Boundary{start:start, len: (position - start) as u32});
    }
    Ok(boundaries)
}

pub struct LinesIndex<R> {
    reader: MultiRead<R>,
    boundaries: Vec<Line>
}

impl<R: Read + Seek + Send> LinesIndex<R> {
    pub fn new(reader: MultiRead<R>, boundaries: Vec<Line>) -> LinesIndex<R> {
        LinesIndex{reader, boundaries}
    }

    pub fn filter_which<F>(&mut self, f: F) -> LineResult<Vec<usize>> where F: Fn(&[u8]) -> bool + Send + Sync {
        self.reader.filter_which(&f, &self.boundaries)
    }

    pub fn map<F, Ret>(&mut self, f: F) -> LineResult<Vec<Ret>> where F: Fn(&[u8]) -> Ret + Send + Sync, Ret : Send {
        self.reader.map(&f, &self.boundaries)
    }
}

impl From<std::io::Error> for LineError {
    fn from(e: std::io::Error) -> LineError {
        LineError::IoError(e)
    }
}

impl <T: Read + Seek + Send> LinesIndex<T> {
    pub fn from_multiread(r: MultiRead<T>) -> LineResult<LinesIndex<T>> {
        r.lines()
    }

    pub fn len(&self) -> usize {
        self.boundaries.len()
    }

    pub fn line(&mut self, line: usize) -> LineResult<Vec<u8>> {
        if line >= self.len() {
            return Err(LineError::OutOfBounds(line))
        }
        self.reader.read_line(&self.boundaries[line])
    }
}

