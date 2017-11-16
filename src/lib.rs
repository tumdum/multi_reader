use std::io::{BufReader,Error,ErrorKind,Read,Result,Seek,SeekFrom};
use std::iter::once;

pub struct MultiRead<R> {
    readers: Vec<R>,
    ends: Vec<u64>,
    reader: usize,
    total_size: u64,
}

fn get_size<R: Seek>(r: &mut R) -> Result<u64> {
    let size = r.seek(SeekFrom::End(0))?;
    r.seek(SeekFrom::Start(0))?;
    Ok(size)
}

impl<R: Read + Seek> MultiRead<R> {
    pub fn new<T: IntoIterator<Item=R>>(rs: T) -> Result<MultiRead<R>> {
        let mut readers = vec![];
        let mut ends = vec![];
        let mut total_size = 0;
        for mut r in rs {
            let size = get_size(&mut r)?;
            if size == 0 {
                continue;
            }
            readers.push(r);
            total_size += size;
            ends.push(total_size);
        }
        let reader = 0;
        Ok(MultiRead{readers, ends, reader, total_size})
    }

    pub fn lines(mut self) -> LineResult<Lines<R>> {
        let mut boundries : Vec<Boundry> = vec![];
        {
            let offsets = once(&0).chain(self.ends.iter());
            for (r, o) in self.readers.iter_mut().zip(offsets) {
                let local_boundries : Vec<Boundry> = count_lines(r, *o as usize)?;
                let skip = match (boundries.last_mut(), local_boundries.first()) {
                    (Some(ref mut last), Some(next)) if (last.start + last.len) == next.start => {
                        last.len += next.len;
                        1
                    }
                    _ => 0
                };
                boundries.extend(local_boundries.into_iter().skip(skip));
            }
        }
        Ok(Lines{reader: self, boundries: boundries})
    }
}

impl<R: Read> Read for MultiRead<R> {
    fn read(&mut self, buf: &mut [u8]) -> Result<usize> {
        if self.reader >= self.readers.len() {
            return Ok(0)
        }
        match self.readers[self.reader].read(buf) {
            Ok(0) => {
                // NOTE: maybe remove recurence?
                self.reader+=1;
                self.read(buf)
            },
            ok @ Ok(_) => ok,
            err @ Err(_) => err,
        }
    }
}

impl<S: Seek> Seek for MultiRead<S> {
    fn seek(&mut self, pos: SeekFrom) -> Result<u64> {
        match pos {
            SeekFrom::End(n) => {
                let total_size = self.total_size as i64;
                self.seek(SeekFrom::Start((total_size + n) as u64))
            }
            SeekFrom::Start(n) => {
                if n > self.total_size {
                    return Err(Error::new(ErrorKind::InvalidInput, "seek past the end of reader"))
                }
                self.reader = 0;
                let mut m = n;
                let mut total = 0;
                for s in &self.ends {
                    let s = s - total;
                    if s >= m {
                        break;
                    }
                    m -= s;
                    total += s;
                    self.reader += 1;
                    // NOTE: seek on skipped reader?
                }
                for i in self.reader+1..self.readers.len() {
                    self.readers[i].seek(SeekFrom::Start(0))?;
                }
                self.readers[self.reader].seek(SeekFrom::Start(m))?;
                Ok(n)
            },
            SeekFrom::Current(n) => {
                let mut current = self.readers[self.reader].seek(SeekFrom::Current(0))?;
                if self.reader > 0 {
                    current += self.ends[self.reader-1]
                }
                let absolute_position = current as i64 + n;
                if absolute_position < 0 {
                    return Err(Error::new(ErrorKind::InvalidInput, "seek before beginning of raeder"))
                }
                self.seek(SeekFrom::Start(absolute_position as u64))
            }
        }
    }
}

#[derive(Debug)]
struct Boundry {
    start: usize,
    len: usize,
}

pub struct Lines<R> {
    reader: MultiRead<R>,
    boundries: Vec<Boundry>
}

fn count_lines<R: Read>(reader: &mut R, offset: usize) -> LineResult<Vec<Boundry>> {
    // TODO: is unicode important here?
    // TODO: run on threads for each chunk in multireader
    let mut position = offset;
    let mut start = offset;
    let mut boundries = vec![];
    let mut in_break = true;
    let reader = BufReader::new(reader);
    for b in reader.bytes() {
        match b {
            Ok(b'\n') | Ok(b'\r') => {
                if !in_break {
                    boundries.push(Boundry{start: start, len: position-start});
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
        boundries.push(Boundry{start:start, len: position - start});
    }
    Ok(boundries)
}

#[derive(Debug)]
pub enum LineError {
    OutOfBounds(usize),
    IoError(std::io::Error)
}

impl From<std::io::Error> for LineError {
    fn from(e: std::io::Error) -> LineError {
        LineError::IoError(e)
    }
}

pub type LineResult<T> = std::result::Result<T, LineError>;

impl <T: Read + Seek> Lines<T> {
    pub fn from_multiread(r: MultiRead<T>) -> LineResult<Lines<T>> {
        r.lines()
    }

    pub fn len(&self) -> usize {
        self.boundries.len()
    }

    pub fn line(&mut self, line: usize) -> LineResult<Vec<u8>> {
        if line >= self.len() {
            return Err(LineError::OutOfBounds(line))
        }
        let ref boundry = self.boundries[line];
        self.reader.seek(SeekFrom::Start(boundry.start as u64))?;

        // TODO: no TryFrom<u32> for usize on stable
        // https://github.com/rust-lang/rust/issues/33417
        let mut buf = vec![0; boundry.len];
        self.reader.read_exact(&mut buf)?;
        Ok(buf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Cursor, Error, ErrorKind};

    const DEFAULT_SIZE : u64 = 100;

    struct ErrorReturningReader {
    }

    impl Read for ErrorReturningReader {
        fn read(&mut self, _buf: &mut [u8]) -> Result<usize> {
            Err(Error::new(ErrorKind::Other, "dummy"))
        }
    }

    impl Seek for ErrorReturningReader {
        fn seek(&mut self, _pos: SeekFrom) -> Result<u64> {
            Ok(DEFAULT_SIZE)
        }
    }

    struct FailingSeek{
        which: usize,
        current: usize
    }

    impl FailingSeek {
        fn new(which: usize) -> FailingSeek {
            FailingSeek{which: which, current: 0}
        }
    }

    impl Read for FailingSeek {
        fn read(&mut self, _buf: &mut [u8]) -> Result<usize> { Ok(0) }
    }

    impl Seek for FailingSeek {
        fn seek(&mut self, _pos: SeekFrom) -> Result<u64> {
            if self.which == self.current {
                self.current += 1;
                return Err(Error::new(ErrorKind::Other, "dummy"))
            }
            self.current += 1;
            Ok(DEFAULT_SIZE)
        }
    }

    enum Op {
        Seek(SeekFrom),
        Read
    }

    fn compare<A: Seek+Read, B: Seek+Read>(mut left: A, mut right: B, ops: &[Op]) {
        for op in ops {
            match *op {
                Op::Seek(s) => {
                    assert_eq!(left.seek(s).unwrap(), right.seek(s).unwrap())
                }
                Op::Read => {
                    let mut left_output = String::new();
                    let mut right_output = String::new();
                    left.read_to_string(&mut left_output).unwrap();
                    right.read_to_string(&mut right_output).unwrap();
                    assert_eq!(left_output, right_output);
                }
            }
        }
    }

    const FIRST : &'static str = "AAAAAAAAAAAAAAAAAA";
    const SECOND : &'static str = "BBBBBBBBBBBBB";
    const LAST : &'static str = "CCCCCCCCCCCCCCCCCCCCCCCCC";

    #[test]
    fn creation_from_failing_seeker() {
        assert!(!MultiRead::new(vec![FailingSeek::new(0)]).is_ok());
        assert!(!MultiRead::new(vec![FailingSeek::new(1)]).is_ok());
    }

    #[test]
    fn no_reader() {
        let mut sut = MultiRead::<Cursor<&str>>::new(vec![]).unwrap();
        let mut output = String::new();
        assert_eq!(0, sut.read_to_string(&mut output).unwrap());
        assert_eq!(0, output.len());
    }

    #[test]
    fn one_reader() {
        let input = "foo bar baz";
        let full = Cursor::new(&input);
        let mut sut = MultiRead::new(vec![full]).unwrap();
        let mut output = String::new();
        assert_eq!(input.len(), sut.read_to_string(&mut output).unwrap());
        assert_eq!(input, output);
    }

    #[test]
    fn two_readers() {
        let input = FIRST.to_owned() + SECOND;
        let mut sut = MultiRead::new(vec![Cursor::new(FIRST), Cursor::new(SECOND)]).unwrap();
        let mut output = String::new();
        assert_eq!(input.len(), sut.read_to_string(&mut output).unwrap());
        assert_eq!(input, output);
    }

    #[test]
    fn empty_readers_in_between() {
        let input = FIRST.to_owned() + SECOND;
        let mut sut = MultiRead::new(vec![
            Cursor::new(""), Cursor::new(""), Cursor::new(""),
            Cursor::new(FIRST), 
            Cursor::new(""), Cursor::new(""), Cursor::new(""),
            Cursor::new(SECOND),
            Cursor::new(""), Cursor::new(""), Cursor::new("")]).unwrap();
        let mut output = String::new();
        assert_eq!(input.len(), sut.read_to_string(&mut output).unwrap());
        assert_eq!(input, output);
    }

    #[test]
    fn error_propagation() {
        let mut sut = MultiRead::new(vec![ErrorReturningReader{}]).unwrap();
        let mut output = String::new();
        assert!(sut.read_to_string(&mut output).is_err());
        assert_eq!(0, output.len());
    }

    #[test]
    fn seek_to_end_should_return_sum_of_sizes() {
        let mut sut = MultiRead::new(vec![Cursor::new(FIRST), Cursor::new(SECOND)]).unwrap();
        assert_eq!((FIRST.len() + SECOND.len()) as u64, sut.seek(SeekFrom::End(0)).unwrap());

        let mut sut = MultiRead::new(vec![
            Cursor::new(""), Cursor::new(""), Cursor::new(""),
            Cursor::new(FIRST), 
            Cursor::new(""), Cursor::new(""), Cursor::new(""),
            Cursor::new(SECOND),
            Cursor::new(""), Cursor::new(""), Cursor::new("")]).unwrap();
        assert_eq!((FIRST.len() + SECOND.len()) as u64, sut.seek(SeekFrom::End(0)).unwrap());
    }

    #[test]
    fn seek_in_one_reader() {
        let left = "foo bar baz";
        let sut = MultiRead::new(vec![Cursor::new(left)]).unwrap();
        let expected = Cursor::new(left);

        compare(sut, expected, &[Op::Seek(SeekFrom::Start(4)), Op::Read]);
    }

    #[test]
    fn seek_should_propagate_failures() {
        let mut sut = MultiRead::new(vec![FailingSeek::new(2)]).unwrap();
        assert!(sut.seek(SeekFrom::Start(10)).is_err());
        let mut sut = MultiRead::new(vec![FailingSeek::new(5), FailingSeek::new(2)]).unwrap();
        assert!(sut.seek(SeekFrom::Start(DEFAULT_SIZE/2)).is_err());
        let mut sut = MultiRead::new(vec![FailingSeek::new(2)]).unwrap();
        assert!(sut.seek(SeekFrom::Current(10)).is_err());
    }

    #[test]
    fn seek_in_two_readers() {
        let sut = MultiRead::new(vec![Cursor::new(FIRST), Cursor::new(SECOND)]).unwrap();
        let expected = Cursor::new(FIRST.to_owned() + SECOND);
        compare(sut, expected, &[Op::Seek(SeekFrom::Start((FIRST.len() + "aa ".len()) as u64)), Op::Read]);
    }

    #[test]
    fn seek_twice() {
        let sut = MultiRead::new(vec![Cursor::new(FIRST), Cursor::new(SECOND)]).unwrap();
        let expected = Cursor::new(FIRST.to_owned() + SECOND);
        compare(sut, expected, &[
            Op::Seek(SeekFrom::Start((FIRST.len() + SECOND.len()) as u64)),
            Op::Seek(SeekFrom::Start((FIRST.len() + SECOND.len()/2) as u64)),
            Op::Read]);
    }

    #[test]
    fn any_seek_from_start() {
        let total = FIRST.to_owned() + SECOND + LAST;
        for i in 0..total.len() {
            let sut = MultiRead::new(vec![Cursor::new(FIRST), Cursor::new(SECOND), Cursor::new(LAST)]).unwrap();
            let expected = Cursor::new(total.clone());

            compare(sut, expected, &[Op::Seek(SeekFrom::Start(i as u64)), Op::Read]);
        }
    }

    #[test]
    fn seek_outside_of_bounds() {
        let total = FIRST.to_owned() + SECOND + LAST;
        let mut sut = MultiRead::new(vec![Cursor::new(FIRST), Cursor::new(SECOND), Cursor::new(LAST)]).unwrap();
        assert!(sut.seek(SeekFrom::Start(total.len() as u64 + 1)).is_err());

        assert!(sut.seek(SeekFrom::End(1)).is_err());
        assert!(sut.seek(SeekFrom::End(-(total.len()as i64 + 1))).is_err());

        assert!(sut.seek(SeekFrom::Current(total.len() as i64 + 1)).is_err());
        assert!(sut.seek(SeekFrom::Current(-1)).is_err());

        let mut output = String::new();
        assert_eq!(total.len(), sut.read_to_string(&mut output).unwrap());
        assert_eq!(output, total);
    }

    #[test]
    fn seek_backwards() {
        let total = FIRST.to_owned() + SECOND + LAST;
        let sut = MultiRead::new(vec![Cursor::new(FIRST), Cursor::new(SECOND), Cursor::new(LAST)]).unwrap();
        let expected = Cursor::new(total.clone());

        compare(sut, expected, &[
            Op::Seek(SeekFrom::Start((FIRST.len()+SECOND.len()+LAST.len()/2) as u64)),
            Op::Seek(SeekFrom::Start((FIRST.len()+SECOND.len()/2) as u64)),
            Op::Read,
        ]);
    }

    #[test]
    fn any_seek_from_end() {
        let total = FIRST.to_owned() + SECOND + LAST;
        for i in 0..total.len() {
            let sut = MultiRead::new(vec![Cursor::new(FIRST), Cursor::new(SECOND), Cursor::new(LAST)]).unwrap();
            let expected = Cursor::new(total.clone());

            compare(sut, expected, &[Op::Seek(SeekFrom::End(-(i as i64))), Op::Read]);
        }
    }

    #[test]
    fn seen_from_current() {
        let total = FIRST.to_owned() + SECOND + LAST;
        for i in 1..total.len()-1 {
            let sut = MultiRead::new(vec![Cursor::new(FIRST), Cursor::new(SECOND), Cursor::new(LAST)]).unwrap();
            let expected = Cursor::new(total.clone());

            compare(sut, expected, &[
                Op::Seek(SeekFrom::Start(i as u64)),
                Op::Seek(SeekFrom::Current(1)), Op::Read, 
                Op::Seek(SeekFrom::Start(i as u64)),
                Op::Seek(SeekFrom::Current(-1)), Op::Read]);
        }
    }

    fn lines_from_string(s: &str) -> Lines<Cursor<&str>> {
        let full = Cursor::new(s);
        let sut = MultiRead::new(vec![full]).unwrap();
        Lines::from_multiread(sut).unwrap()
    }

    #[test]
    fn size_of_lines_from_empty() {
        let lines = lines_from_string("");
        assert_eq!(0, lines.len());
    }

    #[test]
    fn size_of_lines_from_one_line() {
        let lines = lines_from_string("one short line");
        assert_eq!(1, lines.len());
    }

    #[test]
    fn size_of_lines_from_one_line_split_across_separate_readers() {
        let multiread = MultiRead::new(vec![
            Cursor::new(FIRST), 
            Cursor::new(SECOND),
            Cursor::new(LAST)]).unwrap();
        let lines = Lines::from_multiread(multiread).unwrap();
        assert_eq!(1, lines.len());
    }

    #[test]
    fn size_of_lines_from_two_lines_separated_by_multiple_newlines() {
        let lines = lines_from_string("aa\n\r\n\nbb");
        assert_eq!(2, lines.len());
    }

    #[test]
    fn line_from_one_line_lines() {
        let mut lines = lines_from_string(FIRST);
        assert_eq!(FIRST, String::from_utf8(lines.line(0).unwrap()).unwrap());
    }

    #[test]
    fn line_reading() {
        let multiread = MultiRead::new(vec![
            Cursor::new("\n\r\r\n"),
            Cursor::new(FIRST), 
            Cursor::new("\n\n\n\n"),
            Cursor::new(SECOND),
            Cursor::new("\r\n\n\r\n"), 
            Cursor::new(LAST),
            Cursor::new("\r\r\n\n"),
            Cursor::new("foo "), Cursor::new("bar "), Cursor::new("baz")]).unwrap();
        let mut lines = Lines::from_multiread(multiread).unwrap();
        assert_eq!(4, lines.len());
        assert_eq!(FIRST, String::from_utf8(lines.line(0).unwrap()).unwrap());
        assert_eq!(SECOND, String::from_utf8(lines.line(1).unwrap()).unwrap());
        assert_eq!(LAST, String::from_utf8(lines.line(2).unwrap()).unwrap());
        assert_eq!("foo bar baz", String::from_utf8(lines.line(3).unwrap()).unwrap());
    }

    #[test]
    fn line_reading_out_of_bounds() {
        let multiread = MultiRead::new(vec![
            Cursor::new(FIRST), 
            Cursor::new(SECOND),
            Cursor::new(LAST)]).unwrap();
        let mut lines = Lines::from_multiread(multiread).unwrap();
        assert!(lines.line(3).is_err());
        assert_eq!(std::mem::align_of::<Boundry>(), 8);
        assert_eq!(std::mem::size_of::<Boundry>(), 16);
    }

    #[test]
    fn lines_construction_with_failing_io() {
        let multiread = MultiRead::new(vec![ErrorReturningReader{}]).unwrap();
        assert!(Lines::from_multiread(multiread).is_err());
    }
}
