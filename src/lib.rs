pub mod lines;

extern crate rayon;
extern crate rand;

use std::io::{Error,ErrorKind,Read,Result,Seek,SeekFrom};
use std::cell::RefCell;
use std::ops::DerefMut;
use std::borrow::Borrow;
use lines::*;

use rayon::prelude::*;

enum Task {
    Segment(Vec<Line>),
    Copied(Vec<u8>),
}

pub struct MultiRead<R> {
    readers: Vec<RefCell<R>>,
    ends: Vec<u64>,
    reader: usize,
    total_size: u64,
}

fn get_size<R: Seek>(r: &mut R) -> Result<u64> {
    let size = r.seek(SeekFrom::End(0))?;
    r.seek(SeekFrom::Start(0))?;
    Ok(size)
}

fn read_boundry<T: Read + Seek>(reader: &mut T, b: &Line) -> LineResult<Vec<u8>> {
    if let &Line::Boundary{start, len} = b {
        reader.seek(SeekFrom::Start(start as u64))?;
        // TODO: no TryFrom<u32> for usize on stable
        // https://github.com/rust-lang/rust/issues/33417
        let mut buf = vec![0; len as usize];
        reader.read_exact(&mut buf)?;
        return Ok(buf);
    }
    unreachable!();
}

impl<R: Read + Seek + Send> MultiRead<R> {
    pub fn new<T: IntoIterator<Item=R>>(rs: T) -> Result<MultiRead<R>> {
        let mut readers = vec![];
        let mut ends = vec![];
        let mut total_size = 0;
        for mut r in rs {
            let size = get_size(&mut r)?;
            if size == 0 {
                continue;
            }
            readers.push(RefCell::new(r));
            total_size += size;
            ends.push(total_size);
        }
        let reader = 0;
        debug_assert!(readers.len() == ends.len());
        Ok(MultiRead{readers, ends, reader, total_size})
    }

    fn read_line(&self, line: &Line) -> LineResult<Vec<u8>> {
        match line {
            &Line::Copy(ref c) => return Ok(*c.clone()),
            &Line::Boundary{start, len} => {
                let mut reader_index = 0;
                while self.ends.len() >= reader_index && self.ends[reader_index] < start + len as u64 {
                    reader_index += 1;
                }
                let mut reader = self.readers[reader_index].borrow_mut();
                let reader_start = if reader_index == 0 { 0 } else { self.ends[reader_index-1] };
                let start = start - reader_start;
                return Ok(read_boundry(reader.deref_mut(), &Line::Boundary{start, len})?);
            }
        }
    }

    pub fn lines(mut self) -> LineResult<LinesIndex<R>> {
        let mut boundaries : Vec<Line> = vec![];
        {
            let mut local_boundaries;
            {
                let offsets = rayon::iter::once(&0).chain(self.ends.par_iter());
                let tmp : std::result::Result<Vec<Vec<Line>>, _> = self.readers
                    .par_iter_mut()
                    .zip(offsets)
                    .map(|pair| find_lines_boundaries(pair.0.borrow_mut().deref_mut(), *pair.1))
                    .collect();
                local_boundaries = tmp?;
            }

            // None when last line in previous buffer did not end at the end of
            // buffer.
            let mut last_boundry = None;
            for (mut b, i) in local_boundaries.drain(..).zip(0..) {
                let mut start_from = 0;
                if let Some(Line::Boundary{start, len}) = last_boundry {
                    if let Some(&Line::Boundary{start: new_start, len: new_len}) = b.first() {
                        if start + len as u64 == new_start {
/*
 * Line from previous buffer ended where the buffer ended and first line of 
 * current buffer starts at the beginning. Which means that those two lines are
 * in fact one line that crosses buffers foundries.
 */
                            last_boundry = Some(Line::Boundary{start: start, len: len + new_len});
                            if b.len() == 1 {
                                continue
                            }
                            start_from = 1;
                        }
                    }
                    if let boundary @ Line::Boundary{..} = last_boundry.unwrap() {
                        boundaries.push(Line::Copy(Box::new(read_boundry(&mut self,&boundary)?)));
                    } else {
                        unreachable!();
                    }
                }

                let mut end = b.len();
                last_boundry = None;
                if let Some(&Line::Boundary{start, len}) = b.last() {
                    if start+len as u64  >= self.ends[i] {
/*
 * The last boundary of line reaches end of buffer. So we are unable to determine
 * if it is real end of line or maybe the line continues in the next reader.  
 * This means that we can't insert it as is and we need to see beginning of the
 * next buffer.
 */
                        end -= 1;
                        last_boundry = Some(Line::Boundary{start, len});
                    }
                }
                boundaries.extend(b.drain(start_from..end));
            }
            if let Some(boundary @ Line::Boundary{..}) = last_boundry {
                boundaries.push(Line::Copy(Box::new(read_boundry(&mut self, &boundary)?)));
            }
        }
        Ok(LinesIndex::new(self, boundaries))
    }

    pub fn filter_which<F>(&self, f: &F, lines: &[Line]) -> LineResult<Vec<usize>> 
        where F: Fn(&[u8]) -> bool {
        Ok(self.map(f, lines)?.into_iter().enumerate().filter(|&(_,b)| b).map(|(v,_)| v).collect())
    }

    fn map_aux<F, Ret>(&self, f: &F, task: Task) -> LineResult<Vec<Ret>> where F: Fn(&[u8]) -> Ret {
        match task {
            Task::Segment(ref lines) => {
                Ok(lines.iter().map(|l| f(&self.read_line(l).unwrap())).collect())
            },
            Task::Copied(ref bytes) => {
                Ok(vec![f(bytes)])
            }
        }
    }

    pub fn map<F, Ret>(&self, f: &F, lines: &[Line]) -> LineResult<Vec<Ret>> 
        where F: Fn(&[u8]) -> Ret {
        let tasks = self.split_by_readers(lines);
        let mapped_segments : LineResult<Vec<Vec<Ret>>> = tasks.into_iter().map(|s| self.map_aux(f, s)).collect();
        Ok(mapped_segments?.into_iter().flat_map(|v| v.into_iter()).collect())
    }

    fn split_by_readers(&self, lines: &[Line]) -> Vec<Task> {
        let mut current_reader = 0;
        let mut current_segment = vec![];
        let mut tasks : Vec<Task> = vec![];

        for line in lines {
            match line {
                &Line::Boundary{start, len} => {
                    let end = start + len as u64;
                    if end <= self.ends[current_reader] {
                        current_segment.push(Line::Boundary{start, len});
                    } else {
                        tasks.push(Task::Segment(current_segment));
                        current_segment = vec![Line::Boundary{start, len}];
                        current_reader += 1;
                    }
                },
                &Line::Copy(ref content) => {
                    tasks.push(Task::Segment(current_segment));
                    let c : &Vec<u8> = content.borrow();
                    tasks.push(Task::Copied(c.clone()));
                    current_segment = vec![];
                    current_reader += 1;
                }
            }
        }
        tasks.push(Task::Segment(current_segment));
        tasks
    }
    
}

impl<R: Read> Read for MultiRead<R> {
    fn read(&mut self, buf: &mut [u8]) -> Result<usize> {
        if self.reader >= self.readers.len() {
            return Ok(0)
        }
        let val = self.readers[self.reader].borrow_mut().read(buf);
        match val {
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
                    self.readers[i].borrow_mut().seek(SeekFrom::Start(0))?;
                }
                self.readers[self.reader].borrow_mut().seek(SeekFrom::Start(m))?;
                Ok(n)
            },
            SeekFrom::Current(n) => {
                let mut current = self.readers[self.reader].borrow_mut().seek(SeekFrom::Current(0))?;
                if self.reader > 0 {
                    current += self.ends[self.reader-1]
                }
                let absolute_position = current as i64 + n;
                if absolute_position < 0 {
                    return Err(Error::new(ErrorKind::InvalidInput, "seek before beginning of reader"))
                }
                self.seek(SeekFrom::Start(absolute_position as u64))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Cursor, Error, ErrorKind};
    use rand::Rng;

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

    fn lines_from_string(s: &str) -> LinesIndex<Cursor<&str>> {
        let full = Cursor::new(s);
        let sut = MultiRead::new(vec![full]).unwrap();
        LinesIndex::from_multiread(sut).unwrap()
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
        let lines = LinesIndex::from_multiread(multiread).unwrap();
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
            Cursor::new("foo "), Cursor::new("bar "), Cursor::new("baz"),
            Cursor::new("\ntest\n")
        ]).unwrap();
        let mut lines = LinesIndex::from_multiread(multiread).unwrap();
        assert_eq!(5, lines.len());
        assert_eq!(FIRST, String::from_utf8(lines.line(0).unwrap()).unwrap());
        assert_eq!(SECOND, String::from_utf8(lines.line(1).unwrap()).unwrap());
        assert_eq!(LAST, String::from_utf8(lines.line(2).unwrap()).unwrap());
        assert_eq!("foo bar baz", String::from_utf8(lines.line(3).unwrap()).unwrap());
        assert_eq!("test", String::from_utf8(lines.line(4).unwrap()).unwrap());
    }

    #[test]
    fn random_split_line_reading() {
        for _ in 0..1000 {
            let mut rng = rand::thread_rng();
            let input = "aaaa\nb\n\nccc\n\rdddddddd\neeee\nf\ng\rh\n\niiiiiii\r\rjjj\rkkkkkkkkkkkkkkk";
            let split = rng.gen_range(0, input.len());
            let (x,y) = input.split_at(split);
            if x.len() == 0 {
                continue
            }
            let split_x = rng.gen_range(0, x.len());
            let (a,b) = x.split_at(split_x);
            if y.len() == 0 {
                continue
            }
            let split_y = rng.gen_range(0, y.len());
            let (c,d) = y.split_at(split_y);
            let multiread = MultiRead::new(vec![
                Cursor::new(a.to_owned().into_bytes()),
                Cursor::new(b.to_owned().into_bytes()),
                Cursor::new(c.to_owned().into_bytes()),
                Cursor::new(d.to_owned().into_bytes())
            ]).unwrap();
            let mut lines = LinesIndex::from_multiread(multiread).unwrap();

            assert_eq!("aaaa",              String::from_utf8(lines.line(0).unwrap()).unwrap());
            assert_eq!("b",                 String::from_utf8(lines.line(1).unwrap()).unwrap());
            assert_eq!("ccc",               String::from_utf8(lines.line(2).unwrap()).unwrap());
            assert_eq!("dddddddd",          String::from_utf8(lines.line(3).unwrap()).unwrap());
            assert_eq!("eeee",              String::from_utf8(lines.line(4).unwrap()).unwrap());
            assert_eq!("f",                 String::from_utf8(lines.line(5).unwrap()).unwrap());
            assert_eq!("g",                 String::from_utf8(lines.line(6).unwrap()).unwrap());
            assert_eq!("h",                 String::from_utf8(lines.line(7).unwrap()).unwrap());
            assert_eq!("iiiiiii",           String::from_utf8(lines.line(8).unwrap()).unwrap());
            assert_eq!("jjj",               String::from_utf8(lines.line(9).unwrap()).unwrap());
            assert_eq!("kkkkkkkkkkkkkkk",   String::from_utf8(lines.line(10).unwrap()).unwrap());

            assert_eq!(11, lines.len());

            let lengths = lines.map(|s| s.len()).unwrap();
            assert_eq!(4, lengths[0]);
            assert_eq!(1, lengths[1]);
            assert_eq!(3, lengths[2]);
            assert_eq!(8, lengths[3]);
            assert_eq!(4, lengths[4]);
            assert_eq!(1, lengths[5]);
            assert_eq!(1, lengths[6]);
            assert_eq!(1, lengths[7]);
            assert_eq!(7, lengths[8]);
            assert_eq!(3, lengths[9]);
            assert_eq!(15, lengths[10]);

            assert_eq!(11, lengths.len());
        }
    }

    #[test]
    fn line_reading_out_of_bounds() {
        let multiread = MultiRead::new(vec![
            Cursor::new(FIRST), 
            Cursor::new(SECOND),
            Cursor::new(LAST)]).unwrap();
        let mut lines = LinesIndex::from_multiread(multiread).unwrap();
        assert!(lines.line(3).is_err());
        assert_eq!(std::mem::align_of::<Line>(), 8);
        assert_eq!(std::mem::size_of::<Line>(), 16);
    }

    #[test]
    fn lines_construction_with_failing_io() {
        let multiread = MultiRead::new(vec![ErrorReturningReader{}]).unwrap();
        assert!(LinesIndex::from_multiread(multiread).is_err());
    }
    
    #[test]
    fn lines_mapping() {
        let multiread = MultiRead::new(vec![
            Cursor::new("\n\r\r\n"),
            Cursor::new(FIRST), 
            Cursor::new("\n\n\n\n"),
            Cursor::new(SECOND),
            Cursor::new("\r\n\n\r\n"), 
            Cursor::new(LAST),
            Cursor::new("\r\r\n\n"),
            Cursor::new("foo "), Cursor::new("bar "), Cursor::new("baz"),
            Cursor::new("\ntest\n")
        ]).unwrap();
        let mut lines = LinesIndex::from_multiread(multiread).unwrap();
        let lengths = lines.map(|s| s.len()).unwrap();

        assert_eq!(FIRST.len(),     lengths[0]);
        assert_eq!(SECOND.len(),    lengths[1]);
        assert_eq!(LAST.len(),      lengths[2]);
        assert_eq!(11,              lengths[3]);
        assert_eq!(4,               lengths[4]);

        assert_eq!(5, lengths.len());

        let positive = lines.filter_which(|s| s.len() % 2 == 0).unwrap();
        assert_eq!(2, positive.len());
        assert_eq!(0, positive[0]);
        assert_eq!(4, positive[1]);
    }
}
