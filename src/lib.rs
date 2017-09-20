use std::io::{Read,Result,Seek,SeekFrom};

pub struct MultiRead<R> {
    readers: Vec<R>,
    ends: Vec<u64>,
    reader: usize,
    total_size: u64,
}

fn get_size<R: Seek>(r: &mut R) -> Result<u64> {
    let size;
    match r.seek(SeekFrom::End(0)) {
        Ok(n) => size = n,
        e @ Err(_) => return e
    }
    if let e @ Err(_) = r.seek(SeekFrom::Start(0)) {
        panic!(e)
    }
    Ok(size)
}

impl<R: Read + Seek> MultiRead<R> {
    pub fn new<T: IntoIterator<Item=R>>(rs: T) -> Result<MultiRead<R>> {
        let mut readers = vec![];
        let mut ends = vec![];
        let mut running_total = 0;
        for mut r in rs {
            let size = get_size(&mut r)?;
            if size == 0 {
                continue;
            }
            readers.push(r);
            running_total += size;
            ends.push(running_total);
        }
        let total_size = running_total;
        let reader = 0;
        Ok(MultiRead{readers, ends, reader, total_size})
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
                for i in self.reader..self.readers.len() {
                    self.readers[i].seek(SeekFrom::Start(0)).unwrap();
                }
                self.readers[self.reader].seek(SeekFrom::Start(m)).unwrap();
                Ok(n)
            },
            SeekFrom::Current(n) => {
                let mut current = self.readers[self.reader].seek(SeekFrom::Current(0)).unwrap();
                if self.reader > 0 {
                    current += self.ends[self.reader-1]
                }
                self.seek(SeekFrom::Start((current as i64 +n) as u64))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Cursor, Error, ErrorKind};

    struct ErrorReturningReader {
    }

    impl Read for ErrorReturningReader {
        fn read(&mut self, _buf: &mut [u8]) -> Result<usize> {
            Err(Error::new(ErrorKind::Other, "dummy"))
        }
    }

    impl Seek for ErrorReturningReader {
        fn seek(&mut self, _pos: SeekFrom) -> Result<u64> {
            Ok(10)
        }
    }

    struct FailingSeek;

    impl Read for FailingSeek {
        fn read(&mut self, _buf: &mut [u8]) -> Result<usize> { Ok(0) }
    }

    impl Seek for FailingSeek {
        fn seek(&mut self, _pos: SeekFrom) -> Result<u64> {
            Err(Error::new(ErrorKind::Other, "dummy"))
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
        assert!(!MultiRead::new(vec![FailingSeek{}]).is_ok());
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
    fn seek_backwards2() {
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

    /*
    #[test]
    fn error_handling() {

        let input = "foo bar baz";
        let full = Cursor::new(&input);
        let first : Box<Read> = Box::new(full);
        let second : Box<Read> = Box::new(ErrorReturningReader{});
        let mut sut = MultiRead::new(vec![first, second]);
        let mut output = String::new();
        // TODO: check error
        assert_eq!(input, output);
    }
    */
}
