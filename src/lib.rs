use std::io::{Read,Result};

pub struct MultiRead<R> {
    readers: Vec<R>,
    reader: usize,
}

impl<R: Read + Clone> MultiRead<R> {
    pub fn new<T: AsRef<[R]>>(readers: T) -> MultiRead<R> {
        MultiRead{readers: readers.as_ref().to_vec(), reader: 0}
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Cursor, Error, ErrorKind};

    #[derive(Clone)]
    struct ErrorReturningReader {
    }

    impl Read for ErrorReturningReader {
        fn read(&mut self, _buf: &mut [u8]) -> Result<usize> {
            Err(Error::new(ErrorKind::Other, "dummy"))
        }
    }

    #[test]
    fn no_reader() {
        let mut sut = MultiRead::<Cursor<&str>>::new(vec![]);
        let mut output = String::new();
        assert_eq!(0, sut.read_to_string(&mut output).unwrap());
        assert_eq!(0, output.len());
    }

    #[test]
    fn one_reader() {
        let input = "foo bar baz";
        let full = Cursor::new(&input);
        let mut sut = MultiRead::new(&[full]);
        let mut output = String::new();
        assert_eq!(input.len(), sut.read_to_string(&mut output).unwrap());
        assert_eq!(input, output);
    }

    #[test]
    fn two_readers() {
        let left = "foo bar baz";
        let right = "aa bb cc";
        let input = left.to_owned() + right;
        let mut sut = MultiRead::new(&[Cursor::new(left), Cursor::new(right)]);
        let mut output = String::new();
        assert_eq!(input.len(), sut.read_to_string(&mut output).unwrap());
        assert_eq!(input, output);
    }

    #[test]
    fn empty_readers_in_between() {
        let left = "foo bar baz";
        let right = "aa bb cc";
        let input = left.to_owned() + right;
        let mut sut = MultiRead::new(&[
            Cursor::new(""), Cursor::new(""), Cursor::new(""),
            Cursor::new(left), 
            Cursor::new(""), Cursor::new(""), Cursor::new(""),
            Cursor::new(right),
            Cursor::new(""), Cursor::new(""), Cursor::new("")]);
        let mut output = String::new();
        assert_eq!(input.len(), sut.read_to_string(&mut output).unwrap());
        assert_eq!(input, output);
    }

    #[test]
    fn error_propagation() {
        let mut sut = MultiRead::new(&[ErrorReturningReader{}]);
        let mut output = String::new();
        assert!(sut.read_to_string(&mut output).is_err());
        assert_eq!(0, output.len());
    }

    /*
    #[test]
    fn error_handling() {

        let input = "foo bar baz";
        let full = Cursor::new(&input);
        let first : Box<Read> = Box::new(full);
        let second : Box<Read> = Box::new(ErrorReturningReader{});
        let mut sut = MultiRead::new(&[first, second]);
        let mut output = String::new();
        // TODO: check error
        assert_eq!(input, output);
    }
    */
}
