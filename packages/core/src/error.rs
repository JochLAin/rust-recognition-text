use std::fmt::{Debug, Display};

pub enum Error {
    Candle(candle_core::Error),
    Msg(String),
    Std(Box<dyn std::error::Error>),
}

impl Error {
    fn print(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Candle(e) => write!(f, "{}", e),
            Error::Msg(e) => write!(f, "{}", e),
            Error::Std(e) => write!(f, "{}", e),
        }
    }
}

impl Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.print(f)
    }
}

impl Debug for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.print(f)
    }
}

impl From<candle_core::Error> for Error {
    fn from(e: candle_core::Error) -> Self {
        Error::Candle(e)
    }
}

impl From<Box<dyn std::error::Error>> for Error {
    fn from(e: Box<dyn std::error::Error>) -> Self {
        Error::Std(e)
    }
}

impl From<&str> for Error {
    fn from(e: &str) -> Self {
        Error::Msg(String::from(e))
    }
}

impl From<String> for Error {
    fn from(e: String) -> Self {
        Error::Msg(e)
    }
}

pub type Result<T> = std::result::Result<T, Error>;
