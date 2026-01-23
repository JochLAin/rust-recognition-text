use indicatif::style::TemplateError;
use indicatif::{ProgressBar, ProgressStyle};

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Level {
    Dump = 0,
    Debug = 1,
    Info = 2,
    Message = 3,
    Warning = 4,
    Error = 5,
    Critical = 6,
}

pub struct Logger {
    pub progress_bar: ProgressBar,
    quiet: bool,
    level: Level,
}

impl Logger {
    pub fn new(nb_iter: usize, quiet: bool, level: Level) -> Result<Self, TemplateError> {
        let progress_bar = ProgressBar::new(nb_iter as u64);
        progress_bar.set_style(
            ProgressStyle::with_template(
                "[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ({percent}%)",
            )?
            .progress_chars("=>-"),
        );

        Ok(Self {
            progress_bar,
            quiet,
            level,
        })
    }

    pub fn is_quiet(&self) -> bool {
        self.quiet
    }

    pub fn iter(&self) {
        self.progress_bar.inc(1);
    }

    pub fn quiet(&mut self, quiet: bool) {
        self.quiet = quiet;
    }

    pub fn dump<I: AsRef<str>>(&self, msg: I) {
        self.print(msg, Level::Dump);
    }

    pub fn debug<I: AsRef<str>>(&self, msg: I) {
        self.print(msg, Level::Debug);
    }

    pub fn info<I: AsRef<str>>(&self, msg: I) {
        self.print(msg, Level::Info);
    }

    pub fn log<I: AsRef<str>>(&self, msg: I) {
        self.print(msg, Level::Message);
    }

    pub fn warn<I: AsRef<str>>(&self, msg: I) {
        self.print(msg, Level::Warning);
    }

    pub fn error<I: AsRef<str>>(&self, msg: I) {
        self.print(msg, Level::Error);
    }

    pub fn critical<I: AsRef<str>>(&self, msg: I) {
        self.print(msg, Level::Critical);
    }

    fn print<I: AsRef<str>>(&self, msg: I, level: Level) {
        if !self.quiet && level >= self.level {
            self.progress_bar.println(msg);
        }
    }
}
