#[macro_use]
extern crate rocket;

use rocket::response::content::RawHtml;
use std::fs;
use std::path::Path;

#[get("/")]
fn home() -> Result<RawHtml<String>, std::io::Error> {
    let html = fs::read_to_string(Path::new("packages/webserver/templates/index.html"))?;
    Ok(RawHtml(html))
}

#[launch]
fn rocket() -> _ {
    rocket::build().mount("/", routes![home])
}
