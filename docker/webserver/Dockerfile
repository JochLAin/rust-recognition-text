FROM rust:1.91

ENV ROCKET_ADDRESS=0.0.0.0
ENV ROCKET_PORT=8080

RUN rustup default nightly
RUN cargo install cargo-watch
EXPOSE 8080

WORKDIR /home/docker
COPY . .

RUN cargo install --path packages/webserver

CMD ["cargo", "build"]
