FROM rust:1.92

ENV ROCKET_ADDRESS=0.0.0.0
ENV ROCKET_PORT=8080

RUN rustup default nightly
RUN cargo install cargo-watch
EXPOSE 8080

WORKDIR /home/docker
COPY . .

WORKDIR /home/docker/packages/webserver
RUN cargo install

CMD ["cargo", "build"]
