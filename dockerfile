# Use an official Rust image as the base image
FROM rust:latest

# Set the working directory inside the container
WORKDIR /app

# Copy your project files into the container
COPY . .

WORKDIR /app/examples/stdlib

RUN rustup install nightly
RUN rustup default nightly
RUN cargo install --git https://github.com/a16z/jolt --force --bins jolt

CMD ["cargo", "run", "--release", "-p", "stdlib"]
