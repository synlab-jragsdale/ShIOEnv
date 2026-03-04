FROM alpine:3.20 AS build
RUN apk add --no-cache build-base musl-dev linux-headers
WORKDIR /src
COPY fc_agent.c ./
RUN gcc -static -O2 -s -o fc-agent fc_agent.c

FROM scratch AS export
COPY --from=build /src/fc-agent /
