# TensorFold at TF_COMMIT plus patches/, built on NVIDIA's CUDA image, with libtfgrammar (structured output) from
# xgrammar XGRAMMAR_REF; the server and the library then go into a small CUDA base image. prepare.sh passes every
# argument from config.sh.
ARG BUILD_IMAGE=nvidia/cuda:13.3.1-devel-ubuntu24.04
ARG RUN_IMAGE=nvidia/cuda:13.3.1-base-ubuntu24.04

FROM ${BUILD_IMAGE} AS build
ARG TF_REPO
ARG TF_COMMIT
ARG XGRAMMAR_REPO
ARG XGRAMMAR_REF
ARG ZIG_VERSION
ARG ZIG_SHA256_AARCH64
ARG ZIG_SHA256_X86_64
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl git xz-utils \
    && rm -rf /var/lib/apt/lists/*
RUN set -eu; \
    case "$(uname -m)" in \
      aarch64) arch=aarch64; sum="$ZIG_SHA256_AARCH64" ;; \
      x86_64) arch=x86_64; sum="$ZIG_SHA256_X86_64" ;; \
      *) echo "unsupported architecture $(uname -m)"; exit 1 ;; \
    esac; \
    curl -fsSL -o /tmp/zig.tar.xz "https://ziglang.org/download/${ZIG_VERSION}/zig-${arch}-linux-${ZIG_VERSION}.tar.xz"; \
    echo "${sum}  /tmp/zig.tar.xz" | sha256sum -c -; \
    mkdir -p /opt/zig && tar -xJf /tmp/zig.tar.xz -C /opt/zig --strip-components=1 && rm /tmp/zig.tar.xz
RUN git clone --filter=blob:none --no-checkout "$TF_REPO" /src && git -C /src checkout --quiet --detach "$TF_COMMIT"
RUN git clone --quiet --depth 1 --branch "$XGRAMMAR_REF" --recurse-submodules --shallow-submodules "$XGRAMMAR_REPO" /xgrammar
COPY patches/ /patches/
RUN cd /src && git apply --whitespace=nowarn /patches/*.patch
RUN cd /src && /opt/zig/zig build native --prefix /opt/tensorfold -Dnvcc=/usr/local/cuda/bin/nvcc
RUN cd /src && /opt/zig/zig build grammar --prefix /opt/tensorfold -Dxgrammar=/xgrammar

FROM ${RUN_IMAGE}
ARG PATCHES_HASH=unknown
ARG TF_COMMIT
LABEL tf.patches="${PATCHES_HASH}" \
      tf.commit="${TF_COMMIT}" \
      org.opencontainers.image.title="Qwen3.8-Flash-Next on one DGX Spark (TensorFold)" \
      org.opencontainers.image.source="https://github.com/Arminova/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold" \
      org.opencontainers.image.licenses="Apache-2.0"
COPY --from=build /opt/tensorfold/native/bin/tensorfold-native /opt/tensorfold/bin/tensorfold-native
COPY --from=build /opt/tensorfold/native/lib/ /opt/tensorfold/lib/
ENV LD_LIBRARY_PATH=/opt/tensorfold/lib
ENTRYPOINT ["/opt/tensorfold/bin/tensorfold-native"]
