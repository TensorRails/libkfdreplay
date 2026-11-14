#!/usr/bin/env bash
# Download the TensorRails DMTCP fork used with libkfdreplay into $PREFIX (default /opt/dmtcp).
#   scripts/install_dmtcp.sh [prefix]
set -euo pipefail
VERSION=${DMTCP_VERSION:-v5.0.0-warmctl.2}
PREFIX=${1:-/opt/dmtcp}
url=https://github.com/TensorRails/dmtcp/releases/download/$VERSION/dmtcp-$VERSION-linux-amd64-debian12.tar.gz
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
curl -fsSL -o "$tmp/dmtcp.tgz" "$url"
tar -xzf "$tmp/dmtcp.tgz" -C "$tmp"
src=$(dirname "$(find "$tmp" -path '*/bin/dmtcp_launch' | head -1)")/..
mkdir -p "$PREFIX"
cp -a "$src"/. "$PREFIX"/
echo "DMTCP $VERSION installed in $PREFIX; export DMTCP_ROOT=$PREFIX"
