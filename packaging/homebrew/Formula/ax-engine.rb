class AxEngine < Formula
  desc "Mac-first LLM inference engine targeting Apple M4+ Silicon"
  homepage "https://github.com/defai-digital/ax-engine"
  url "https://github.com/defai-digital/ax-engine/releases/download/v6.9.0/ax-engine-v6.9.0-macos-arm64.tar.gz"
  version "6.9.0"
  sha256 "4909c3aa7436413720472182d2887e66efa7cc98aec1cdca5825f3b3ab7e5757"
  license "Apache-2.0"

  depends_on arch: :arm64
  depends_on :macos

  # Homebrew otherwise rewrites @rpath dylib IDs to opt-prefix paths during
  # formula installation and replaces our Developer ID signatures with ad-hoc
  # signatures. Preserve the signed load commands from the release archive.
  preserve_rpath

  def install
    bin.install "ax-engine",
                "ax-engine-server",
                "ax-engine-bench",
                "ax-engine-download-model.py",
                "ax-engine-prepare-mtp-sidecar.py",
                "ax-engine-prepare-gemma4-assistant-mtp.py",
                "ax-engine-prepare-glm-mtp-sidecar.py",
                "ax-engine-prepare-qwen36-mtp-sidecar.py",
                "ax-engine-check-mtp-sidecar-provenance.py"
    # Keep mlx.metallib beside libmlx.dylib: MLX resolves its precompiled
    # kernels relative to the loaded dylib. Release binaries also carry
    # @loader_path/../libexec so this private Homebrew runtime remains
    # relocatable without editing or invalidating Developer ID signatures.
    # libexec also avoids colliding with a separately installed mlx formula.
    libexec.install "libmlx.dylib", "libjaccl.dylib", "mlx.metallib"
    doc.install "MLX-LICENSE.txt"
  end

  def caveats
    <<~EOS
      ax-engine binaries for this release are Developer ID signed and notarized
      by Apple. The formula installs the release's pinned, prebuilt MLX runtime
      beside its mlx.metallib without rewriting the signed Mach-O files.
      End users do not need Python, Homebrew MLX, Xcode, or the Metal Toolchain.
    EOS
  end

  test do
    assert_match "ax-engine", shell_output("#{bin}/ax-engine --help 2>&1")
    # These two load libmlx; --help must succeed under dyld (catches broken rpath).
    assert_match "ax-engine-server", shell_output("#{bin}/ax-engine-server --help 2>&1")
    assert_match "ax-engine-bench", shell_output("#{bin}/ax-engine-bench --help 2>&1")
  end
end
