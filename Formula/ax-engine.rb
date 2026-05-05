class AxEngine < Formula
  desc "Mac-first LLM inference engine targeting Apple M4+ Silicon"
  homepage "https://github.com/defai-digital/ax-engine"
  version "4.2.2"

  on_macos do
    if Hardware::CPU.arm?
      url "https://github.com/defai-digital/ax-engine/releases/download/v4.2.2/ax-engine-v4.2.2-macos-arm64.tar.gz"
      sha256 "6325fef1eeec1d720a950d58da556caaee03659063f1f15a3b501138968f7e26"
    else
      odie "ax-engine requires Apple Silicon (arm64)."
    end
  end

  depends_on :macos
  depends_on arch: :arm64

  def install
    bin.install "ax-engine-server"
    bin.install "ax-engine-bench"
  end

  test do
    assert_match "ax-engine-server", shell_output("#{bin}/ax-engine-server --help 2>&1")
    assert_match "ax-engine-bench", shell_output("#{bin}/ax-engine-bench --help 2>&1")
  end
end
