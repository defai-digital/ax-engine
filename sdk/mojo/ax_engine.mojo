"""
AX Engine Mojo SDK — thin wrapper around the ax_engine Python package.

Mojo's `PythonObject` interop makes it straightforward to delegate all
inference calls to the Python SDK while keeping Mojo-idiomatic call sites.

Prerequisites:
    maturin develop          # build and install the ax_engine Python extension
    magic add python ax-engine  # or ensure ax_engine is on PYTHONPATH
"""

from python import Python, PythonObject


struct GenerateResult:
    """Holds the result of a generate() or generate_text() call."""

    var output_text: String
    var output_tokens: PythonObject
    var finish_reason: String

    fn __init__(out self, py_result: PythonObject) raises:
        self.output_tokens = py_result.output_tokens
        var text = py_result.output_text
        self.output_text = str(text) if text.__class__.__name__ != "NoneType" else ""
        var reason = py_result.finish_reason
        self.finish_reason = str(reason) if reason.__class__.__name__ != "NoneType" else ""


struct Session:
    """
    A thin Mojo wrapper around ax_engine.Session.

    Usage::

        var s = Session("qwen3_dense", mlx=True,
                        mlx_model_artifacts_dir="/path/to/artifacts")
        var result = s.generate("Hello from Mojo")
        print(result.output_text)
        s.close()

    Or as a context manager via the Python session directly::

        var ax = Python.import_module("ax_engine")
        with ax.Session(...) as session:
            ...
    """

    var _session: PythonObject
    var _ax: PythonObject

    fn __init__(
        out self,
        model_id: String,
        mlx: Bool = False,
        mlx_model_artifacts_dir: String = "",
        llama_model_path: String = "",
        llama_server_url: String = "",
        mlx_lm_server_url: String = "",
        support_tier: String = "",
    ) raises:
        self._ax = Python.import_module("ax_engine")
        var kwargs = Python.dict()
        kwargs["model_id"] = model_id
        if mlx:
            kwargs["mlx"] = True
        if mlx_model_artifacts_dir:
            kwargs["mlx_model_artifacts_dir"] = mlx_model_artifacts_dir
        if llama_model_path:
            kwargs["llama_model_path"] = llama_model_path
        if llama_server_url:
            kwargs["llama_server_url"] = llama_server_url
        if mlx_lm_server_url:
            kwargs["mlx_lm_server_url"] = mlx_lm_server_url
        if support_tier:
            kwargs["support_tier"] = support_tier
        self._session = self._ax.Session(**kwargs)
        self._session.__enter__()

    fn generate(self, input_text: String, max_output_tokens: Int = 256) raises -> GenerateResult:
        """Generate text from a prompt string."""
        var py_result = self._session.generate(
            input_text=input_text, max_output_tokens=max_output_tokens
        )
        return GenerateResult(py_result)

    fn generate_tokens(
        self, input_tokens: PythonObject, max_output_tokens: Int = 256
    ) raises -> GenerateResult:
        """Generate from a pre-tokenized token list (PythonObject list of ints)."""
        var py_result = self._session.generate(
            input_tokens, max_output_tokens=max_output_tokens
        )
        return GenerateResult(py_result)

    fn close(self) raises:
        """Close the underlying Python session. Call this when done."""
        self._session.__exit__(None, None, None)


fn download_model(repo_id: String, dest: String = "", force: Bool = False) raises -> String:
    """
    Download an mlx-community model from Hugging Face Hub.

    Returns the local directory path. Equivalent to ax_engine.download_model().

    Requires: pip install huggingface_hub
    """
    var ax = Python.import_module("ax_engine")
    var kwargs = Python.dict()
    kwargs["force"] = force
    if dest:
        kwargs["dest"] = dest
    var path = ax.download_model(repo_id, **kwargs)
    return str(path)
