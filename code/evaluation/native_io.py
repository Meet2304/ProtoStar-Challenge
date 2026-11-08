"""Path tripwires for ordinary Arrow APIs that bypass CPython's open event.

Installed inside a disposable worker before importing an adapter. This covers
the documented entry points below, not arbitrary extension internals or hostile
code that restores original functions. Existing buffers/descriptors cannot be
retroactively authorized. It deliberately makes no OS-sandbox claim.
"""
from functools import wraps
import inspect


def install_native_guards(audit):
    import pyarrow as pa
    import pyarrow.fs as fs
    import pyarrow.parquet as parquet
    import pyarrow.csv as csv
    import pyarrow.feather as feather
    import pyarrow.dataset as dataset
    import pyarrow.ipc as ipc
    import pyarrow.json as arrow_json

    # Each worker installs once. Standard public calls use guarded entry points;
    # deliberate introspection/restoration remains outside this trust boundary.
    def wrap(module, name, parameter, *, writing=False):
        original = getattr(module, name, None)
        if original is None:
            return
        try:
            signature = inspect.signature(original)
        except (TypeError, ValueError):
            signature = None

        @wraps(original)
        def guarded(*args, **kwargs):
            if parameter in kwargs:
                value = kwargs[parameter]
            elif signature is not None:
                value = signature.bind_partial(*args, **kwargs).arguments.get(parameter)
            else:
                value = args[0] if args else None
            audit.check_path(value, writing=writing)
            return original(*args, **kwargs)
        setattr(module, name, guarded)

    for module, names, parameter in (
        (parquet, ("read_table", "read_pandas", "read_metadata", "read_schema"), "source"),
        (csv, ("read_csv", "open_csv"), "input_file"),
        (feather, ("read_feather", "read_table"), "source"),
        (dataset, ("dataset",), "source"),
        (dataset, ("parquet_dataset",), "metadata_path"),
        (ipc, ("open_file", "open_stream"), "source"),
        (arrow_json, ("read_json", "open_json"), "input_file"),
        (pa, ("input_stream",), "source"),
    ):
        for name in names:
            wrap(module, name, parameter)

    wrap(pa, "create_memory_map", "path", writing=True)
    wrap(pa, "output_stream", "source", writing=True)
    original_memory_map = pa.memory_map

    def guarded_memory_map(path, mode="r"):
        audit.check_path(path, writing=any(char in mode for char in "wax+"))
        return original_memory_map(path, mode)

    pa.memory_map = guarded_memory_map
    for module, name, parameter in (
        (parquet, "write_table", "where"), (parquet, "write_metadata", "where"),
        (csv, "write_csv", "output_file"), (feather, "write_feather", "dest"),
        (dataset, "write_dataset", "base_dir"),
    ):
        wrap(module, name, parameter, writing=True)

    original_os_file = pa.OSFile

    class AuditedStreamMeta(type(original_os_file)):
        def __instancecheck__(cls, instance):
            return isinstance(instance, original_os_file)

    class AuditedOSFile(original_os_file, metaclass=AuditedStreamMeta):
        def __new__(cls, path, mode="r", *args, **kwargs):
            # OSFile opens in __cinit__, before a Python __init__ override.
            # Check BEFORE constructing the original native object.
            audit.check_path(path, writing=any(char in mode for char in "wax+"))
            return original_os_file(path, mode, *args, **kwargs)

    pa.OSFile = AuditedOSFile

    original_mapped_file = pa.MemoryMappedFile

    class AuditedMappedMeta(type(original_mapped_file)):
        def __instancecheck__(cls, instance):
            return isinstance(instance, original_mapped_file)

    class AuditedMappedFile(original_mapped_file, metaclass=AuditedMappedMeta):
        @staticmethod
        def create(path, size):
            audit.check_path(path, writing=True)
            return original_mapped_file.create(path, size)

    pa.MemoryMappedFile = AuditedMappedFile

    original_parquet_file = parquet.ParquetFile

    class AuditedParquetFile(original_parquet_file):
        def __init__(self, source, *args, **kwargs):
            audit.check_path(source)
            super().__init__(source, *args, **kwargs)

    parquet.ParquetFile = AuditedParquetFile

    original_local = fs.LocalFileSystem

    class AuditedLocalFileSystem(original_local):
        def open_input_file(self, path, *args, **kwargs):
            audit.check_path(path)
            return super().open_input_file(path, *args, **kwargs)

        def open_input_stream(self, path, *args, **kwargs):
            audit.check_path(path)
            return super().open_input_stream(path, *args, **kwargs)

        def open_output_stream(self, path, *args, **kwargs):
            audit.check_path(path, writing=True)
            return super().open_output_stream(path, *args, **kwargs)

        def open_append_stream(self, path, *args, **kwargs):
            audit.check_path(path, writing=True)
            return super().open_append_stream(path, *args, **kwargs)

        def copy_file(self, source, destination, *args, **kwargs):
            audit.check_path(source)
            audit.check_path(destination, writing=True)
            return super().copy_file(source, destination, *args, **kwargs)

        def move(self, source, destination, *args, **kwargs):
            audit.check_path(source, writing=True)
            audit.check_path(destination, writing=True)
            return super().move(source, destination, *args, **kwargs)

        def delete_file(self, path, *args, **kwargs):
            audit.check_path(path, writing=True)
            return super().delete_file(path, *args, **kwargs)

    fs.LocalFileSystem = AuditedLocalFileSystem

    original_filesystem = fs.FileSystem

    class AuditedFilesystemMeta(type(original_filesystem)):
        def __instancecheck__(cls, instance):
            return isinstance(instance, original_filesystem)

        def __subclasscheck__(cls, subclass):
            return issubclass(subclass, original_filesystem)

    class AuditedFileSystem(original_filesystem, metaclass=AuditedFilesystemMeta):
        @staticmethod
        def from_uri(uri):
            audit.check_path(uri)
            filesystem, path = original_filesystem.from_uri(uri)
            if isinstance(filesystem, original_local):
                filesystem = AuditedLocalFileSystem()
            else:
                audit.check_network("native_remote_filesystem")
            return filesystem, path

    # Preserve isinstance/isubclass behavior for actual Arrow filesystem
    # objects while covering the public factory that otherwise returns the
    # original native LocalFileSystem type rather than our exported subclass.
    fs.FileSystem = AuditedFileSystem
    wrap(fs, "copy_files", "source")
    wrap(fs, "copy_files", "destination", writing=True)

    # Remote SDK filesystems need not emit Python socket events. Refuse their
    # documented constructors before initialization, without logging options.
    def disabled_remote_class(original):
        class DisabledRemote(original):
            def __new__(cls, *args, **kwargs):
                audit.check_network("native_remote_filesystem")
                return original(*args, **kwargs)
        return DisabledRemote

    for name in ("S3FileSystem", "GcsFileSystem", "AzureFileSystem", "HadoopFileSystem"):
        original = getattr(fs, name, None)
        if original is not None:
            setattr(fs, name, disabled_remote_class(original))
