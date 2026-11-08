"""Path tripwires for ordinary Arrow APIs that bypass CPython's open event.

Installed inside a disposable worker before importing an adapter. This covers
the documented entry points below, not arbitrary extension internals or hostile
code that restores original functions. Existing buffers/descriptors cannot be
retroactively authorized. It deliberately makes no OS-sandbox claim.
"""
from functools import wraps
import inspect
import os
import posixpath


def install_native_guards(audit):
    import pyarrow as pa
    import pyarrow.fs as fs
    import pyarrow.parquet as parquet
    import pyarrow.parquet.core as parquet_core
    import pyarrow.orc as orc
    import pyarrow.csv as csv
    import pyarrow.feather as feather
    import pyarrow.dataset as dataset
    import pyarrow.ipc as ipc
    import pyarrow.json as arrow_json

    # Each worker installs once. Standard public calls use guarded entry points;
    # deliberate introspection/restoration remains outside this trust boundary.
    original_subtree = fs.SubTreeFileSystem

    def check_location(value, filesystem=None, *, writing=False):
        # Subtree names are relative to a different filesystem root. Resolve
        # them before policy checks, including when C++ bypasses Python methods.
        if isinstance(filesystem, original_subtree):
            if isinstance(value, (list, tuple)):
                for item in value:
                    check_location(item, filesystem, writing=writing)
                return
            if isinstance(value, (str, bytes, os.PathLike)):
                value = posixpath.join(filesystem.base_path, os.fsdecode(value).replace("\\", "/"))
                return check_location(value, filesystem.base_fs, writing=writing)
        audit.check_path(value, writing=writing)

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
            bound = signature.bind_partial(*args, **kwargs).arguments if signature is not None else {}
            if parameter in kwargs:
                value = kwargs[parameter]
            elif signature is not None:
                value = bound.get(parameter)
            else:
                value = args[0] if args else None
            filesystem = kwargs.get("filesystem", bound.get("filesystem"))
            check_location(value, filesystem, writing=writing)
            return original(*args, **kwargs)
        setattr(module, name, guarded)

    for module, names, parameter in (
        (parquet, ("read_table", "read_pandas", "read_metadata", "read_schema"), "source"),
        (parquet_core, ("read_table", "read_pandas", "read_metadata", "read_schema"), "source"),
        (orc, ("read_table",), "source"),
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
        (parquet_core, "write_table", "where"), (parquet_core, "write_metadata", "where"),
        (orc, "write_table", "where"),
        (csv, "write_csv", "output_file"), (feather, "write_feather", "dest"),
        (dataset, "write_dataset", "base_dir"),
        (ipc, "new_file", "sink"), (ipc, "new_stream", "sink"),
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
            check_location(source, kwargs.get("filesystem"))
            super().__init__(source, *args, **kwargs)

    parquet.ParquetFile = AuditedParquetFile
    parquet_core.ParquetFile = AuditedParquetFile

    original_parquet_dataset = parquet.ParquetDataset

    class AuditedParquetDataset(original_parquet_dataset):
        def __init__(self, path_or_paths, filesystem=None, *args, **kwargs):
            check_location(path_or_paths, filesystem)
            super().__init__(path_or_paths, filesystem, *args, **kwargs)

    parquet.ParquetDataset = AuditedParquetDataset
    parquet_core.ParquetDataset = AuditedParquetDataset

    original_orc_file = orc.ORCFile

    class AuditedORCFile(original_orc_file):
        def __init__(self, source):
            audit.check_path(source)
            super().__init__(source)

    orc.ORCFile = AuditedORCFile

    # These Python factories return original native writers after checking the
    # destination, before native __cinit__ can open or truncate it.
    def writer_class(module, name, parameter):
        original = getattr(module, name, None)
        if original is None:
            return

        class GuardedWriterMeta(type(original)):
            def __instancecheck__(cls, instance):
                return isinstance(instance, original)

        class GuardedWriter(original, metaclass=GuardedWriterMeta):
            def __new__(cls, *args, **kwargs):
                destination = kwargs.get(parameter, args[0] if args else None)
                check_location(destination, kwargs.get("filesystem"), writing=True)
                return original(*args, **kwargs)

        setattr(module, name, GuardedWriter)
        return GuardedWriter

    guarded_parquet_writer = writer_class(parquet, "ParquetWriter", "where")
    parquet_core.ParquetWriter = guarded_parquet_writer
    writer_class(csv, "CSVWriter", "sink")
    writer_class(orc, "ORCWriter", "where")
    writer_class(ipc, "RecordBatchFileWriter", "sink")
    writer_class(ipc, "RecordBatchStreamWriter", "sink")

    original_format = dataset.ParquetFileFormat

    class AuditedParquetFileFormat(original_format):
        def make_fragment(self, file, filesystem=None, *args, **kwargs):
            check_location(file, filesystem)
            return super().make_fragment(file, filesystem, *args, **kwargs)

    dataset.ParquetFileFormat = AuditedParquetFileFormat

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

        def create_dir(self, path, *args, **kwargs):
            audit.check_path(path, writing=True)
            return super().create_dir(path, *args, **kwargs)

        def delete_dir(self, path, *args, **kwargs):
            audit.check_path(path, writing=True)
            return super().delete_dir(path, *args, **kwargs)

        def delete_dir_contents(self, path, *args, **kwargs):
            audit.check_path(path, writing=True)
            return super().delete_dir_contents(path, *args, **kwargs)

    fs.LocalFileSystem = AuditedLocalFileSystem

    class GuardedSubtreeHandler(fs.FileSystemHandler):
        """Preserve Arrow's native subtree semantics, guard actual I/O in Python.

        PyFileSystem dispatches even C++-initiated opens through these methods;
        a LocalFileSystem Python subclass alone does not provide that property.
        Metadata listing is delegated; opening/mutating data is path-checked.
        """
        def __init__(self, subtree):
            self.subtree = subtree

        def check(self, path, writing=False):
            check_location(path, self.subtree, writing=writing)

        def get_type_name(self):
            return "protostar-guarded-subtree"

        def normalize_path(self, path):
            return self.subtree.normalize_path(path)

        def get_file_info(self, paths):
            return self.subtree.get_file_info(paths)

        def get_file_info_selector(self, selector):
            return self.subtree.get_file_info(selector)

        def create_dir(self, path, recursive):
            self.check(path, True)
            self.subtree.create_dir(path, recursive=recursive)

        def delete_dir(self, path):
            self.check(path, True)
            self.subtree.delete_dir(path)

        def delete_dir_contents(self, path, missing_dir_ok):
            self.check(path, True)
            self.subtree.delete_dir_contents(path, missing_dir_ok=missing_dir_ok)

        def delete_root_dir_contents(self):
            self.check("", True)
            self.subtree.delete_dir_contents("")

        def delete_file(self, path):
            self.check(path, True)
            self.subtree.delete_file(path)

        def move(self, source, destination):
            self.check(source, True)
            self.check(destination, True)
            self.subtree.move(source, destination)

        def copy_file(self, source, destination):
            self.check(source)
            self.check(destination, True)
            self.subtree.copy_file(source, destination)

        def open_input_stream(self, path):
            self.check(path)
            return self.subtree.open_input_stream(path)

        def open_input_file(self, path):
            self.check(path)
            return self.subtree.open_input_file(path)

        def open_output_stream(self, path, metadata):
            self.check(path, True)
            return self.subtree.open_output_stream(path, metadata=metadata)

        def open_append_stream(self, path, metadata):
            self.check(path, True)
            return self.subtree.open_append_stream(path, metadata=metadata)

    class GuardedSubtreeMeta(type(original_subtree)):
        def __instancecheck__(cls, instance):
            return isinstance(instance, original_subtree) or (
                isinstance(instance, fs.PyFileSystem) and isinstance(instance.handler, GuardedSubtreeHandler))

    class GuardedSubtreeFileSystem(fs.PyFileSystem):
        @property
        def base_path(self):
            return self.handler.subtree.base_path

        @property
        def base_fs(self):
            return self.handler.subtree.base_fs

    class AuditedSubTreeFileSystem(original_subtree, metaclass=GuardedSubtreeMeta):
        def __new__(cls, base_path, base_fs):
            native = original_subtree(base_path, base_fs)
            return GuardedSubtreeFileSystem(GuardedSubtreeHandler(native))

    fs.SubTreeFileSystem = AuditedSubTreeFileSystem

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
