"""Throwaway: does onnxruntime import in a platform pod?

PROBE_MODE: conda | tunable | patch | image     ORT_VERSION: x.y.z
"""

import os

from metaflow import FlowSpec, anaconda_base, environment, kubernetes, step

MODE = os.environ.get("PROBE_MODE", "conda")
VERSION = os.environ.get("ORT_VERSION", "1.24.4")
IMAGE = os.environ.get("PROBE_IMAGE", "python:3.12-slim")
CONDA = {"numpy": "2.5.4", "onnxruntime": VERSION}


def step_decorators(f):
    if MODE == "tunable":
        f = environment(vars={"GLIBC_TUNABLES": "glibc.rtld.execstack=2"})(f)
    if MODE == "image":
        f = kubernetes(image=IMAGE)(f)
    return f


@anaconda_base(python="3.12", packages=CONDA, channels=["https://repo.anaconda.com/pkgs/main"])
class OrtProbe(FlowSpec):
    @step_decorators
    @step
    def start(self):
        import glob
        import struct
        import subprocess

        ldd = subprocess.run(["ldd", "--version"], capture_output=True, text=True)
        print("PROBE_LIBC", ldd.stdout.splitlines()[0])
        print("PROBE_OS", open("/etc/os-release").read().splitlines()[:2])
        if MODE == "patch":
            import sys

            for path in glob.glob(
                os.path.join(sys.prefix, "lib/python3.12/site-packages/onnxruntime/capi/*.so")
            ) + glob.glob("/opt/outerbounds/fastbakery/lib/python3.12/site-packages/onnxruntime/capi/*.so"):  # noqa: E501
                with open(path, "r+b") as f:
                    ehdr = f.read(64)
                    phoff = struct.unpack_from("<Q", ehdr, 32)[0]
                    phentsize, phnum = struct.unpack_from("<HH", ehdr, 54)
                    for i in range(phnum):
                        off = phoff + i * phentsize
                        f.seek(off)
                        p_type, p_flags = struct.unpack("<II", f.read(8))
                        if p_type == 0x6474E551:
                            print("PROBE_PATCH", path, hex(p_flags))
                            f.seek(off + 4)
                            f.write(struct.pack("<I", p_flags & ~1))
        import onnxruntime

        print("PROBE_OK", MODE, onnxruntime.__version__, onnxruntime.__file__)
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    OrtProbe()
