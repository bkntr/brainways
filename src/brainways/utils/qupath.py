import platform
import shutil
from pathlib import Path
from typing import Optional

from brainglobe_atlasapi.utils import check_internet_connection
from paquo import settings as paquo_settings
from paquo.jpype_backend import find_qupath

from brainways.utils.paths import get_brainways_dir


def get_brainways_qupath_dir() -> Path:
    brainways_dir = get_brainways_dir()
    brainways_qupath_dir = brainways_dir / "qupath"
    if not brainways_qupath_dir.exists():
        brainways_qupath_dir.mkdir()
    return brainways_qupath_dir


def add_brainways_qupath_dir_to_paquo_settings():
    qupath_brainways_dir = str(get_brainways_qupath_dir())
    if qupath_brainways_dir not in paquo_settings.qupath_search_dirs:
        paquo_settings.qupath_search_dirs = paquo_settings.qupath_search_dirs + [
            qupath_brainways_dir
        ]


def is_qupath_downloaded(version: str):
    """As we can't know the local version a priori, search candidate dirs
    using name and not version number. If none is found, return None.
    """
    add_brainways_qupath_dir_to_paquo_settings()
    try:
        app_dir, runtime_dir, jvm_dir, jvm_options = find_qupath(
            **{k.lower(): v for k, v in paquo_settings.to_dict().items()}
        )
        if f"QuPath-{version}" not in str(app_dir):
            shutil.rmtree(get_brainways_qupath_dir())
            return False
        return True
    except ValueError:
        return False


def download_qupath(
    install_path: Optional[Path] = None,
    version: str = "0.5.0",
    system: str = platform.system(),
    download_path: Optional[Path] = None,
    ssl_verify: bool = False,
):
    """
    Adapted from:
    https://github.com/bayer-science-for-a-better-life/paquo/blob/main/paquo/__main__.py#L316
    download a specific QuPath version
    """
    from paquo._utils import download_qupath, extract_qupath

    if install_path is None:
        install_path = get_brainways_qupath_dir()
    if download_path is None:
        download_path = get_brainways_qupath_dir()

    check_internet_connection()

    # TODO: callback to GUI
    def _download_cb(it, name):
        if name:
            print("# downloading:", name)
        print("# progress ", end="", flush=True)
        try:
            for chunk in it:
                print(".", end="", flush=True)
                yield chunk
            print(" OK", end="", flush=True)
        finally:
            print("")

    file = download_qupath(
        version,
        path=download_path,
        callback=_download_cb,
        system=system,
        ssl_verify=ssl_verify,
    )
    print("# extracting:", file)
    app = extract_qupath(file, install_path, system=system)
    print("# available at:", app)

    print("#\n# use via environment variable:")
    if system in {"Linux", "Darwin"}:
        print(f"#  $ export PAQUO_QUPATH_DIR={app}")
    else:
        print("#  REM Windows CMD")
        print(f'#  C:\\> set PAQUO_QUPATH_DIR="{app}"')
        print("#  # Windows PowerShell")
        print(f'#  PS C:\\> $env:PAQUO_QUPATH_DIR="{app}"')
    print("#\n# use via .paquo.toml config file:")
    print(f'#  qupath_dir="{app}"')
    print(app)
