"""
PlatformIO: собирает модели esp-sr (srmodels.bin) и прошивает их в раздел "model".

idf.py делает это сам, а PlatformIO собирает проект через SCons и пропускает
CMake-шаг esp-sr, который упаковывает модели. Скрипт повторяет этот шаг:
  1. movemodel.py из esp-sr читает sdkconfig и упаковывает выбранные модели
     (CONFIG_SR_WN_WN10_RU_PRIVETESP) в <build>/srmodels/srmodels.bin
  2. esptool записывает файл по адресу раздела "model"

Запускается автоматически после `pio run -t upload`.
Отдельно, без перепрошивки приложения: `pio run -t flash_models`.
"""

import glob
import hashlib
import os
import subprocess

Import("env")  # noqa: F821 (определяется PlatformIO)

MODEL_OFFSET = "0x610000"  # должен совпадать с разделом "model" в partitions.csv


def find_esp_sr(project_dir):
    candidates = glob.glob(os.path.join(project_dir, "managed_components", "*esp-sr*"))
    for path in candidates:
        if os.path.isfile(os.path.join(path, "model", "movemodel.py")):
            return path
    return None


def build_models(env):
    project_dir = env.subst("$PROJECT_DIR")
    build_dir = env.subst("$BUILD_DIR")
    sdkconfig = os.path.join(project_dir, "sdkconfig." + env.subst("$PIOENV"))
    esp_sr = find_esp_sr(project_dir)

    if not os.path.isfile(sdkconfig):
        print("[models] %s не найден - сначала соберите проект (pio run)" % sdkconfig)
        env.Exit(1)
    if esp_sr is None:
        print("[models] esp-sr не найден в managed_components - сначала соберите проект (pio run)")
        env.Exit(1)

    print("[models] Упаковка моделей esp-sr из %s" % esp_sr)
    subprocess.check_call([
        env.subst("$PYTHONEXE"),
        os.path.join(esp_sr, "model", "movemodel.py"),
        "-d1", sdkconfig,
        "-d2", esp_sr,
        "-d3", build_dir,
    ], cwd=os.path.join(esp_sr, "model"))

    image = os.path.join(build_dir, "srmodels", "srmodels.bin")
    if not os.path.isfile(image):
        print("[models] movemodel.py не создал %s" % image)
        env.Exit(1)

    with open(sdkconfig, encoding="utf-8") as f:
        if "CONFIG_SR_WN_WN10_RU_PRIVETESP=y" not in f.read():
            print("[models] ВНИМАНИЕ: CONFIG_SR_WN_WN10_RU_PRIVETESP не включён в %s" % sdkconfig)
    return image


def flash_models(source, target, env):
    image = build_models(env)
    size_kb = os.path.getsize(image) // 1024
    with open(image, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()[:12]

    port = env.subst("$UPLOAD_PORT")
    cmd = [
        env.subst("$PYTHONEXE"), env.subst("$UPLOADER"),
        "--chip", "esp32s3",
        "--baud", env.subst("$UPLOAD_SPEED") or "460800",
        "write_flash", MODEL_OFFSET, image,
    ]
    if port:
        cmd[2:2] = ["--port", port]

    print("[models] Прошивка srmodels.bin (%d KB, sha %s) по адресу %s" % (size_kb, digest, MODEL_OFFSET))
    subprocess.check_call(cmd)
    print("[models] Готово")


env.AddPostAction("upload", flash_models)  # noqa: F821

env.AddCustomTarget(  # noqa: F821
    name="flash_models",
    dependencies=None,
    actions=[flash_models],
    title="Flash esp-sr models",
    description="Pack srmodels.bin and write it to the 'model' partition",
)
