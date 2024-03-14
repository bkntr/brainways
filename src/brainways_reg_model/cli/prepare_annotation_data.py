import shutil
from pathlib import Path


def main():
    dst_path = Path("data/annotate/images")
    if dst_path.exists():
        shutil.rmtree(dst_path)

    dirlist = {
        "/data/cfos/2015_08_pilot_2/jpg/Alexa Fluor 488/": "2015_08_pilot_2/",
        "/data/cfos/2018_06_dev_round_3_and_4/jpg/AF488/": "2018_06_dev_round_3_and_4/",
        "/data/cfos/2021_helping_behavior/jpg/EGFP/": "2021_helping_behavior/",
        "/data/cfos/2018_retrograde/jpg/AF647/": "2018_retrograde/",
    }
    for src, dst in dirlist.items():
        shutil.copytree(src, dst_path / dst)

    Path("data/annotate.zip").unlink(missing_ok=True)
    shutil.make_archive("data/annotate", "zip", dst_path)


if __name__ == "__main__":
    main()
