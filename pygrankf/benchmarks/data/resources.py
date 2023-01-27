import gzip
import tarfile
import zipfile
import wget
import shutil
import os
import yaml
from pyfop import *
from pygrankf.core import utils


def datasets():
    resources_folder = os.path.join(os.path.expanduser("~"), ".pygrank")
    resources_path = os.path.join(resources_folder, "resources.yml")
    if not os.path.exists(resources_path):
        os.makedirs(resources_folder, exist_ok=True)
        wget.download(
            "https://raw.githubusercontent.com/maniospas/pygrank-f/main/resources.yaml",
            resources_path,
        )
    with open(resources_path) as file:
        datasets = yaml.load(file, Loader=yaml.SafeLoader)
    return datasets


@lazy_no_cache
@autoaspects
def download(
    dataset, path: str = os.path.join(os.path.expanduser("~"), ".pygrank/data")
):  # pragma: no cover
    if not isinstance(dataset, dict):
        all_datasets = datasets()
        if dataset not in all_datasets:
            return
        credentials = (
            f"REQUIRED CITATION: Please visit the url {all_datasets[dataset]['url']} "
            f"for instructions on how to cite the dataset {dataset} in your research\n"
        )
        path = os.path.join(path, dataset)
        dataset = all_datasets[dataset]
        sys.stdout.flush()
        sys.stderr.write(credentials)
        sys.stderr.flush()
    os.makedirs(path, exist_ok=True)
    if os.path.exists(path + "/pairs.txt") and os.path.exists(path + "/groups.txt"):
        return
    downloaded = list()
    for download_url in dataset.get("download", ()):
        download_path = os.path.join(path, download_url.split("/")[-1])
        utils.log(f"Downloading {download_url} to {download_path}")
        downloaded.append(download_path)
        if not os.path.exists(download_path):
            wget.download(download_url, download_path)

    for compressed in dataset.get("unzip", ()):
        compressed = os.path.join(path, compressed)
        utils.log(f"Extracting {compressed}")
        with zipfile.ZipFile(compressed, "r") as file:
            file.extractall(path)

    for compressed in dataset.get("untar", ()):
        compressed = os.path.join(path, compressed)
        utils.log(f"Extracting {compressed}")
        with tarfile.open(compressed, "r") as file:
            file.extractall(path)

    for compressed in dataset.get("ungzip", ()):
        compressed = os.path.join(path, compressed)
        utils.log(f"Extracting {compressed}")
        with gzip.open(path, "rb") as file:
            Exception("have not inpmplemented ungzip yet")

    for remove in dataset.get("remove", ()):
        downloaded.append(os.path.join(path, remove))

    utils.log(f"Processing {os.path.join(path, 'pairs.txt')}")
    if dataset["pairs"].get("process", "False") == "False":
        shutil.move(
            os.path.join(path, dataset["pairs"]["file"]),
            os.path.join(path, "pairs.txt"),
        )
    else:
        entries = [int(pos) for pos in dataset["pairs"]["process"].split()]
        pairs = list()
        with open(
            os.path.join(path, dataset["pairs"]["file"]), "r", encoding="utf-8"
        ) as file:
            for line in file:
                if line.startswith("#"):
                    continue
                line = line[:-1]
                line = line.split()
                line = [line[pos] for pos in entries]
                pairs.append(line)
        with open(os.path.join(path, "pairs.txt"), "w", encoding="utf-8") as file:
            for pair in pairs:
                file.write(pair[0] + "\t" + pair[1] + "\n")

    utils.log(f"Processing {os.path.join(path, 'groups.txt')}")
    if dataset["groups"].get("process", "False") == "False":
        shutil.move(
            os.path.join(path, dataset["groups"]["file"]),
            os.path.join(path, "groups.txt"),
        )
    else:
        entries = [int(pos) for pos in dataset["groups"]["process"].split()]
        groups = dict()
        with open(
            os.path.join(path, dataset["groups"]["file"]), "r", encoding="utf-8"
        ) as file:
            for line in file:
                if line.startswith("#"):
                    continue
                line = line[:-1]
                line = line.split()
                line = [line[pos] for pos in entries]
                for group in line[1:]:
                    if group not in groups:
                        groups[group] = list()
                    groups[group].append(line[0])
        with open(os.path.join(path, "groups.txt"), "w", encoding="utf-8") as file:
            for group in groups.values():
                file.write("\t".join(group) + "\n")

        with open(os.path.join(path, "groupnames.txt"), "w", encoding="utf-8") as file:
            file.write("\n".join(groups))

    for download_path in downloaded:
        if download_path.endswith("/"):
            shutil.rmtree(download_path)
        else:
            os.remove(download_path)
    utils.log()
