import requests
import os

def requisicao(owner, repo):
    url_release = f"https://api.github.com/repos/{owner}/{repo}/releases/latest"
    res = requests.get(url_release)
    return res.json()

def localizar(owner, repo, nome_arquivo, destino) -> None:
    release_data = requisicao(owner, repo)
    asset_url = None

    for asset in release_data["assets"]:
        if asset["name"] == nome_arquivo:
            asset_url = asset["browser_download_url"]
            break

    if asset_url:
        os.makedirs(os.path.dirname(destino), exist_ok=True)
        response = requests.get(asset_url)
        with open(destino, "wb") as f:
            f.write(response.content)

def main():
    owner = "tamires123-hub"
    repo = "pipeline_dadosESI"
    nome_arquivo = "dataset_filmes_class.csv"
    destino = "dados/dataset_filmes_class.csv"
    localizar(owner, repo, nome_arquivo, destino)

if __name__ == "__main__":
    main()
