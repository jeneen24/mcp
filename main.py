from mcp.server.fastmcp import FastMCP
from chatbot import Chatbot
from vae_model import VAE
import urllib.parse
import webbrowser
import torch
import base64
import io
from torchvision.utils import make_grid
from PIL import Image


device = torch.device("cpu")
latent_dim = 20
vae = VAE(latent_dim).to(device)

vae.load_state_dict(torch.load("output/vae_epoch_50.pth", map_location=device))
vae.eval()

chatbot = Chatbot()
mcp = FastMCP("Jeneen's Chatbot")





@mcp.tool()
def search_google(query: str) -> str:
    encoded_query = urllib.parse.quote(query)
    url = f"https://www.google.com/search?q={encoded_query}"
    webbrowser.open(url)
    return f" تم فتح بحث جوجل عن: {query}"

@mcp.tool()
def legal_chat(query: str) -> str:
    return chatbot.get_response(query)


@mcp.tool()
def vae_generate(n_images: int = 8) -> str:
    with torch.no_grad():
        z = torch.randn(n_images, latent_dim).to(device)
        samples = vae.decoder(z).cpu()

        grid = make_grid(samples, nrow=4, pad_value=1)
        ndarr = (grid.numpy().transpose(1, 2, 0) * 255).astype("uint8")
        image = Image.fromarray(ndarr.squeeze(), mode="L")

        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f" Base64 Image:\n{img_str}"


if __name__ == "__main__":
    mcp.run()
