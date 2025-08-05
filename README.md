# 🧠 Jeneen's MCP Agent

This project is a multi-functional AI agent built using [FastMCP](https://github.com/modelcontextprotocol). It includes:

- ✅ An Arabic legal chatbot that answers common legal questions.
- 🔍 A Google search tool.
- 🧬 A Variational Autoencoder (VAE) model that generates handwritten digit images.

---

## 📁 Project Structure


├── main.py # Main script to run the MCP agent
├── chatbot.py # Arabic legal chatbot logic
├── vae_model.py # VAE model definitions (Encoder, Decoder, VAE)
├── output/ # Model checkpoints and generated images
├── data/ # MNIST dataset (auto-downloaded)
├── VAE.ipynb # Jupyter notebook for training the VAE model
└── README.md # This documentation file


---

## ⚙️ Available MCP Tools

### 1. `legal_chat(query: str) → str`
Arabic-language chatbot that responds to legal questions such as:
- Annual leave
- Divorce
- Custody
- Employment rights
- Rental agreements
**Example:**
{ "tool": "legal_chat", "input": "ما هي حقوقي في حال الطلاق؟" }

### 2. `search_google(query: str) → str`
Opens a Google search in the default browser.
Example:
{ "tool": "search_google", "input": "قانون العمل الأردني" }


### `3. vae_generate(n_images: int) → str`
Generates handwritten digit images using a trained VAE model.
Returns a base64-encoded PNG image.
Example:
{ "tool": "vae_generate", "input": { "n_images": 8 } }


 How to Run
Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate


Install dependencies
pip install -r requirements.txt
(Optional) Train the VAE model using VAE.ipynb
Or use the pre-trained model in: output/vae_epoch_50.pth

Run the MCP agent
python main.py





