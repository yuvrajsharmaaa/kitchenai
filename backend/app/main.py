from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response

from app.api.routes import router
from app.core.config import get_settings

settings = get_settings()
app = FastAPI(title="Kitchen Redesign API")

# Allow local frontend apps (and the demo UI) to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> Response:
    # Avoid noisy 404s from browsers requesting a favicon.
    return Response(status_code=204)


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    # Minimal HTML UI for quick manual testing without the frontend app.
    return """
<!doctype html>
<html lang=\"en\">
    <head>
        <meta charset=\"utf-8\" />
        <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
        <title>Kitchen Segmentation MVP</title>
        <link rel=\"icon\" href=\"data:,\" />
        <style>
            :root { color-scheme: light; }
            body { font-family: ui-sans-serif, system-ui, -apple-system, Arial, sans-serif; margin: 24px; }
            h1 { margin: 0 0 12px; }
            form { margin: 16px 0; display: flex; gap: 12px; align-items: center; }
            input[type=file] { max-width: 320px; }
            button { padding: 8px 14px; cursor: pointer; }
            .grid { display: grid; gap: 16px; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); }
            figure { margin: 0; }
            figcaption { font-size: 12px; color: #444; margin-top: 6px; }
            img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 6px; }
            .status { color: #666; font-size: 14px; }
            .error { color: #b00020; font-size: 14px; white-space: pre-wrap; }
        </style>
    </head>
    <body>
        <h1>Kitchen Segmentation MVP</h1>
        <p class=\"status\">Upload a kitchen image to generate masks and overlay.</p>
        <form id=\"upload-form\">
            <input id=\"image\" name=\"image\" type=\"file\" accept=\"image/*\" required />
            <button type=\"submit\">Run segmentation</button>
        </form>
        <hr />
        <h2>Inpaint / Edit</h2>
        <p class=\"status\">Upload original image, a binary mask (white=fill), and a prompt to edit. First inpaint can take a few minutes while the model loads.</p>
        <form id=\"inpaint-form\">
            <input id=\"inpaint-image\" name=\"image\" type=\"file\" accept=\"image/*\" required />
            <input id=\"inpaint-mask\" name=\"mask\" type=\"file\" accept=\"image/*\" required />
            <input id=\"inpaint-prompt\" name=\"prompt\" type=\"text\" placeholder=\"e.g. replace tiles with white marble\" style=\"flex:1; min-width:240px;\" required />
            <button type=\"submit\">Run inpaint</button>
        </form>
        <div id=\"inpaint-error\" class=\"error\"></div>
        <div id=\"inpaint-results\" class=\"grid\"></div>
        <div id=\"error\" class=\"error\"></div>
        <div id=\"results\" class=\"grid\"></div>

        <script>
            const form = document.getElementById("upload-form");
            const results = document.getElementById("results");
            const error = document.getElementById("error");

            form.addEventListener("submit", async (event) => {
                event.preventDefault();
                results.innerHTML = "";
                error.textContent = "";

                const fileInput = document.getElementById("image");
                const file = fileInput.files[0];
                window._lastUploadFile = file;
                if (!file) return;

                const formData = new FormData();
                formData.append("image", file);

                try {
                    const response = await fetch("/api/segment", { method: "POST", body: formData });
                    if (!response.ok) {
                        const payload = await response.json().catch(() => ({}));
                        throw new Error(payload.detail || `Request failed: ${response.status}`);
                    }

                    const data = await response.json();
                    const items = [
                        { label: "Original", data: data.original_image },
                        { label: "Overlay", data: data.overlay },
                        { label: "Segmentation Map", data: data.segmentation_map },
                    ];

                    for (const [label, mask] of Object.entries(data.masks || {})) {
                        items.push({ label: `Mask: ${label}`, data: mask });
                    }

                    for (const item of items) {
                        const figure = document.createElement("figure");
                        const img = document.createElement("img");
                        img.src = `data:image/png;base64,${item.data}`;
                        img.alt = item.label;
                        const caption = document.createElement("figcaption");
                        caption.textContent = item.label;
                        figure.appendChild(img);
                        figure.appendChild(caption);
                        // If this is a mask item, add an Edit button
                        if (item.label.startsWith("Mask: ")) {
                            const className = item.label.replace(/^Mask: /, "");
                            const btn = document.createElement("button");
                            btn.type = "button";
                            btn.textContent = "Edit this class";
                            btn.style.marginTop = "8px";
                            btn.addEventListener("click", async () => {
                                const prompt = document.getElementById("inpaint-prompt").value || "";
                                const lastFile = window._lastUploadFile;
                                if (!lastFile) {
                                    alert("Please re-upload the original image in the Inpaint form or run segmentation again.");
                                    return;
                                }
                                try {
                                    const fd = new FormData();
                                    fd.append("image", lastFile);
                                    fd.append("class_name", className);
                                    fd.append("prompt", prompt || `Edit ${className}`);
                                    const resp = await fetch("/api/inpaint_from_class", { method: "POST", body: fd });
                                    if (!resp.ok) {
                                        const payload = await resp.json().catch(() => ({}));
                                        throw new Error(payload.detail || `Request failed: ${resp.status}`);
                                    }
                                    const js = await resp.json();
                                    const fig = document.createElement("figure");
                                    const im = document.createElement("img");
                                    im.src = `data:image/png;base64,${js.image}`;
                                    const cap = document.createElement("figcaption");
                                    cap.textContent = `Inpainted: ${className}`;
                                    fig.appendChild(im);
                                    fig.appendChild(cap);
                                    document.getElementById("inpaint-results").appendChild(fig);
                                } catch (e) {
                                    document.getElementById("inpaint-error").textContent = e.message || String(e);
                                }
                            });
                            figure.appendChild(btn);
                        }
                        results.appendChild(figure);
                    }
                } catch (err) {
                    error.textContent = err.message || String(err);
                }
            });

            // Inpaint form handling
            const inpaintForm = document.getElementById("inpaint-form");
            const inpaintResults = document.getElementById("inpaint-results");
            const inpaintError = document.getElementById("inpaint-error");
            const inpaintButton = inpaintForm.querySelector('button[type="submit"]');

            inpaintForm.addEventListener("submit", async (event) => {
                event.preventDefault();
                inpaintResults.innerHTML = "";
                inpaintError.textContent = "";
                inpaintButton.disabled = true;
                inpaintButton.textContent = "Running...";

                const imageFile = document.getElementById("inpaint-image").files[0];
                const maskFile = document.getElementById("inpaint-mask").files[0];
                const prompt = document.getElementById("inpaint-prompt").value || "";
                if (!imageFile || !maskFile) return;

                const formData = new FormData();
                formData.append("image", imageFile);
                formData.append("mask", maskFile);
                formData.append("prompt", prompt);

                try {
                    const resp = await fetch("/api/inpaint", { method: "POST", body: formData });
                    if (!resp.ok) {
                        const payload = await resp.json().catch(() => ({}));
                        throw new Error(payload.detail || `Request failed: ${resp.status}`);
                    }
                    const json = await resp.json();
                    const figure = document.createElement("figure");
                    const img = document.createElement("img");
                    img.src = `data:image/png;base64,${json.image}`;
                    const caption = document.createElement("figcaption");
                    caption.textContent = "Inpaint Result";
                    figure.appendChild(img);
                    figure.appendChild(caption);
                    inpaintResults.appendChild(figure);
                } catch (err) {
                    inpaintError.textContent = err.message || String(err);
                } finally {
                    inpaintButton.disabled = false;
                    inpaintButton.textContent = "Run inpaint";
                }
            });
        </script>
    </body>
</html>
"""
