"""
Predefined style to SD inpainting prompt mapping.
Each style targets kitchen surfaces: floor, cabinets, backsplash, countertop.
"""

STYLE_CONFIG = {
    "italian_marble": {
        "label": "Italian Marble",
        "prompt": (
            "photorealistic kitchen with Italian Carrara marble surfaces, "
            "white marble with grey veining on countertops cabinets and floor, "
            "polished stone, elegant, luxury, soft lighting, interior photography"
        ),
        "negative_prompt": (
            "wood, dark colors, rustic, cartoon, painting, illustration, "
            "blurry, low quality, watermark"
        ),
        "strength": 0.80,
        "guidance_scale": 8.5,
        "num_inference_steps": 30,
    },
    "wooden_rustic": {
        "label": "Wooden Rustic",
        "prompt": (
            "photorealistic rustic farmhouse kitchen with warm oak wood surfaces, "
            "natural wood grain cabinets and countertops, reclaimed wood flooring, "
            "cozy, warm lighting, interior photography"
        ),
        "negative_prompt": (
            "marble, metal, modern, glossy, dark, cartoon, painting, blurry, "
            "low quality, watermark"
        ),
        "strength": 0.80,
        "guidance_scale": 8.5,
        "num_inference_steps": 30,
    },
    "modern_white": {
        "label": "Modern White",
        "prompt": (
            "photorealistic contemporary minimalist kitchen with clean white surfaces, "
            "white lacquered cabinets, white quartz countertops, white tile backsplash, "
            "bright, airy, Scandinavian design, interior photography"
        ),
        "negative_prompt": (
            "wood grain, rustic, dark colors, ornate, cartoon, painting, blurry, "
            "low quality, watermark"
        ),
        "strength": 0.82,
        "guidance_scale": 8.0,
        "num_inference_steps": 30,
    },
    "dark_luxury": {
        "label": "Dark Luxury",
        "prompt": (
            "photorealistic luxury dark kitchen with matte black cabinets, "
            "dark marble countertops, deep charcoal and obsidian surfaces, "
            "dramatic moody lighting, high-end design, interior photography"
        ),
        "negative_prompt": (
            "white, bright, rustic, wood, light colors, cartoon, painting, blurry, "
            "low quality, watermark"
        ),
        "strength": 0.82,
        "guidance_scale": 9.0,
        "num_inference_steps": 30,
    },
}
