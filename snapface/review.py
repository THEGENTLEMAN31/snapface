"""Revue automatique de rendu : copie de revue + jugement d'un modèle vision.

Le modèle ne remplace pas une mesure : il sert de second avis sur ce que les
chiffres ne captent pas (ressemble-t-il encore à un visage, la déformation
est-elle lisible, la zone dérive-t-elle).
"""
import base64
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import urllib.error
import urllib.request

GEMINI_MODEL = "gemini-3.5-flash-lite"
GEMINI_URL = ("https://generativelanguage.googleapis.com/v1beta/models/"
              f"{GEMINI_MODEL}:generateContent")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "nvidia/nemotron-3.5-super-vl"

SCHEMA = {
    "visage_detecte": "bool",
    "deformation_lisible": "bool",
    "zone_stable": "stable|derive|saute",
    "couture_visible": "bool",
    "artefacts": ["liste courte de défauts visibles"],
    "verdict": "ok|a_corriger",
    "commentaire": "une phrase",
}

PROMPT = (
    "Tu contrôles la qualité d'un effet de déformation faciale temps réel. "
    "Analyse la séquence fournie (frames dans l'ordre chronologique). "
    "Réponds UNIQUEMENT avec un objet JSON valide, sans texte autour, au format : "
    + json.dumps(SCHEMA, ensure_ascii=False)
    + ". Critères : zone_stable vaut 'derive' si la zone déformée glisse par rapport au visage, "
      "'saute' si elle change brutalement de taille ou de position ; couture_visible est vrai "
      "si on voit un bord net, un rectangle ou un halo autour de la zone traitée."
)


def _post(url: str, body: dict, key: str, timeout: int = 300) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {key}"} if key else
                                 {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise ValueError(f"HTTP {e.code}: {e.read().decode()[:300]}") from None


def review_copy(src: Path, dst: Path, width: int = 640, fps: float = 12.0) -> Path:
    """Copie légère (largeur et cadence réduites) pour l'envoi au modèle."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return Path(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", str(src),
           "-vf", f"fps={fps},scale={width}:-2", "-c:v", "libx264", "-crf", "28",
           "-an", str(dst)]
    if subprocess.run(cmd, capture_output=True).returncode != 0:
        return Path(src)
    return dst


def _extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return {"verdict": "a_corriger", "commentaire": text[:200]}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"verdict": "a_corriger", "commentaire": text[:200]}


def review_video(path: Path, workdir: Path | None = None,
                 model: str = GEMINI_MODEL) -> dict:
    """Envoie la vidéo (native, inline) à Gemini et retourne le verdict JSON."""
    key = os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise ValueError("GOOGLE_API_KEY absente de l'environnement")
    small = review_copy(Path(path), (workdir or Path("/tmp/opencode")) / "review.mp4")
    data = base64.b64encode(Path(small).read_bytes()).decode()
    if len(data) > 18_000_000:
        raise ValueError("Copie de revue trop volumineuse (> 18 Mo en base64)")
    body = {"contents": [{"parts": [
        {"inline_data": {"mime_type": "video/mp4", "data": data}},
        {"text": PROMPT}]}],
        "generationConfig": {"maxOutputTokens": 900, "temperature": 0.1}}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    out = _post(url, body, key="")
    text = "".join(p.get("text", "") for p in out["candidates"][0]["content"]["parts"])
    verdict = _extract_json(text)
    verdict["_model"] = model
    verdict["_source"] = str(small)
    return verdict


def review_image(path: Path, backend: str = "gemini",
                 model: str | None = None) -> dict:
    """Envoie une image fixe à Gemini ou à OpenRouter."""
    data = base64.b64encode(Path(path).read_bytes()).decode()
    if backend == "openrouter":
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise ValueError("OPENROUTER_API_KEY absente de l'environnement")
        body = {"model": model or OPENROUTER_MODEL, "max_tokens": 3000, "messages": [{
            "role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{data}"}},
                {"type": "text", "text": PROMPT}]}]}
        out = _post(OPENROUTER_URL, body, key)
        text = out["choices"][0]["message"].get("content") or ""
        return _extract_json(text)
    key = os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise ValueError("GOOGLE_API_KEY absente de l'environnement")
    body = {"contents": [{"parts": [
        {"inline_data": {"mime_type": "image/png", "data": data}},
        {"text": PROMPT}]}],
        "generationConfig": {"maxOutputTokens": 900, "temperature": 0.1}}
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model or GEMINI_MODEL}:generateContent?key={key}")
    out = _post(url, body, key="")
    text = "".join(p.get("text", "") for p in out["candidates"][0]["content"]["parts"])
    return _extract_json(text)


def summarize(verdict: dict) -> str:
    """Rend le verdict lisible en une ligne."""
    ok = verdict.get("verdict") == "ok"
    bits = [
        f"visage={'oui' if verdict.get('visage_detecte') else 'non'}",
        f"deformation={'lisible' if verdict.get('deformation_lisible') else 'illisible'}",
        f"zone={verdict.get('zone_stable', '?')}",
        f"couture={'oui' if verdict.get('couture_visible') else 'non'}",
    ]
    extra = verdict.get("artefacts") or []
    if isinstance(extra, list) and extra:
        bits.append("artefacts=" + ", ".join(str(a) for a in extra[:3]))
    return ("[OK] " if ok else "[KO] ") + " | ".join(bits)
