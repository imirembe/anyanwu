# Optional: set True only if you want the small female-demo ZIP on your laptop.
DOWNLOAD_RESULTS = False
if DOWNLOAD_RESULTS:
    import json
    import zipfile
    from google.colab import files
    manifest = json.loads((PREVIEW_ROOT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["voice"] == "waxal_ibo_5_female" and len(manifest["clips"]) == 3
    archive = PREVIEW_ROOT.parent / "Anyanwu-female-publisher-demos.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for index, clip in enumerate(manifest["clips"], 1):
            assert clip["filename"] == f"female-igbo-{index:02}.mp3"
            z.write(PREVIEW_ROOT / clip["filename"], clip["filename"])
        z.write(PREVIEW_ROOT / "manifest.json", "manifest.json")
    assert archive.stat().st_size <= 3 * 1024**2
    files.download(str(archive))
