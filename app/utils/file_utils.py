import tempfile

def safe_temp_file(suffix: str, content: bytes) -> str:
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_file.write(content)
    tmp_file.close()
    return tmp_file.name
