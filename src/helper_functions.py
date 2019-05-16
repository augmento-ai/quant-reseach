
def decode_bytes(data):
	if isinstance(data, bytes):		return data.decode()
	if isinstance(data, str):		return str(data)
	if isinstance(data, (int, float)):		return data
	if isinstance(data, dict):		return dict(map(decode_bytes, data.items()))
	if isinstance(data, tuple):		return tuple(map(decode_bytes, data))
	if isinstance(data, list):		return list(map(decode_bytes, data))
	if isinstance(data, set):		return set(map(decode_bytes, data))