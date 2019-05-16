import msgpack
import zlib

# define the location of the input file
filename_augmento_data = "data/example_data/augmento_data.msgpack.zlib"

# load the data
with open(filename_augmento_data, "rb") as f:
	raw_data = msgpack.unpackb(zlib.decompress(f.read()))

print(raw_data[0:3])