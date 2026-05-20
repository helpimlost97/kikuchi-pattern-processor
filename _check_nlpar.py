from pyebsdindex import ebsd_pattern
import inspect

print("=== write_header source ===")
print(inspect.getsource(ebsd_pattern.UPFile.write_header)[:2000])
print("\n=== write_data source ===")
print(inspect.getsource(ebsd_pattern.UPFile.write_data)[:3000])
