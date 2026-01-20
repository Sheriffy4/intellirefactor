import importlib
import pkgutil
import traceback
import intellirefactor

def iter_modules(pkg):
    prefix = pkg.__name__ + "."
    for m in pkgutil.walk_packages(pkg.__path__, prefix):
        yield m.name

bad = []
for name in iter_modules(intellirefactor):
    try:
        importlib.import_module(name)
    except Exception as e:
        bad.append((name, e, traceback.format_exc()))

print("BAD:", len(bad))
for name, e, tb in bad[:50]:
    print("\n===", name, "===\n", tb)