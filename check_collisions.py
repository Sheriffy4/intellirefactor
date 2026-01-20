from pathlib import Path

pkg_dir = Path('intellirefactor')
collisions = []

print('Checking for name collisions...')
for d in pkg_dir.rglob('*'):
    if d.is_dir():
        # Get .py files in directory
        items = {p.stem for p in d.glob('*.py')}
        # Get subdirectories (excluding __pycache__)
        dirs = {p.name for p in d.iterdir() if p.is_dir() and not p.name.startswith('__')}
        
        # Find intersections
        for name in sorted(items & dirs):
            collisions.append(f'Collision in {d}: both \'{name}.py\' and \'{name}/\' exist')

print('Collisions found:')
for collision in collisions:
    print(f' - {collision}')

if not collisions:
    print('No name collisions found!')