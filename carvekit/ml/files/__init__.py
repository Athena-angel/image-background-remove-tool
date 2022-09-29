from pathlib import Path

carvekit_dir = Path.home().joinpath('.carvekit')

if carvekit_dir.exists():  # backwards compatibility. Moved from home to cache to reduce clutter in home dir
    print('Moving models from home to .cache (backwards compatibility fix)')
    carvekit_dir.rename(Path.home().joinpath('.cache/carvekit'))

carvekit_dir = Path.home().joinpath('.cache/carvekit')

checkpoints_dir = carvekit_dir.joinpath('checkpoints')
