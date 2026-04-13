# Build Windows EXE

## Run the GUI in development

From the project root:

```powershell
Set-Location src
../.venv/Scripts/python.exe gui_app.py
```

## Install packager

```powershell
../.venv/Scripts/python.exe -m pip install -r requirements.txt
../.venv/Scripts/python.exe -m pip install pyinstaller
```

## Build one-file EXE

From the project root:

```powershell
Set-Location src
../.venv/Scripts/pyinstaller.exe --onefile --windowed --name HeuristicsUI gui_app.py
```

The generated executable is in:

- `src/dist/HeuristicsUI.exe`

## Important for instances

The app supports:

- Instance names like `R108` when running from source with the `Archive/` folder available.
- Direct file selection with the `Browse file` button (recommended for EXE usage).

When distributing the EXE, keep your instance `.txt` files accessible and load them through `Browse file`.
