## 0. Requirements
- Python >3.12
## 1 Installation in virtueller Umgebung
1. python -m venv .venv
    (remove:Remove-Item -Recurse -Force .venv # powershell)
2. .venv\Scripts\activate.bat   #cmd
   . .venv\Scripts\Activate    #powershell
    (deactivate venv: deactivate)
3. In vscode über `strg+shift+P` nach `Python: Select Interpreter` suchen und `.venv` als Interpreter auswählen.
4. pip install box2d
5. pip install "gymnasium[box2d2]" numpy matplotlib scipy opencv-python
6. pip install pygame

7. Testen der Installation:
    python test_installation.py

## 3. Ausführung

Die test Dateien können mit `python <test-file>.py` ausgeführt werden. Die Simulation wird sich öffnen und das Fahrzeug wird sich entsprechend der Pipeline bewegen. Um das Endergebnis zu simulieren, kann die `main.py` Datei ausgeführt werden. Sie führt die Pipeline über 5 Iterationen aus und berechnet den durchschnittlichen Score.

python test_installation.py
python test_lane_detection.py
python test_lateral_control.py
python test_longitudinal_control.py
python test_path_planning.py
python test_pipeline.py

python test_wasd
(test_lane_detection.py ohne lanedetection funktionen um einfach mit wasd fahren zu können)

## 4. Hinweise

Die Dateien `main.py`, `env_wrapper.py` und `input_controller.py` dürfen nicht verändert werden, damit die Ergebnisse vergleichbar bleiben. Alle anderen Dateien können beliebig verändert werden.