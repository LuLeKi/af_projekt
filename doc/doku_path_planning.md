Dokumentation: PathPlanning Modul

Einleitung

Das PathPlanning Modul ist verantwortlich für die Planung eines Fahrzeugs auf Basis erkannter Fahrspurbegrenzungen in der Gymnasium Car Racing Simulation. Es sorgt für eine stabile, geglättete Trajektorie und passt den Pfad dynamisch an die Straßenkrümmung an.

Funktionsweise

1. Initialisierung

Beim Erzeugen eines PathPlanning-Objekts werden wichtige Statusvariablen initialisiert:

_last_debug_time: Zeitpunkt der letzten Debug-Ausgabe.

vehicle_position: Annahme einer Startposition (x=48.0, y=64.0).

last_valid_waypoints: Letzter gültiger Pfad.

last_valid_curvature: Letzte gültige Krümmung.

invalid_counter: Zähler für fehlerhafte Spurinformationen.

track_finished: Flag für Streckenende.

2. Planung des Pfades (plan)

Überprüfung auf Streckenende oder ungültige Fahrspuren.

Berechnung der Mittellinie (Waypoints) zwischen linker und rechter Spurbegrenzung.

Glättung der Punkte mit Spline-Interpolation.

Berechnung der Krümmung.

Normierung der Krümmung für die weitere Verarbeitung.

Ermittlung der Normalenvektoren entlang der Mittellinie.

Extrahierung eines lokalen Pfadabschnitts.

Dynamische Anpassung und erneute Glättung des lokalen Pfads.

Rückgabe: angepasster lokaler Pfad (np.ndarray) und normierte Krümmung (float).

3. Kernmethoden

lanes_are_valid

Prüft, ob die linke und rechte Spur plausibel und synchron verlaufen.

build_waypoints

Berechnet die Mittelpunkte zwischen linker und rechter Spurbegrenzung. Wenn ausreichend Punkte vorhanden sind, wird ein Spline über die Punkte gelegt.

handle_invalid_data

Behandelt fehlerhafte Spurdaten.

Bei kurzen Ausfällen wird ein Dummy-Pfad generiert.

Bei langanhaltendem Fehlschlagen wird ein Notfall-Stopp-Pfad erzeugt.

generate_emergency_stop_path / generate_dummy_path

Erzeugt Notfallpfade:

Stop-Path: Hält Fahrzeug sofort an.

Dummy-Path: Rollt Fahrzeug leicht nach vorne.

update_valid_state

Aktualisiert gespeicherten gültigen Pfad und setzt Fehlerzähler zurück.

calculate_curvature

Berechnet mittlere absolute Krümmung aus der zweiten Ableitung der Trajektorie (Spline).

normalize_curvature

Normiert die Krümmung zwischen 0 (gerade Strecke) und 1 (sehr enge Kurve).

calculate_normals

Bestimmt die Normalenvektoren entlang der Trajektorie zur späteren dynamischen Pfadverschiebung.

extract_local_path

Extrahiert einen Abschnitt des Pfades, beginnend am nächstgelegenen Punkt zum Fahrzeug.

find_closest_index

Findet den Index des Pfadpunkts, der dem Fahrzeug am nächsten ist.

adjust_and_smooth_path

Verschiebt den lokalen Pfad basierend auf der Kurvenkrümmung leicht entlang der Normalenvektoren und glättet ihn erneut.

print_debug_info

Gibt einmal pro Sekunde nützliche Debug-Informationen über den Pfad und Krümmung auf der Konsole aus