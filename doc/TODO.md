1. 






2. path_planning.py
        ________________________________________________________________________________________________________
        1. Mitte der Fahrspur als Pfad                                                                      ✅ 
            Für jeden Punkt der linken und rechten Begrenzung den Mittelpunkt bilden                           
        ________________________________________________________________________________________________________
        2. Menge von Punkten Fahrspur anzeigen lassen                                                       ✅
            Die Mittelpunkte als np.ndarray sauber aufbereiten (Shape beachten)
        ________________________________________________________________________________________________________
        2.1 test_path_planning.py ausführen:                                                                ✅
            Prüfen, ob die Punkte in der Visualisierung erscheinen (weiße Punkte in der Mitte).
        ________________________________________________________________________________________________________
        3. Trajektorie als Funktion erstellen                                                               ✅
            Punkte glätten: z. B. mittels Spline-Interpolation (scipy.interpolate.splprep, splev)
        ________________________________________________________________________________________________________
        3.1 test_path_planning.py ausführen                                                                 ✅
            Prüfen, ob der Pfad glatter und weicher aussieht.
        ________________________________________________________________________________________________________
        4. Krümmung der Spur/Kurve berchnen                                                                 ✅
            Krümmung aus der Trajektorie ableiten: z. B. zweite Ableitung oder geometrische Approximation.
        ________________________________________________________________________________________________________
        5. Trajektorie anhand der Krümmung anpassen                                                          ✅
            Pfad leicht nach innen oder außen verschieben, um Kurven besser zu schneiden.                
        ________________________________________________________________________________________________________
        5.1 test_path_planning.py ausführen                                                                  ✅                                                              
            Testen, ob Auto stabiler durch Kurven fährt(manuell)
        ________________________________________________________________________________________________________