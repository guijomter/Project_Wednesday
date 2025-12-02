1) Reemplazar los archivos features.yaml y conf.yaml por los que están en la carpeta "Experimentos"
2) Ejecutar main_ss.py con las siguientes configuraciones en conf.yaml:
    a) Familia de exp 1:
        -  undersampling y undersampling_final: 0.025
        -  undersampling y undersampling_final: 0.05
        -  undersampling y undersampling_final: 0.075
        -  undersampling y undersampling_final: 0.01
    b) Familia de exp 2: 
        Para todos los experimentos undersampling_final: 0.3  y 
        -  undersampling: 0.025
        -  undersampling: 0.05
        -  undersampling: 0.075
        -  undersampling: 0.01

3) Analizar logs para ver resultados. Hay una copia de los mismos en la carpeta Experimentos