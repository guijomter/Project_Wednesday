Instrucciones

1) Correr en forma secuencial o paralela main.py cambiando los siguientes parámetros en conf.yaml entre ejecuciones:
    A) STUDY_NAME : "Compe_03_V3_1s"  

    competencia03:
    MES_TRAIN: ["201901","201902","201903","201904","201905","201906","202001","202002","202003","202004","202005","202006","202101","202102","202103"]
    MES_TEST: "202107"
    MES_VALIDACION: "202104"
    FINAL_TRAIN: ["201901","201902","201903","201904","201905","201906","202001","202002","202003","202004","202005","202006","202101","202102","202103","202104","202105","202106"] 
    FINAL_PREDIC: "202109" 

    parametros_lgb:
    undersampling: 0.075        
    undersampling_final: 0.3

    B) STUDY_NAME : "Compe_03_V3_4m"  

    competencia03:
    MES_TRAIN: ["201906","201907","201908","201909","202007","202008","202009"]
    MES_TEST: "202107"
    MES_VALIDACION: "202104"
    FINAL_TRAIN: ["201906","201907","201908","201909","202007","202008","202009","202106","202107"] 
    FINAL_PREDIC: "202109"

    parametros_lgb:
    undersampling: 0.075        
    undersampling_final: 0.3

    C) STUDY_NAME : "Compe_03_V3_2q"  

    competencia03:
    MES_TRAIN: ["201904","201905","201906","202004","202005","202006"] 
    MES_TEST: "202107"
    MES_VALIDACION: "202104"
    FINAL_TRAIN: ["201904","201905","201906","202004","202005","202006","202104","202105","202106"]
    FINAL_PREDIC: "202109"  

    parametros_lgb:
    undersampling: 0.05        
    undersampling_final: 0.3

    D) STUDY_NAME : "Compe_03_V3_3q"  

    competencia03:
    MES_TRAIN: ["201907","201908","201909","202007","202008","202009"]
    MES_TEST: "202107"
    MES_VALIDACION: "202104"
    FINAL_TRAIN: ["201907","201908","201909","202007","202008","202009","202104","202107"] 
    FINAL_PREDIC: "202109"

    parametros_lgb:
    undersampling: 0.05        
    undersampling_final: 0.3

    E) STUDY_NAME : "Compe_03_V3_4q"  

    competencia03:
    MES_TRAIN: ["201910","201911","201912","202010","202011","202012"] 
    MES_TEST: "202107"
    MES_VALIDACION: "202104"
    FINAL_TRAIN: ["201910","201911","201912","202010","202011","202012", "202104"] 
    FINAL_PREDIC: "202109"  

    parametros_lgb:
    undersampling: 0.05        
    undersampling_final: 0.3



2) Mover los archivos en la carpeta /resultados con el formato "predicciones_probabilidad_Compe_03_{MODELO}_varias_seeds_{FECHA-HORA}.csv" al subdirectorio /predicciones_mejores
3) Ejecutar la notebook "Predicciones_finales.ipynb" y se obtendrá el archivo de salida "entrega_final_formato_bot.csv"
