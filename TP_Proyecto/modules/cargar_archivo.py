import numpy as np
import scipy.io as sio
import os

def extraer_y_guardar_modelo(archivo_mat="iMM904.mat", archivo_salida="imm904_procesado.npz"):
    # if not os.path.exists(archivo_mat):
    #     raise FileNotFoundError(f"No se encontró el archivo '{archivo_mat}' en el directorio. "
    #                             f"Asegúrate de descargarlo de BiGG Models y ponerlo aquí.")
    # Construir la ruta relativa al directorio del script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    archivo_mat_path = os.path.join(script_dir, archivo_mat)
    
    if not os.path.exists(archivo_mat_path):
        raise FileNotFoundError(f"No se encontró el archivo '{archivo_mat}' en {archivo_mat_path}. "
                                f"Asegúrate de descargarlo de BiGG Models y ponerlo en la carpeta modules/.")
    
    print(f"Cargando {archivo_mat}...")
    # mat_data = sio.loadmat(archivo_mat)
    mat_data = sio.loadmat(archivo_mat_path)
    # Extraer la estructura interna de BiGG
    modelo_core = mat_data['iMM904'][0, 0]
    
    # 1. Matriz estequiométrica S (convertida de sparse/dispersa a matriz densa normal)
    # Dependiendo de la versión de scipy, puede requerir .toarray() o .todense()
    try:
        S_matrix = modelo_core['S'].toarray()
    except AttributeError:
        S_matrix = np.array(modelo_core['S'])
        
    # 2. Límites de flujos (vectores columna estirados a 1D usando flatten)
    lower_bounds = modelo_core['lb'].flatten().astype(float)
    upper_bounds = modelo_core['ub'].flatten().astype(float)
    
    # 3. Limpieza de Identificadores (reacciones y metabolitos)
    # MATLAB guarda los strings en celdas complejas; aquí los convertimos a listas de strings de Python
    reacciones_ids = [str(r[0][0]).strip() if len(r[0]) > 0 else "" for r in modelo_core['rxns']]
    metabolitos_ids = [str(m[0][0]).strip() if len(m[0]) > 0 else "" for m in modelo_core['mets']]
    
    # 4. Guardar todo en el formato compacto .npz de NumPy
    np.savez(archivo_salida, 
             S=S_matrix, 
             rxns=np.array(reacciones_ids), 
             mets=np.array(metabolitos_ids), 
             lb=lower_bounds, 
             ub=upper_bounds)
    
    print(f"¡Procesamiento completado con éxito!")
    print(f"-> Archivo generado: '{archivo_salida}'")
    print(f"-> Dimensiones de la Matriz S: {S_matrix.shape} (Metabolitos x Reacciones)")
    print(f"-> Total de reacciones mapeadas: {len(reacciones_ids)}")

if __name__ == "__main__":
    # Al ejecutar este script directamente, procesará el modelo
    extraer_y_guardar_modelo()
