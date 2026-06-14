import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, linprog

#=============================================================================
# CARGA DE DATOS DEL MODELO
#=============================================================================

try:
    datos_modelo = np.load('imm904_procesado.npz', allow_pickle=True)
except FileNotFoundError:
    print("Error: No se encontró 'imm904_procesado.npz'.")
    print("Por favor, ejecuta primero 'python procesar_modelo.py' para generar las matrices.")
    exit()

# Estas son las variables globales que usaremos en las funciones matemáticas
S = datos_modelo['S']
RXNS = list(datos_modelo['rxns'])  # Convertimos a lista para usar .index() fácilmente
METS = list(datos_modelo['mets'])
LB_ORIGINAL = datos_modelo['lb']
UB_ORIGINAL = datos_modelo['ub']

print(f"Modelo cargado: {S.shape[0]} metabolitos, {S.shape[1]} reacciones.")

# =============================================================================
# DICCIONARIO DE CONFIGURACIÓN — esto lo editamos dependiendo que queremos mostrar
# =============================================================================
CONFIG = {
    # ── Condición experimental ──────────────────────────────────────────────
    # Opciones: "anaerobic" o "aerobic"
    "condicion": "anaerobic",
 
    # ── Qué modelos mostrar en la gráfica ───────────────────────────────────
    "mostrar": {
        "experimental":    True,   # puntos reales del paper
        "clasico_biomasa": True,   # FBA tradicional (maximizar biomasa)
        "paper_original":  True,   # paper con w = [1, 1, 1, 1]
        "adaptativo":      True,   # nuestra propuesta de pesos ajustables
    },
 
    # ── Variable a graficar en el eje Y ─────────────────────────────────────
    # Opciones: "crecimiento" o "etanol"
    "variable_y": "crecimiento",
 
    # ── Pesos por compartimento para el modo ADAPTATIVO ─────────────────────
    # Estos son los w_k del problema (7) del paper: max w^T * C^T * v
    # Cada peso representa cuánto contribuye ese compartimento al objetivo.
    # El script los normaliza automáticamente (suma = 1).
    #
    # Biología detrás de cada compartimento:
    #   citosol      (P1): minimizar prod. NADH → fuerza desvío a etanol
    #   mitocondria  (P2): minimizar cons. NADH/NADPH → eficiencia respiratoria
    #   peroxisoma   (P3): maximizar prod. ácidos grasos → crecimiento lento
    #   virtual      (P4): maximizar biomasa → objetivo clásico
    #
    #   Anaeróbico óptimo  → citosol=0.55, mitocondria=0.05, peroxisoma=0.10, virtual=0.50
    #   Aeróbico óptimo    → citosol=0.48, mitocondria=0.35, peroxisoma=0.08, virtual=0.45
    #   Paper original     → todos = 0.25 (equivalente a w = [1,1,1,1])
    "pesos": {
        "citosol":     0.55,
        "mitocondria": 0.05,
        "peroxisoma":  0.10,
        "virtual":     0.50,
    },
 
    # ── Optimizar pesos automáticamente ─────────────────────────────────────
    # Si True: ignora los pesos de arriba y los busca minimizando el error
    # Si False: usa exactamente los pesos que pusiste arriba
    "optimizar_pesos": False, 
}

# =============================================================================
# DATOS EXPERIMENTALES (Tablas 2 y 3 del paper)
# =============================================================================
DATOS_EXPERIMENTALES = {
    # Nissen et al. (1997)
    # Columnas: glucosa [mmol/gPS·h], etanol [mmol/gPS·h], crecimiento [1/h]
    "anaerobic": np.array([
        [ 5.56,  8.28, 0.10],
        [11.50, 17.12, 0.20],
        [17.37, 25.74, 0.30],
        [23.65, 35.27, 0.40],
    ]),
    # Heyland et al. (2009) 
    "aerobic": np.array([
        [ 7.2,  9.0, 0.16],
        [10.2, 11.2, 0.17],
        [12.2, 15.6, 0.23],
        [12.3, 15.2, 0.21],
        [15.1, 20.1, 0.33],
        [18.4, 28.2, 0.36],
        [19.9, 29.6, 0.40],
        [20.2, 30.0, 0.40],
    ]),
}
# =============================================================================
# IDENTIFICADORES DE REACCIONES — esto no lo editamos, es para mapear las reacciones del modelo
# =============================================================================

#Los ID son case sensitives y tienen coincidir exactamente con los del modelo
#   _c = citosol   _m = mitocondria   _x = peroxisoma   _e = extracelular
REA = {
    "biomasa":  "BIOMAS_SC5_notrace",   # reacción de crecimiento
    "glucosa":  "EX_glc__D_e",          # consumo de glucosa
    "etanol":   "EX_etoh_e",            # producción de etanol
    "oxigeno":  "EX_o2_e",              # intercambio O2
    "co2":      "EX_co2_e",             # intercambio CO2
    "mantenimiento": "ATPM",            # mantenimiento de ATP
    # Compartimentos para ergosterol/ácidos grasos (anaeróbico)
    "ergosterol":   "EX_ergst_e",
    "zymosterol":   "EX_zymst_e",
    "hdcea":        "EX_hdcea_e",
    "ocdca":        "EX_ocdca_e",
    "ocdcea":       "EX_ocdcea_e",
    "ocdcya":       "EX_ocdcya_e",
}

# Función objetivo por compartimento  (sign: +1 max, -1 min)
OBJ_COMPARTIMENTOS = {
    # P1 – citosol: minimizar producción de NADH  (mejor para anaeróbico)
    "citosol": [
        ("NADH2_u6m",   -1),   # NADH deshidrogenasa (aprox citosol)
        ("GAPD",         -1),   # gliceraldehído-3-P deshidrogenasa → NADH
        ("PDHm",         -1),   # piruvato deshidrogenasa → NADH
        ("CSm",           -1),   # citrato sintasa
        ("TPI",          +1),   # triosa fosfato isomerasa
        ("PGK",          +1),   # fosfoglicerato quinasa → ATP
    ],
    # P2 – mitocondria: minimizar consumo de NADH/NADPH  (mejor para aeróbico)
    "mitocondria": [
        ("NADH2_u6m",   -1),
        ("SUCD1m",       -1),   # succinato deshidrogenasa
        ("ICDHyr",      -1),   # isocitrato deshidrogenasa
        ("MDHm",         -1),   # malato deshidrogenasa
        ("AKGDam",       -1),
    ],
    # P3 – peroxisoma: maximizar producción de ácidos grasos
    "peroxisoma": [
        ("r_0658",       +1),   # beta-oxidación (iMM904 IDs varían)
        ("FAO181p_even",   +1),
        ("FAO161p_even",  +1),   # beta-oxidación C16:1
        ("FAO80p",        +1),   # beta-oxidación C8
        ("ACLSm",        +1),
    ],
    # P4 – virtual: maximizar biomasa (objetivo clásico)
    "virtual": [
        ("BIOMAS_SC5_notrace", +1),
    ],
}

# =============================================================================
# VERIFICACIÓN DE IDs AL INICIO
# Avisa si algún ID del diccionario no existe en el modelo real
# =============================================================================
def verificar_ids():
    print("\n── Verificando IDs de reacciones ──")
    todos_ok = True
    # Verificar REA
    for clave, rxn_id in REA.items():
        if rxn_id not in RXNS:
            print(f"REA['{clave}'] = '{rxn_id}' NO encontrado en el modelo")
            todos_ok = False
    # Verificar OBJ_COMPARTIMENTOS
    for comp, lista in OBJ_COMPARTIMENTOS.items():
        for rxn_id, _ in lista:
            if rxn_id not in RXNS:
                print(f"OBJ['{comp}'] → '{rxn_id}' NO encontrado en el modelo")
                todos_ok = False
    if todos_ok:
        print("Todos los IDs verificados correctamente")
    print()

#=============================================================================
# FUNCIONES AUXILIARES
#=============================================================================
def get_idx(rxn_id):
    """Retorna el índice de una reacción, o None si no existe."""
    return RXNS.index(rxn_id) if rxn_id in RXNS else None
 
def normalizar_pesos(pesos: dict) -> dict:
    """Normaliza los pesos para que sumen 1."""
    total = sum(abs(v) for v in pesos.values())
    if total < 1e-10:
        raise ValueError("Todos los pesos son cero.")
    return {k: abs(v) / total for k, v in pesos.items()}

def aplicar_condicion_matemática(condicion: str, glucosa_uptake: float):
    """
    Ajusta los vectores lb y ub según la Tabla 5 del paper usando índices de NumPy.
    Retorna copias modificadas de (lb, ub).
    """
    # 1. Clonar los límites originales para no sobreescribir el modelo base
    lb_mod = np.copy(LB_ORIGINAL)
    ub_mod = np.copy(UB_ORIGINAL)

    # 2. Fijar consumo de Glucosa (Fila 1 de la Tabla 5)
    idx_glc = get_idx(REA["glucosa"])
    if idx_glc is not None:
        # En FBA, el consumo/uptake se modela como flujo negativo
        lb_mod[idx_glc] = -glucosa_uptake
        ub_mod[idx_glc] = -glucosa_uptake  # Forzamos a que sea exactamente el valor experimental

    # 3. Configurar según condiciones Anaeróbica / Aeróbica
    if condicion == "anaerobic":
        # (sin oxigeno) Oxígeno = 0 (Fila 2)
        idx_o2 = get_idx(REA["oxigeno"])
        if idx_o2 is not None:
            lb_mod[idx_o2] = 0.0
            ub_mod[idx_o2] = 0.0
            
        # CO2 libre hacia afuera (Fila 3: lb=0, ub=1000)
        idx_co2 = get_idx(REA["co2"])
        if idx_co2 is not None:
            lb_mod[idx_co2] = 0.0
            ub_mod[idx_co2] = 1000.0
            
        # Nutrientes anaeróbicos: Esteroles y Ácidos Grasos (Fila 4: lb=-1000, ub=1000)
        # Permite que la célula los absorba del medio simulado ya que no los puede fabricar sin O2
        nutrientes_anaerobicos = ["ergosterol", "zymosterol", "hdcea", "ocdca", "ocdcea", "ocdcya"]
        for nut in nutrientes_anaerobicos:
            idx_nut = get_idx(REA[nut])
            if idx_nut is not None:
                lb_mod[idx_nut] = -1000.0
                ub_mod[idx_nut] = 1000.0

    else:  # aerobic
        # Oxígeno libre ilimitado (Fila 5: lb=-1000, ub=0)
        idx_o2 = get_idx(REA["oxigeno"])
        if idx_o2 is not None:
            lb_mod[idx_o2] = -1000.0
            ub_mod[idx_o2] = 0.0  # Solo consumo, la levadura no "exhala" O2 puro

    return lb_mod, ub_mod

#=============================================================================
#Funcionse FBA
#=============================================================================

def fba_clasico(condicion: str, glucosa: float):
    """
    FBA tradicional: Maximiza únicamente la reacción de biomasa.
    """
    idx_bio= get_idx(REA["biomasa"])
    idx_eth= get_idx(REA["etanol"])
    if idx_bio is not None or idx_eth is not None:
        return 0.0, 0.0  # Si no se encuentra alguna de las reacciones clave, retornamos ceros
    
    c_vector = np.zeros(len(RXNS))
    c_vector[idx_bio] = 1.0  # Coeficiente 1 solo a la biomasa

    lb_mod, ub_mod = aplicar_condicion_matemática(condicion, glucosa)
    bounds = list(zip(lb_mod, ub_mod))
    b_eq = np.zeros(S.shape[0])

    res = linprog(c=-c_vector, A_eq=S, b_eq=b_eq, bounds=bounds, method='highs')

    if not res.success:
        return 0.0, 0.0

    return float(res.x[idx_bio]), float(abs(res.x[idx_eth]))  # Crecimiento y producción neta de etanol

def fba_combinado(pesos: dict, condicion: str, glucosa: float):
    """
    Resuelve el FBA con la función objetivo combinada usando scipy.optimize.linprog.
    Implementa el problema (7) del paper: max w^T * C^T * v sujeto a S*v=0 y límites de flujo.
    """
    idx_bio= get_idx(REA["biomasa"])
    idx_eth= get_idx(REA["etanol"])
    if idx_bio is None or idx_eth is None:
        return 0.0, 0.0  # Si no se encuentra alguna de las reacciones clave, retornamos ceros

    pesos_norm = normalizar_pesos(pesos)

    # 2. Construir vector de la función objetivo c
    c_vector = np.zeros(len(RXNS))
    for comp, peso in pesos_norm.items():
        if comp not in OBJ_COMPARTIMENTOS or abs(peso) < 1e-10:
            continue
        for rxn_id, signo in OBJ_COMPARTIMENTOS[comp]:
            idx = get_idx(rxn_id)
            if idx is not None:
                c_vector[idx] += peso * float(signo)

    # 3. Obtener límites modificados para la simulación
    lb_mod, ub_mod = aplicar_condicion_matemática(condicion, glucosa)
    bounds = list(zip(lb_mod, ub_mod))

    # 4. Resolver el problema lineal: S * v = 0
    # b_eq es un vector de ceros del tamaño de las filas de S (metabolitos)
    b_eq = np.zeros(S.shape[0]) 
    
    res = linprog(c=-c_vector, A_eq=S, b_eq=b_eq, bounds=bounds, method='highs')

    if not res.success:
        return 0.0, 0.0

    return float(res.x[idx_bio]), float(abs(res.x[idx_eth]))  # Crecimiento y producción neta de etanol

#=============================================================================
#Optimizacion de pesos
#=============================================================================
def optimizar_pesos(condicion: str, datos_exp: np.ndarray):
    '''Busca los pesos w_k que minimizan el error entre predicciones y datos experimentales
    usando scipy.optimize.minimize. Solo optimiza para la condición seleccionada en CONFIG.'''

    glc_vals = datos_exp[:, 0]  # Columna de glucosa
    crec_exp = datos_exp[:, 2] #crecimento siempre en columna 2
    eth_exp = datos_exp[:, 1] #etanol siempre en columna 1

    #contruimos funcion objetivo
    def funcion_objetivo(w):
        #Proyectar a positicvos y normalizar
        w_pos = np.maximum(w, 0.0)
        total = w_pos.sum()
        if total < 1e-8:
            return 1e6  
        w_norm = w_pos / total
        pesos_dict = {
            "citosol":     w_norm[0],
            "mitocondria": w_norm[1],
            "peroxisoma":  w_norm[2],
            "virtual":     w_norm[3],
        }

        err_crec, err_eth = [], []
        for glc, crec_e, eth_e in zip(glc_vals, crec_exp, eth_exp):
            pred_crec, pred_eth = fba_combinado(pesos_dict, condicion, glucosa=glc)
            err_crec.append(abs(pred_crec - crec_e) / abs(crec_e + 1e-8))  # Evitamos división por cero
            err_eth.append(abs(pred_eth - eth_e) / abs(eth_e + 1e-8))  # Evitamos división por cero
        return np.mean(err_crec) + np.mean(err_eth)

    #Punto de partida(pesos iguales)
    w0 = np.array([0.25, 0.25, 0.25, 0.25])
    result = minimize (funcion_objetivo, w0, method="Nelder-Mead", options={"maxiter": 2000, "xtol": 1e-4, "fatol": 1e-4})

    #Aseguramos valores absolutos no negativos y normalizamos 
    w_opt = np.maximum(result.x, 0.0)
    w_opt /= w_opt.sum()
    return {
        "citosol":     float(w_opt[0]),
        "mitocondria": float(w_opt[1]),
        "peroxisoma":  float(w_opt[2]),
        "virtual":     float(w_opt[3]),
    }

#=============================================================================
#Error promedio
#=============================================================================

def error_promedio(pred: list, exp: np.ndarray):
    """Error relativo promedio |pred - exp| / |exp|. Ignora NaN."""
    pred = np.array(pred, dtype=float)
    exp  = np.array(exp,  dtype=float)
    mask = (np.isfinite(pred)) & (np.isfinite(exp)) & (exp != 0)
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(pred[mask] - exp[mask]) / np.abs(exp[mask])))
    
# =============================================================================
# SIMULACION
# =============================================================================
def simular(condicion: str, datos: np.ndarray, config: dict):
    """
    Corre todos los modos de simulación sobre todos los puntos experimentales.
    """
    glc_vals   = datos[:, 0]
    crec_exp   = datos[:, 2] 
    etoh_exp   = datos[:, 1]
    resultados = {
        "glucosa_exp": glc_vals,
        "crecimiento_exp": crec_exp,
        "etanol_exp":      etoh_exp,
    }
 
    mostrar = config["mostrar"]
    pesos   = config["pesos"]
 
    # FBA clásico (max biomasa)
    if mostrar.get("clasico_biomasa"):
        print (f"\n── Simulando condición: {condicion} con FBA clásico (max biomasa) ...")
        crec_b, etoh_b = [], []
        for glc in glc_vals:
            c, e = fba_clasico(condicion, glc)
            crec_b.append(c)
            etoh_b.append(e)
        resultados["crecimiento_clasico"] = crec_b
        resultados["etanol_clasico"]      = etoh_b
        print(f"    e_gw = {(error_promedio(crec_b, crec_exp)):.3f} "
              f"  e_etoh = {(error_promedio(etoh_b, etoh_exp)):.3f}")
 
    # Paper original (w = [1,1,1,1])
    if mostrar.get("paper_original"):
        print(f"\n── Simulando condición: {condicion} con función objetivo del paper (w=[1,1,1,1]) ...")
        pesos_paper = {"citosol": 1.0, "mitocondria": 1.0,
                       "peroxisoma": 1.0, "virtual": 1.0}
        crec_p, etoh_p = [], []
        for glc in glc_vals:
            c, e = fba_combinado(pesos_paper, condicion, glc)
            crec_p.append(c)
            etoh_p.append(e)
        resultados["crecimiento_paper"] = crec_p
        resultados["etanol_paper"]      = etoh_p
        print(f"    e_gw = {(error_promedio(crec_p, crec_exp)):.3f} "
              f"  e_etoh = {(error_promedio(etoh_p, etoh_exp)):.3f}")
 
    # Adaptativo (pesos personalizados)
    if mostrar.get("adaptativo"):
        if config.get("optimizar_pesos"):
            print ("\n── Optimizando pesos (puede tardar un poco) ...")
            pesos = optimizar_pesos(condicion, datos)
            print(f"Pesos optimizados: {pesos}")
        else:
            pesos = normalizar_pesos(pesos)
            print(f"  Simulando adaptativo con pesos normalizados: "
                  f"{ {k: round(v,3) for k,v in pesos.items()} }")
        crec_a, etoh_a = [], []
        for glc in glc_vals:
            c, e = fba_combinado(pesos, condicion, glc)
            crec_a.append(c)
            etoh_a.append(e)
        resultados["crecimiento_adaptativo"] = crec_a
        resultados["etanol_adaptativo"]      = etoh_a
        print(f"    e_gw = {(error_promedio(crec_a, crec_exp)):.3f} "
              f"  e_etoh = {(error_promedio(etoh_a, etoh_exp)):.3f}")
 
    return resultados
 
#=============================================================================
#Graficas
#=============================================================================


#=============================================================================
#Imprimir resumen
#=============================================================================
def imprimir_resumen(resultados: dict, config: dict):
    condicion = config["condicion"]
    var=config["variable_y"]
    datos=DATOS_EXPERIMENTALES[condicion]
    col =2 if var=="crecimiento" else 1

    e_clasico = error_promedio(resultados.get("crecimiento_clasico"  if var == "crecimiento" else "etanol_clasico",  [np.nan]), datos[:,col])
    e_paper   = error_promedio(resultados.get("crecimiento_paper"    if var == "crecimiento" else "etanol_paper",    [np.nan]), datos[:,col])
    e_adapt   = error_promedio(resultados.get("crecimiento_adaptativo" if var == "crecimiento" else "etanol_adaptativo", [np.nan]), datos[:,col])

    linea = "──" * 55
    print(f"\n{linea}")
    print(f"  RESUMEN — FBA S. cerevisiae")
    print(f"{linea}")
    print(f"  Condición  : {condicion}")
    print(f"  Variable   : {var}")
    print(f"\n  {'Modelo':<28} {'Error':>8}")
    print(f"  {'─'*36}")
    for nombre, error in [("FBA clásico (biomasa)", e_clasico), 
                          ("Función objetivo paper", e_paper), 
                          ("Función objetivo adaptativo", e_adapt)]:
        print(f"  {nombre:<28} {error:>8.3f}" if not np.isnan(error) else f"  {nombre:<28} {'—':>8}")

    pesos= resultados.get("pesos_adaptativos", normalizar_pesos(config.get("pesos")))
    print(f"\n  Pesos adaptativos (normalizados):") 
    for comp, val in pesos.items():
        barras = '█' * int(val * 30)
        print(f"    {comp:<14}: {val:.3f} {barras}")
    print(f"{linea}\n")