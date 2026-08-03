import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation# Para la animación
# ─────────────────────────────────────────────────────────────────────────────
# PARÁMETROS DE LA SIMULACIÓN
# ─────────────────────────────────────────────────────────────────────────────
T  = 0.001   # Periodo de integración [s]: cuamto tiempo avanza cada iteración
 
# Pesos que regulan cuánto influye cada regla en la velocidad final
WA = 0.1     # Peso de alineación  
WS = 0.1     # Peso de separación  
WC = 0.1     # Peso de cohesión    
 
# Radios de vecindad: distancia maxima para considerar a otro boid como vecino
RHO_A = 4.0  # Radio de alineación  [m]
RHO_S = 0.5  # Radio de separación  [m] --> tiene que ser menor que RHO_C para que haya agrupamiento
RHO_C = 4.0  # Radio de cohesión    [m]
 
# Velocidad inicial tipica de un ave ( aprox 5 m/s, valor sugerido en el enunciado)
V0 = 5.0     # [m/s]
 
# Límite de velocidad máxima (evita que los boids se disparen al infinito)
V_MAX = 15.0  # [m/s]
 
# Dominio espacial observable [m] --> region 2D donde se mueven los boids
X_MIN, X_MAX = 0.0, 50.0
Y_MIN, Y_MAX = 0.0, 50.0
# historial=[] # Para guardar las posiciones de los boids en cada paso y luego graficar la trayectoria

# ─────────────────────────────────────────────────────────────────────────────
# CLASE BOID

class Boid:
    def __init__(self, pos, vel):
        self.pos = np.array(pos, dtype=float)  # Posición del boid (vector 2D)
        self.vel = np.array(vel, dtype=float)  # Velocidad del boid (vector 2D)

    # ─────────────────────────────────────────────────────────────────────────────
    # FUNCIONES DE LAS REGLAS DE COMPORTAMIENTO
    # ─────────────────────────────────────────────────────────────────────────────
    def calcular_alineacion(self, todos):
        """
    Regla de alineación --> a_i = (1 / #A_i) * Σ_{j ∈ A_i} v_j  −  v_i
 
    Empuja al boid para que iguale su velocidad con el promedio de sus vecinos
    dentro del radio RHO_A
    Retorna el vector de influencia (0 si no hay vecinos)
    """
        vecinos_vel = [
            otro.vel for otro in todos
            if otro is not self
            and np.linalg.norm(self.pos - otro.pos) < RHO_A
        ]
 
        # for j, otro in enumerate(todos):
        #     if otro is self:
        #         continue  # Un boid no es vecino de si mismo
 
        #     distancia = np.linalg.norm(self.pos - otro.pos)  # Distancia euclidea [m]
 
        #     if distancia < RHO_A:  # El boid j esta dentro de la vecindad de alineación
        #         vecinos_vel.append(otro.vel)
 
        if not vecinos_vel:
            return np.zeros(2)  # Sin vecinos --> influencia nula (boid sigue su camino sin cambiar)
 
        velocidad_promedio = np.mean(vecinos_vel, axis=0)  # Velocidad media de los vecinos
        return velocidad_promedio - self.vel                # Diferencia respecto a la propia
 
    def calcular_separacion(self, todos):
        """
        Regla de separación --> s_i = Σ_{j ∈ S_i} (x_i − x_j)
    
        Aleja al boid de sus vecinos muy cercanos (radio RHO_S) para evitar colisiones
        La dirección del vector apunta DESDE el vecino HACIA el boid --> lo empuja a alejarse
        Retorna el vector de influencia (0 si no hay vecinos)
        """
        influencia = np.zeros(2)
    
        for otro in todos:
            if otro is self:
                continue
    
            distancia = np.linalg.norm(self.pos - otro.pos)
    
            if 0 < distancia < RHO_S:  # Vecino demasiado cerca
                influencia += self.pos - otro.pos     # Vector que apunta hacia afuera del vecino
    
        return influencia
    
    def calcular_cohesion(self, todos):
        """
        Regla de cohesión --> c_i = (1 / #C_i) * Σ_{j ∈ C_i} x_j  −  x_i
    
        Atrae al boid hacia el centro geométrico de su vecindad (radio RHO_C)
        Opuesta a la separación: en lugar de alejarse, se acerca al grupo
        Retorna el vector de influencia (0 si no hay vecinos)
        """
        vecinos_pos = []
    
        for otro in todos:
            if otro is self:
                continue
    
            distancia = np.linalg.norm(self.pos - otro.pos)
    
            if distancia < RHO_C:  # Vecino dentro de la vecindad de cohesión
                vecinos_pos.append(otro.pos)
    
        if not vecinos_pos:
            return np.zeros(2)  # Sin vecinos --> influencia nula (boid sigue su camino si cambiar)
    
        centro = np.mean(vecinos_pos, axis=0)  # Centro geométrico del grupo
        return centro - self.pos               # Vector que apunta hacia ese centro

    def calcular_fuerza_borde(self):
        """
        Fuerza de retorno cuando un boid sale del área observable (condición de borde)
    
        Segun el TP --> si el boid cruza el limite izquierdo (x < X_MIN),
        se aplica f = [X_MIN - x_i, 0]; si cruza el derecho (x > X_MAX),
        f = [X_MAX - x_i, 0], si cruza el borde inferior (y < Y_MIN), f = [0, Y_MIN - y_i], 
        y si cruza el borde superior (y > Y_MAX), f = [0, Y_MAX - y_i]
        Esta fuerza se suma a la ec 4 como un termino adicional
        """
        fuerza = np.zeros(2)
    
        # Borde izquierdo
        if self.pos[0] < X_MIN:
            fuerza[0] += X_MIN - self.pos[0]
    
        # Borde derecho
        if self.pos[0] > X_MAX:
            fuerza[0] += X_MAX - self.pos[0]
    
        # Borde inferior
        if self.pos[1] < Y_MIN:
            fuerza[1] += Y_MIN - self.pos[1]
    
        # Borde superior
        if self.pos[1] > Y_MAX:
            fuerza[1] += Y_MAX - self.pos[1]
    
        return fuerza

    def actualizar(self,todos):

        vel_prev = self.vel.copy()
        # Calcular las contribuciones de cada regla
        a = self.calcular_alineacion(todos)
        s = self.calcular_separacion(todos)
        c = self.calcular_cohesion(todos)
        f = self.calcular_fuerza_borde()

        # Ecuación (4)
        self.vel = vel_prev + WA*a + WS*s + WC*c + f

        # Limitar velocidad máxima
        velocidad = np.linalg.norm(self.vel)

        if velocidad > V_MAX:
            self.vel = (self.vel / velocidad) * V_MAX

        # Actualizar posición
        self.pos = self.pos + T * vel_prev 
# ─────────────────────────────────────────────────────────────────────────────
# Simulacion
# ─────────────────────────────────────────────────────────────────────────────
PALETA = ['#E63946', "#55CA78", "#B1C219", '#E9C46A', '#F4A261']
def simular_y_animar (boids, pasos = 1000, titulo = "Simulación de Boids", intervalos = 30):
    '''Ejecuta la simulacion y muestra una animacion de los boids
    boids: lista de objetos Boid
    pasos: cantidad de pasos a simular
    intervalo: ms entre cada frame '''

    #Guardar historial
    historial = []
    for _ in range(pasos):

        copia =[Boid(b.pos.copy(), b.vel.copy()) for b in boids] # Copia de la lista de boids para que cada uno vea el estado del grupo antes de actualizarse (actualización sincronica)
        for b in boids:
            b.actualizar(copia)
        historial.append([b.pos.copy() for b in boids])

    fig, ax = plt.subplots(figsize = (7, 7))
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_aspect('equal')
    ax.set_title(titulo, fontsize = 16)
    ax.tick_params(colors='#555')
    for sp in ax.spines.values():
        sp.set_edgecolor('#555')

    ax.add_patch(plt.Rectangle((X_MIN, Y_MIN), X_MAX - X_MIN, Y_MAX - Y_MIN, linewidth = 1.5, edgecolor = '#333'))
    # Estela (últimos K pasos como líneas semitransparentes)
    K_TRAIL = 120
    colors = [PALETA[i % len(PALETA)] for i in range(len(boids))]
    trails = [ax.plot([], [], '-', color=colors[i], alpha=0.25, linewidth=0.8)[0]
              for i in range(len(boids))]
    dots = [ax.plot([], [], 'o', color=colors[i], markersize=6)[0]
            for i in range(len(boids))]
 
    step_text = ax.text(0.02, 0.97, '', transform=ax.transAxes,
                        color='#888', fontsize=8, va='top')
 
    def init():
        for tr in trails:
            tr.set_data([], [])
        for d in dots:
            d.set_data([], [])
        step_text.set_text('')
        return trails + dots + [step_text]
 
    def update(frame):
        inicio = max(0, frame - K_TRAIL)
        for i, d in enumerate(dots):
            pos = historial[frame][i]
            d.set_data([pos[0]], [pos[1]])
            xs = [historial[t][i][0] for t in range(inicio, frame + 1)]
            ys = [historial[t][i][1] for t in range(inicio, frame + 1)]
            trails[i].set_data(xs, ys)
        step_text.set_text(f't = {frame * T:.2f} s')
        return trails + dots + [step_text]
 
    # Saltar frames para que la animación no sea demasiado lenta
    SALTO = 10
    frames_anim = range(0, pasos, SALTO)
 
    anim = FuncAnimation(fig, update, frames=frames_anim,
                         init_func=init, interval=intervalos,
                         blit=True)
    plt.tight_layout()
    plt.show()
    return historial

def vel_aleatoria(angulo_deg=None):
    """Devuelve un vector velocidad de módulo V0.
    Si se pasa angulo_deg se usa ese ángulo; si no, se sortea aleatoriamente."""
    if angulo_deg is None:
        angulo_deg = np.random.uniform(0, 360)
    rad = np.deg2rad(angulo_deg)
    return np.array([V0 * np.cos(rad), V0 * np.sin(rad)])

def caso1():
    """
    Caso 1 – Un boid se encuentra con otro boid.
 
    Dos boids ubicados en extremos opuestos del dominio, moviéndose
    el uno hacia el otro. Al acercarse, la cohesión los atrae y la
    separación evita la colisión; luego se alinean y vuelan juntos.
    """
    boids = [
        Boid([10.0, 25.0], vel_aleatoria(0)),   # → derecha
        Boid([40.0, 25.0], vel_aleatoria(180)),   # → izquierda
    ]
    simular_y_animar(boids, pasos=8000,
                     titulo="Caso 1 – Un boid se encuentra con otro boid")
 
def caso2():
    """
    Caso 2 – Varios boids se encuentran.
 
    Seis boids dispersos por el dominio, cada uno con una dirección
    inicial aleatoria. Eventualmente forman una o varias bandadas.
    """
    np.random.seed(42)
    posiciones = [
        [10.0, 10.0], [40.0, 10.0], [25.0, 25.0],
        [10.0, 40.0], [40.0, 40.0], [25.0,  5.0],
    ]
    boids = [
        Boid(pos, vel_aleatoria())
        for pos in posiciones
    ]
    simular_y_animar(boids, pasos=10000,
                     titulo="Caso 2 – Varios boids se encuentran")
    
def caso3():
    """
    Caso 3 – Un boid solitario se encuentra con un grupo.
 
    Un boid aislado vuela hacia un grupo de cinco boids ya formado
    en el centro del dominio. Se observa cómo el solitario es
    "absorbido" por la bandada.
    """
    # Grupo de 5 boids en el centro, moviéndose hacia la derecha
    grupo_pos  = [[22.0 + dx, 24.0 + dy]
                  for dx, dy in [(-1,1),(0,0),(1,-1),(0,2),(-1,-2)]]
    grupo_vel  = vel_aleatoria(10)   # casi horizontal
 
    boids = [Boid(p, grupo_vel + np.random.randn(2)*0.2)
             for p in grupo_pos]
 
    # Boid solitario a la izquierda, apuntando al grupo
    boids.append(Boid([5.0, 25.0], vel_aleatoria(5)))
 
    simular_y_animar(boids, pasos=10000,
                     titulo="Caso 3 – Un boid solitario se encuentra con un grupo")
def caso4():
    """
    Caso 4 – Dos grupos del mismo tamaño se encuentran
 
    Dos grupos de 5 boids cada uno, moviendose en sentidos opuestos
    Los boids del mismo grupo comienzan muy juntos (separación < RHO_C)
    para que ya formen una bandada cohesionada antes del encuentro
 
    Esperado: interacción compleja — pueden fusionarse en una sola bandada,
    o desviarse y seguir caminos separados dependiendo de las velocidades.
    """
    boids = []
 
    # Grupo A — viene de la izquierda moviéndose hacia la derecha
    # Posiciones apretadas en torno a (12, 25) para que se reconozcan como grupo
    for i in range(5):
        offset = np.array([i % 3, i // 3]) * 0.8   # Desplazamiento compacto (< RHO_C)
        boids.append(Boid(
            pos=[11.0 + offset[0], 24.0 + offset[1]],
            vel=vel_aleatoria(0) + np.random.randn(2) * 0.3   # Casi hacia la derecha
        ))
 
    # Grupo B — viene de la derecha moviéndose hacia la izquierda
    for i in range(5):
        offset = np.array([i % 3, i // 3]) * 0.8
        boids.append(Boid(
            pos=[38.0 + offset[0], 24.0 + offset[1]],
            vel=vel_aleatoria(180) + np.random.randn(2) * 0.3  # Casi hacia la izquierda
        ))
 
    simular_y_animar(boids, pasos=12000,
                     titulo="Caso 4 – Dos grupos del mismo tamaño se encuentran")
 
 
def caso5():
    """
    Caso 5 – Dos grupos de distinto tamaño se encuentran
 
    Grupo grande (8 boids) vs grupo pequeño (3 boids) en trayectorias convergentes
    Esperado: el grupo pequeño puede ser absorbido por el grande, o desviado
    por la fuerza colectiva. El tamaño desbalanceado genera asimetría visible.
    """
    boids = []
 
    # Grupo grande (8 boids) — viene de abajo-izquierda, sube hacia arriba-derecha
    for i in range(8):
        offset = np.array([i % 4, i // 4]) * 0.8   
        boids.append(Boid(
            pos=[8.0 + offset[0], 14.0 + offset[1]],
            vel=vel_aleatoria(45) + np.random.randn(2) * 0.3   
        ))
 
    # Grupo pequeño (3 boids) — viene de arriba-derecha, baja hacia abajo-izquierda
    for i in range(3):
        boids.append(Boid(
            pos=[38.0 + i * 0.8, 36.0],
            vel=vel_aleatoria(225) + np.random.randn(2) * 0.3  
        ))
 
    simular_y_animar(boids, pasos=12000,
                     titulo="Caso 5 – Dos grupos de distinto tamaño se encuentran")
                     
if __name__ == '__main__':
    casos = {
        '1': ('Un boid se encuentra con otro boid',               caso1),
        '2': ('Varios boids se encuentran',                        caso2),
        '3': ('Un boid solitario se encuentra con un grupo',       caso3),
        '4': ('Dos grupos del mismo tamaño se encuentran',        caso4),
        '5': ('Dos grupos de distinto tamaño se encuentran',      caso5),
       
    }

    print(" 0. Ejecutar todos los casos en secuencia")
    for k, (desc, _) in casos.items():
        print(f" {k}. {desc:<49}")
 
    opcion = input("\n Ingresá el número de caso (0–5): ").strip()
 
    if opcion == '0':
        for k, (_, fn) in casos.items():
            print(f"\n Corriendo Caso {k}...")
            fn()
    elif opcion in casos:
        casos[opcion][1]()
    else:
        print("Opción inválida. Ejecutando Caso 1 por defecto.")
        caso1()

    