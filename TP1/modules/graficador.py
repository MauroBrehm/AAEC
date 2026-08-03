import matplotlib.pyplot as plt

def graficar_resultados (simulacion, tiempos):
    '''Grafica modelo'''
    plt.figure(figsize=(10, 4))
    plt.plot(tiempos, simulacion, color='pink', label='ECG Simulado')
    plt.title('Evolución temporal del potencial cardíaco (derivación bipolar II)')
    plt.xlabel('Tiempo (ms)')
    plt.ylabel('Potencial total vertical')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

  
