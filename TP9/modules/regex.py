import re

# Secuencia FASTA de la proteína PML (Dominio PXL / 10th_A)
secuencia_pml = (
    "SNASSESSDLQLEGPSTLRVLDENLADPQAEDRPLVFFDLKIDNETQKISQLAAVNRESK"
    "FRVVIQPEASKAVSLEVGLQHFLSFLSSMRRPILACYKLWGPGLPNFFRALEDINRLWEFQ"
    "EAISGFLAALPLIRERVPGASSFKLKNLAQTYLARNMSERSAMAAVLAMRDLCRLLEVSPG"
    "PQLAQHVYPFSSLQCFASLQPLVQAAVLPRAEARLLALHNVSFMELLSAHRRDRQGGLKKY"
    "SRYLSLQTTTLPPAQPAFNLQALGTYFEGLLEGPALAR"
)

# Diccionario con patrones Regex de interés biológico
# Explicación de sintaxis: [] = conjunto de opciones, [^] = negación, . = cualquier carácter, {n,m} = repeticiones
patrones = {
    "Sitio de N-glicosilación": r"N[^P][ST][^P]",
    "Fosforilación por Proteína Kinasa C (PKC)": r"[ST][KR]",
    "Motivo rico en Prolinas (P...P)": r"P.{1,3}P",
    "Posible motivo de unión/interacción (Leucinas repetidas)": r"L[VI]L",
}

print(f"--- Analizando secuencia PML (Longitud: {len(secuencia_pml)} aa) ---\n")

for nombre_patron, regex_str in patrones.items():
  print(f"Buscando: {nombre_patron} (Patrón: {regex_str})")
  # finditer permite encontrar todas las coincidencias y sus posiciones exactas
  coincidencias = list(re.finditer(regex_str, secuencia_pml))

  if coincidencias:
    for match in coincidencias:
      inicio = match.start() + 1  # Base 1 para lectura biológica
      fin = match.end()
      subseq = match.group()
      print(f"  -> Encontrado '{subseq}' en la posición {inicio}..{fin}")
  else:
    print("  -> No se encontraron coincidencias.")
  print("-" * 40)
