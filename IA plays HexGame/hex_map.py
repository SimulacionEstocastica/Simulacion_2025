import numpy as np

class HEX_MAP:
	def __init__(self, length:int, tipo:str="diamond"):
		"""
		inputs
		length : representa la longitud del mapa jugable
		tipo : representa el tipo de mapa (diamond, hexagon, triangle) (de momento solo diamante)
		"""
		assert length >= 1
		self.tipo = tipo # para implementar otras formas hay que modificar la forma en la que se hace el display


		self.size = length
		self.tablero = np.zeros((length+2, length+2), dtype=int)									# tablero cuadrado para la vizualización
		self.adyacencia = {(i,j):[] for i in range(length+2) for j in range(length+2)} 	# lista de adyacencia

		self.vecinos = [    np.array((-1,0),dtype=int),		np.array((-1,1),dtype=int),
		np.array((0, -1),dtype=int),							np.array((0,1),dtype=int), 
		np.array((1, -1),dtype=int), np.array((1,0),dtype=int)]

		self.tablero[0,...] = -1
		self.tablero[length+1,...] = -1
		self.tablero[...,0] = 1
		self.tablero[...,length+1] = 1

		# esquinas
		self.adyacencia[0,1].append(np.array([0,2]))
		self.adyacencia[0,length].append(np.array([0,length-1]))
	
		self.adyacencia[length+1,1].append(np.array([length+1,2]))
		self.adyacencia[length+1,length].append(np.array([length+1,length-1]))


		self.adyacencia[1,0].append(np.array([2,0]))
		self.adyacencia[length,0].append(np.array([length-1,0]))
	
		self.adyacencia[1,length+1].append(np.array([2,length+1]))
		self.adyacencia[length,length+1].append(np.array([length-1,length+1]))

		for i in range(2,length):
			# adyacencia de los vertices superiores
			self.adyacencia[0,i].append(np.array([0,i+1]))
			self.adyacencia[0,i].append(np.array([0,i-1]))

			# adyacencia inferior
			self.adyacencia[length+1,i].append(np.array([length+1,i+1]))
			self.adyacencia[length+1,i].append(np.array([length+1,i-1]))

			# adyacencia izquierda
			self.adyacencia[i,0].append(np.array([i+1,0]))
			self.adyacencia[i,0].append(np.array([i-1,0]))

			# adyacencia derecha
			self.adyacencia[i,length+1].append(np.array([i+1,length+1]))
			self.adyacencia[i,length+1].append(np.array([i-1,length+1]))

	def __repr__(self):
		'''
		Esto indica el comportamiento de print(HEX_MAP)
		'''
		blue = '\033[94m'
		red = '\033[91m'
		gray = "\033[1;30m"
		end = '\033[0m'
		text = ""
		clr = [blue, gray, red]
		shp = ['X', '▪', 'O']
		for fila in range(self.size+2):
			text += ' '*fila+' '.join([f"{clr[i+1]}{shp[i+1]}{end}" for i in self.tablero[fila]]) + '\n'		
		return text
	
	def jugada(self, col:int, pos):
		if pos.size != 2 or pos.dtype != int:
			return
		
		if (pos[0] == 0 or pos[0] == self.size+1) or (pos[1] == 0 or pos[1] == self.size+1):
			return
		
		if self.tablero[*pos] != 0: # esto también garantiza que no exista la arista
			return
		


		self.tablero[*pos] = col
		for dir in self.vecinos:
			if self.tablero[*(pos + dir)] == col:
				self.adyacencia[*pos].append(pos + dir)
				self.adyacencia[*(pos+dir)].append(pos)

	def check(self, col: int) -> bool:
		'''
		Aquí implementar el chequeo si es que se puede llegar de un lado de un color a otro
		como self contiene la lista de adyacencia se puede implementar DFS de forma directa
		desde el centro del lado correspondiente al color y después ver si es que el centro
		del otro lado es alcanzable
		'''
		
		match col:
			case -1:
				start 	= np.array((0,1), dtype=int)
				end 	= np.array((self.size+1, 1), dtype=int)
			case 1:
				start 	= np.array((1,0), dtype=int)
				end 	= np.array((1,self.size+1), dtype=int)
			case _:
				return False

		visitados = np.zeros((self.size+2, self.size+2), dtype=bool)

		stack = self.adyacencia[*start].copy()
		#print(stack)
		while len(stack) != 0:
			current = stack.pop()
			if visitados[*current] == True:
				continue
			
			visitados[*current] = True
			stack += self.adyacencia[*current].copy()

		return visitados[*end] # type: ignore

"""# Ejemplo de uso con tu clase HEX_MAP
a = HEX_MAP(5)  # Tablero 5x5
a.jugada(1, np.array([1,1]))
# Obtener todas las posiciones vacías (valor 0)
# Si tu clase HEX_MAP tiene el tamaño real
# Las claves del diccionario de adyacencia SON todas las posiciones existentes
posiciones_vacias = []
for pos in a.adyacencia.keys():
    i, j = pos
    if a.tablero[i][j] == 0:  # Verificar si está vacía
        posiciones_vacias.append(pos)

print(f"Posiciones vacías: {posiciones_vacias}")
#print(a.adyacencia)
print(a.tablero)
print(type(a.tablero))"""