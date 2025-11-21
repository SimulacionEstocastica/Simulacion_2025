import random as rd
import numpy as np

class BaseBRW:
    def __init__(self, p=0.5):
        self._particulas = []
        self.p = p

    @property
    def particulas(self):
        return np.asarray(self._particulas.copy())
    

    @particulas.setter
    def particulas(self, value):
        self._particulas = [list(p) for p in value]

    @property
    def vacio(self):
        return len(self._particulas) == 0

    def crear_particula(self, x=0, y=0):
        self._particulas.append([x, y])

    def mover(self):
        "Ecoge aleatoriamente una particula y la mueve"
        n = len(self._particulas)
        i = rd.randint(0, n-1)
        dx, dy = rd.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
        self._particulas[i][0] += dx
        self._particulas[i][1] += dy

    def mover(self, i):
        """Mueve aleatoriamente la i-esima particula"""
        try:
            dx, dy = rd.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
            self._particulas[i][0] += dx
            self._particulas[i][1] += dy
        except: Exception("Se intenta mover una particula que no existe")

    def aparear(self): # Añadir la capacidad de morir!
        n = len(self._particulas)
        i = rd.randint(0, n-1)
        if self.p == 0:
            self.mover(i)
            return
        if rd.uniform(0.0, 1.0) < self.p: # Nos morimos
           self._particulas.pop(i) 
        else:               # else Nos duplicamos y nos movemos
            self._particulas.append(self._particulas[i].copy())
            self.mover(i)
            self.mover(n) # ASi solo hay que modificar o implementar mover en las clases hijo


    # TODO IMplementar que mienstras mas particulas hay
    # más probable es que se aparear()
    def actualizar(self):
        """Actualiza el moviemiento de una particula"""
        if self.vacio:
            raise Exception("Se intenta actualizar un RW vacío") 
        self.aparear()
        
        """
        self.mover()
        # Mover OR aparear es bueno porque 
        # las particulas se pueden morir con cualquiera de los dos
        if (not self.vacio) and rd.random() < self.p:
            self.aparear()"""
        



class BRW(BaseBRW):
    """ Branching Random Walk estándar """
    pass


class BRW_IDLA(BaseBRW):
    """ Branching RW + IDLA """
    def __init__(self, p=0.5):
        super().__init__(p)
        self._mapa = [[0, 0]]  # origen ocupado

    @property
    def mapa(self):
        return np.asarray(self._mapa.copy())

    @mapa.setter
    def mapa(self, value):
        self._mapa = [list(v) for v in value]

    @property
    def N(self):
        '''Tamaño del cluster'''
        return len(self._mapa)

    
    def mover(self):
        n = len(self._particulas)
        i = rd.randint(0, n-1)
        dx, dy = rd.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
        x_new = self._particulas[i][0] + dx
        y_new = self._particulas[i][1] + dy

        if [x_new, y_new] not in self._mapa:
            self._mapa.append([x_new, y_new])
            self._particulas.pop(i)
        else:
            self._particulas[i] = [x_new, y_new]

    def mover(self, i):
        """Mueve aleatoriamente la i-esima particula,
        considera la restricción del cluster """
        try:
            dx, dy = rd.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
            x_new = self._particulas[i][0] + dx
            y_new = self._particulas[i][1] + dy
            if [x_new, y_new] not in self._mapa:
                self._mapa.append([x_new, y_new])
                self._particulas.pop(i)
            else:
                self._particulas[i] = [x_new, y_new]
        except: Exception("Se intenta mover una particula que no existe")
        
    def obtener_parametros(self):
        """Retorna una aproximación de los parámetros a y b
        correspondientes a la anchura y altura "radial" del cluster"""
        #Parametros
        Aizq = 0
        Ader = 0
        Bup = 0
        Bdown = 0
        for u in self.mapa:
            if u[0] < Aizq: # Coordenada x?
                Aizq = u[0] 
            if u[0] > Ader:
                Ader = u[0]
            if u[1] < Bdown: # Coordenada y ?!
                Bdown = u[1]
            if u[1] > Bup:
                Bup = u[1]
        A = (abs(Aizq) + abs(Ader))/2
        B = (abs(Bup) + abs(Bdown))/2
        # N = \pi * a * b * S^2
        # a = N / (\pi * b * S^2)
        return A, B



class BRW_IDLA_PERC(BRW_IDLA):
    """ Branching RW + IDLA + Percolación """
    def __init__(self, p=0.5):
        super().__init__(p)
        self.perc = set()

    def crear_perc(self, N, pvert, phor):
        """Crea una perecolación de tmaño 2Nx2N CUIDADO
        forzará la creación de una perc. VÁLIDA. iterará hasta conseguirla
        parámetros altos de prob. de descarte puede hacer que itere por siempre (c.s)
        """
        print("Generando Percolación")
        self.perc = set()
        # SOLUCION
        # almacenar arista nomas como {[(1,2),(2,2)]} pero implementar funcion que haga el chequeo como (uv in E) or (vu in E)
       

        # la idea es partir desde (-N,-N) (hacemos la grilla de 2Nx2N)
        # y avanzamos calculando la "L" de aristas adyacentes que se quedan
        # LLLLL
        # LLLLL
        # LLLLL
        for i in range(-N,N):
            # Desde (-N,-N) a (N,-N)        # (-N,i) a (N,i)
            for j in range(-N,N):
                
                actual = (i,j)
                # quietar Arista Horizontal
                if rd.random() < phor: # quitamos la arista (añadirla a perc)
                    self.perc.add( ((i,j), (i,j+1)) )
                # quitar arista vertical
                if rd.random() < pvert:
                    self.perc.add( ((i,j), (i+1,j)) )                

        print("Validando percolacion ")
        if not self.validar_percolacion(N): #esto esta loco
            print("Percolacion inválida, reintentando...")
            self.crear_perc(N, pvert, phor) #esto esta loco

        pass

    def validar_percolacion(self, N):
        """
        Retorna True si es posible escapar desde (0,0)
        a cualquier punto del borde [-N,N]^2 usando solo aristas NO bloqueadas.
        Hecho con ia en base a la mi implementacion propia
        """
        
        from collections import deque

        # movimientos posibles: N, S, E, O
        vecinos = [(1,0), (-1,0), (0,1), (0,-1)]

        origen = (0,0)

        # si por alguna razón el origen ya está fuera, trivial
        if not (-N <= 0 <= N):
            return False

        visitado = set([origen])
        q = deque([origen])

        # función auxiliar para revisar si una arista está bloqueada
        def arista_bloqueada(a, b):
            return (a, b) in self.perc or (b, a) in self.perc

        while q:
            x, y = q.popleft()

            # si está en el borde, escapamos
            if x == -N or x == N or y == -N or y == N:
                return True

            # explorar vecinos
            for dx, dy in vecinos:
                nx, ny = x + dx, y + dy

                # fuera del dominio: no interesa
                if not (-N <= nx <= N and -N <= ny <= N):
                    continue

                # revisar si la arista está bloqueada
                if arista_bloqueada((x,y), (nx,ny)):
                    continue

                # BFS normal
                if (nx, ny) not in visitado:
                    visitado.add((nx,ny))
                    q.append((nx,ny))

        # si terminamos de explorar sin llegar al borde, no se puede escapar
        return False

    def mover(self):
        '''Fuerza un movimiento válido '''
        # TODO: incluir validación contra percolación 
        # se forzará a que una particula 
        n = len(self._particulas)
        i = rd.randint(0, n-1)
        x = self._particulas[i][0]
        y = self._particulas[i][1]
        while True:
            dx, dy = rd.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
            x_new = x + dx
            y_new = y + dy

            if ((x_new,y_new), (x,y)) in self.perc or ((x,y), (x_new,y_new)) in self.perc: 
                continue
        
            if [x_new, y_new] not in self._mapa:
                self._mapa.append([x_new, y_new])
                self._particulas.pop(i)
                
            else:
                self._particulas[i] = [x_new, y_new]
            return
        
    def mover(self, i):
        """Mueve aleatoriamente la i-esima particula,
        considera la restricción del cluster
         y de la percolación generada """
        try:
            x = self._particulas[i][0]
            y = self._particulas[i][1]
            while True:
                # Proponemos movernos a un lugar
                dx, dy = rd.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
                x_new = x + dx
                y_new = y + dy

                # Vemos si este movimiento esta prohibido por la percolacion
                if ((x_new,y_new), (x,y)) in self.perc or ((x,y), (x_new,y_new)) in self.perc: 
                    continue
                
                # Si se permite el movimiento hacemos el movimiento estandar de IDLA
                if [x_new, y_new] not in self._mapa:
                    self._mapa.append([x_new, y_new])
                    self._particulas.pop(i)
                    
                else:
                    self._particulas[i] = [x_new, y_new]
                return
        except: Exception("Se intenta mover una particula que no existe")
