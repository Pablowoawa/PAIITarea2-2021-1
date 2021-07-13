#!/usr/bin/env python
# coding: utf-8

# In[37]:


#Autor: Pablo Sepúlveda Navarrete
#       psepulveda2018@udec.cl
import numpy as np
#!pip install tsplib95
#!pip install networkx
#!pip install pydot
#import pydot 
import networkx as nx
#from networkx.drawing.nx_pydot import graphviz_layout
import matplotlib.pyplot as plt
import tsplib95
import time



entrada = open('entrada.txt','r')
linea = entrada.readline().split()
path = linea[0]
tiempoL = float(linea[1]) #Tiempo límite indicado en la entrada
linea = entrada.readline().split()
coef = np.array(linea, dtype = 'float')
entrada.close()



problem = tsplib95.load(path)



ciudades = list(problem.get_nodes())



def search_node(lista,value):   #Función que busca un valor en una lista específica. Retorna su posición.
    for i in range(len(lista)):
        if lista[i].data == value:
            return i



class Node(): #Estructura principal del problema
    def __init__(self,data,visited=None):
        global n_nodos
        self.data = data #nombre del nodo
        self.children = [] #Lista de hijos del nodo
        self.parent = None #Padre del nodo
        self.visited = visited #Lista de nodos visitados anteriormente
        self.tij = None #Costo de ir desde el padre del nodo al nodo en sí.
        self.FO = 0 #Función objetivo que se irá calculando según la formula
        self.b = 0 #b_i
        self.s = 0 #s_i(b_i)
        n_nodos+=1 #Cada vez que se cree un objeto nodo, se suma 1 a la variable global de nodos

    def add_children(self, child): #Función que agrega un hijo child a un nodo

        child.parent = self #El padre de child es self
        child.visited = child.parent.visited.copy() 
        child.visited.append(child.data) #Se actualiza la lista de nodos visitados

        edge = child.parent.data,child.data 
                                                #Se calcula el costo explicado en el constructor de Node
        child.tij = problem.get_weight(*edge)

        child.b = self.b + self.s + child.tij   #Se calcula b segun la formula
        child.s = coef[0]*child.b**2 + coef[1]*child.b + coef[2] #Se calcula s segun la formula

        child.FO = self.FO + child.tij + child.s #Se acumula la función objetivo

        self.children.append(child) #Se agrega el hijo a la lista de hijos del padre
    

    def print_tree(self):    #Función que imprime el árbol de búsqueda, con nodo, b,s y FO
        spaces = ' ' * self.get_level()*3
        prefix = spaces + '|__' if self.parent else ''
        print(prefix + str(self.data))
        if self.children:
            for child in self.children:
                child.print_tree()

    def get_level(self): #Función que retorna el nivel que se encuentra un nodo
        level = 0
        p = self.parent
        while p:
            level += 1
            p = p.parent
        
        return level


    def create_level(self,ciudades): #Función que crea un nivel completo de hijos
        for elem in ciudades:         #Por ejemplo, si el nodo es 0, y el tamaño del problema es 17, se le crean
                                        #n-1 hijos del 1 al 16
            if elem not in self.visited:
                self.add_children(Node(elem))


    def create_full_tree(self, ciudades): #Función que crea el arbol completo al problema asignado.
                                          #Sin embargo no se usa nunca por el tiempo computacional.
        self.create_level(ciudades)
        for child in self.children:
            child.create_full_tree(ciudades)

    def get_minimum(self):             #Función que retorna el hijo que tiene el menor costo dado por FO.
        best = 999999999999
        index = 0
        for i in range(len(self.children)):
            aux = self.children[i].FO
            if aux<best:
                best = aux
                index = i
        if self.children:
            pick = self.children[index]
            return pick

        
    def vecinoMasCercano(self,ciudades):     #Heurística del vecino más cercano.
        global n_nodos
        global CS
        global ruta_inicial
        
        tiempoInicial = time.time()
        self.create_level(ciudades)       #Se crea un nivel completo de hijos
        pick = self.get_minimum()          #Se busca el hijo que tiene el menor costo
        if pick:
            pick.vecinoMasCercano(ciudades)  #Se usa la recursividad y se vuelve a aplicar la función.
            
        else:
            edge = self.data,  ciudades[0]    #Finalmente se le agrega el costo de volver al nodo inicial (depósito)
            self.FO += problem.get_weight(*edge)
            print('La función objetivo es: '+str(self.FO))  
            CS = self.FO                #Se guarda el costo en una variable que servirá como Cota superior en el B&B
            ruta_inicial = self.visited  #Se guarda la ruta del costo encontrado por la heurística.
            ruta_inicial.append(ciudades[0])
            print(ruta_inicial)
            tiempoFinal = time.time()

    def is_leaf(self,ciudades):    #Función que retorna True si el nodo preguntado es una hoja.
        if len(self.visited)==len(ciudades):
            return True
        return False #False si no lo es.

    def sort_costs(self):    #Función que retorna y ordena una lista de hijos según su FO almacenada.
        l = sorted(self.children, key=lambda v: v.FO)
        return l


    def branchAndBound(self,cota,tiempoLimite,tiempoh):  #Algoritmo de Branch and Bound
        global best_solution
        global ruta_inicial
        global tiempo
        flag = 0
        print('Buscando nuevas soluciones')

        
        self.create_level(ciudades)
        ordenados = self.sort_costs() #Ordenados es la lista en orden de menor a mayor de ciudades con menor costo          
           
        for j in range(len(ordenados)):#Por cada nodo en ordenados, realiza el branch and bound.
            hijos = self.children.copy()#hijos es una lista que contiene a todos los hijos de la raiz
            i = search_node(hijos, ordenados[j].data)#como forma de mejorar el algoritmo, se propone empezar por la
                                                    #primera ciudad con menor costo
            print('Trabajando en el nodo', hijos[i].data)
            while flag==0:    
                child = hijos[i] #se empieza por el primer hijo
                
                #print('data',child.data,'FO',child.FO)
                if child.FO < cota: #Se pregunta si su FO es mejor que la cota actual
                    if child.is_leaf(ciudades): #Si el nodo es una hoja
                        edge = child.data , ciudades[0]
                        child.FO += problem.get_weight(*edge)#Se le suma el costo de volver al depósito

                        if child.FO < cota:
                            cota = child.FO #Se actualiza la cota
                            best_solution = cota
                            print('nueva cota', cota)



                            child.visited.append(ciudades[0])
                            ruta_inicial = child.visited.copy()
                            print(child.visited)

                    else: #Si no es una hoja, se le crea un nivel a ese nodo
                        i=0 #El indice vuelve a la primera posicion
                        child.create_level(ciudades)
                        hijos = child.children.copy() #Se actualiza la lista de hijos, a los hijos de este nuevo nodo

                else: #Si la FO del nodo es mayor o igual a la cota
                    i+=1 #Se recorren los hijos, buscando posibles nuevas soluciones
                    while i==len(hijos):#En este bloque de código se retorna al nivel anterior, buscando posibles
                                            #nuevas soluciones
                        i -=1
                        child = hijos[i]
                        if child.parent.parent:
                            hijos = child.parent.parent.children.copy()
                            i= search_node(hijos,child.parent.data)
                        else:
                            flag=1 #Se activa una bandera para terminar el algoritmo
                            break

                        i+=1


                tiempoFinal = time.time()

                tiempo = tiempoFinal - tiempoh #Verificador de tiempo
                if tiempo>=tiempoLimite:
                    #print('Fin del tiempo',tiempo,'segundos')
                    break
                
                    
            if tiempo>=tiempoLimite:
                print('Fin del tiempo',tiempo,'segundos')
                break
                





#MAIN

if __name__=='__main__':
    tiempo = 0
    CS = 999999999
    n_nodos = 0
    

    tiempoheuristica = time.time() #inicia contador de tiempo
    root = Node(ciudades[0],visited = [ciudades[0]]) #Se crea la raíz para el algoritmo del vecino mas cercano
    root.vecinoMasCercano(ciudades) #Se aplica la heurística del vecino más cercano.
    best_solution = CS #Se guarda la cota encontrada en el vecino mas cercano
    

    tree = Node(ciudades[0],visited=[ciudades[0]]) #Se crea la raiz del arbol de busqueda
    tree.branchAndBound(CS,tiempoL,tiempoheuristica) #Se aplica el Branch and bound y empieza a iterar.
    
    #Escritura en salida.txt
    salida = open('salida.txt','w')
    salida.writelines('Nodos: '+str(n_nodos))
    salida.writelines('\n')
    salida.writelines('Tiempo: '+str(tiempo))
    salida.writelines('\n')
    salida.writelines('FO: '+str(best_solution))
    salida.writelines('\n')
    salida.writelines('Ruta: ')
    salida.writelines([str(ruta_inicial[i])+' ' for i in range(len(ruta_inicial))])
    salida.close()



    #Gráfico mejor solución encontrada

    best_solution_plot = nx.DiGraph()

    aristas = [[ruta_inicial[i],ruta_inicial[i+1]] for i in range(len(ruta_inicial)-1)]
    aristas.append([ruta_inicial[-1],ciudades[0]])
    best_solution_plot.add_edges_from(aristas)
    pos = nx.circular_layout(best_solution_plot)
    nx.draw_networkx(best_solution_plot,pos)
    print('La mejor solución fue', best_solution)
    plt.show()



    #Visualización del árbol de búsqueda
    print('#############################################################')
    tree.print_tree() #Si son muchos nodos se va a demorar mucho en imprimir la visualizacion,
                        #probar con tiempos muy pequeños.


# In[ ]:




