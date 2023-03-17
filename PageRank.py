import numpy as np
from scipy import sparse
import xml.dom.minidom
import math
import os
import re
import pandas as pd
from graph import plotGraph
from graph import Graph
from multipledispatch import dispatch


class PageRank:

    def __init__(self, beta,epsilon, max_iterations,graph=None, node_num=None):
        if graph is None:
            self.epsilon=epsilon
            self.beta=beta
            self.max_iterations = max_iterations
        else:
            self.beta = beta
            self.graph=graph
            self.edges = graph.get_connections()
            self.epsilon = epsilon
            self.node_num = node_num
            self.max_iterations = max_iterations

    # prend une entrée
#     def input_array(self):
#         row = int(input("Enter the Number of rows:"))
#         column = int(input("Enter the Number of columns:"))
#         data = []
#         for c in range(column):
#             row_list = []
#             for r in range(row):
#                 row_list.append(float(input("Enter value for i={}&j{} = ".format(c, r))))
#             data.append(row_list)
#         data_array = np.array(data)
#         return data_array
    
    def getXML(self,F):
        doc=xml.dom.minidom.parse(F)
        pagesList=[]
        totalLinks=[]
        pages=doc.getElementsByTagName('page')
        for page in pages:
            listeLinks=[]
            pagesList.append(page.getAttribute("PageName"))
            Links=page.getElementsByTagName("link")
            for l in range(len(Links)):
                listeLinks.append(Links[l].firstChild.nodeValue)  
            totalLinks.append(listeLinks)
        print("===============================")
        for i in range(len(pagesList)):
            print(pagesList[i]," has links to: ",totalLinks[i])
    
        print("==================Adjacence==============")
        matrice=np.zeros((len(pagesList),len(pagesList)))
        for i in range(len(pagesList)):
            for j in range(len(pagesList)):
                 if pagesList[j] in totalLinks[i]:
                        if len(totalLinks[i]) > 0 :
                            matrice[i][j]=len(totalLinks[i])/len(totalLinks[i])
                        else :
                            matrice[i][j]=0
        return matrice
    
    def create_adj(self,F):
        #F = input("Adjacency matrix txt file: ")
        global size, adj
        adj = []
        with open(F) as file:
            for line in file:
                adj.append([int(i) for i in line.strip('').split('\t')])
        size = adj[0][0]
        #adj.pop(0)
        adj_array = np.array(adj)
        return adj_array
        
    def Transpose_Matrix(self,matrix):
        return matrix.T

        #Convertir la matrice d'adjacence en une matrice de Markov
        #pour que la somme des colonnes devienne 1    
    def ConvertMatrix_To_Markov(self,matrix):
        M = np.zeros(matrix.shape)
        for c in range(matrix.shape[1]):
            s = matrix[:, c].sum()
            if(s!=0):
                M[:, c] = matrix[:, c]/s
        return M
    
# Conserve uniquement les nœuds non nuls
    def Spars_Matrix(self,matrix):
        return sparse.csc_matrix(matrix)
    
    def build_graph_from_file(self,matrix):
        dirpath = os.getcwd()
        file_object = "temp.txt"
        lignes=[]
        print(re.sub('[\[\]]', '', np.array_str(matrix)))
        with open(os.path.join(dirpath,file_object), 'w') as out_file:
            out_file.writelines(str(matrix))
        out_file.close();
        with open (os.path.join(dirpath,file_object), 'r') as in_file:
            lines = in_file.readlines()
            i=0
            for i in range(0,len(lines)):
                linestmp=re.sub('^ \W+|\)\t[0-9]+[.]*[0-9]*[\n]*$', '', lines[i])
                linestmp=re.sub('[\[]+[ ]*|[\]]+[\n]*$', '', linestmp)
                linestmp=re.sub(', ', '\t', linestmp)
                linestmp=re.sub('[ ]+', '\t', linestmp)
                print(linestmp.split('\t'))
                if(isinstance(matrix,sparse.csc.csc_matrix)):
                    to_, from_ = linestmp.split('\t')
                else :
                    from_ ,to_  = linestmp.split('\t')
                linestmp=('\t').join([from_,to_])
                lignes.append(linestmp)
                lignes.append("\n")
        out_file.close(); 
        with open (os.path.join(dirpath,file_object), 'w') as in_file:
            in_file.writelines(lignes)
        out_file.close(); 
        graph2 = Graph(file_object)
        return graph2
    
    def build_graph_from_fileXML(self,filepath):
        dirpath = os.getcwd()
        doc=xml.dom.minidom.parse(str(os.path.join(dirpath,filepath)))
        nodesList=[]
        totalLinks=[]
        nodes=doc.getElementsByTagName('page')
        for node in nodes:
            listeLinks=[]
            nodesList.append(node.getAttribute("PageName"))
            Links=node.getElementsByTagName("link")
            for l in range(len(Links)):
                listeLinks.append(Links[l].firstChild.nodeValue)  
            totalLinks.append(listeLinks)
        return nodesList,totalLinks
    
    def Allin(self):
        nodes=self.graph.nodes() 
        Links=self.edges
        #print("Nodes",nodes)
        #print("Links",Links)
        df_links = pd.DataFrame(columns=['Noeud','', 'has links to:'])
        print("===============================")
        for i in range(len(nodes)):
            print(nodes[i]," has links to: ",Links[i])
            df_links = pd.concat([pd.DataFrame([[nodes[i]," has links to: ",Links[i]]], columns=df_links.columns), df_links], ignore_index=True)
        print("==================Adjacence==============")
        matrice_Trans=np.zeros((len(nodes),len(nodes)))
        matrice_adj=np.zeros((len(nodes),len(nodes)))
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                if nodes[j] in str(Links[i]):
                    matrice_Trans[i][j]=0.85*1/len(Links[i])
                    if len(Links[i]) > 0 :
                        matrice_adj[i][j]=len(Links[i])/len(Links[i])
                    else :
                        matrice_adj[i][j]=0                 
        print(matrice_adj)
        print("==================Transportation==============")
        print(matrice_Trans)
        print("==================== Dangling Nodes==================== ")
        dangling,indexdang=self.isdangling()
        df_dang=pd.DataFrame()
        for i in range(len(dangling)):
            print("dangling node is : ",dangling[i],"his index is :",indexdang[i])
            df_dang = pd.concat([pd.DataFrame([["dangling node is : ",dangling[i],"his index is :",indexdang[i]]]), df_dang], ignore_index=True)

        nbr_nodes=len(nodes)
        matrice_Tele = np.zeros((nbr_nodes,nbr_nodes))
        Ex1 = np.zeros((nbr_nodes,nbr_nodes))
        matrice_Tele[:] = 0.15/nbr_nodes
        Ex1[:] = 1/nbr_nodes
        for i in indexdang:
            matrice_Tele[i,:]=Ex1[i,:]
        print("====================Teleportation ====================")
        print(matrice_Tele)
        print("==================== Random Surfing Probability==========")
        Rsurf_Prob=matrice_Tele+matrice_Trans
        print(Rsurf_Prob)
        return df_links,df_dang,matrice_adj,matrice_Tele,matrice_Trans,Rsurf_Prob

#     def PageRank_Function(self,M_Matrix,V_Matrix,EN_Matrix):
        #implementation originale
#         #  Valeurs initiales
#         finish_num = 1
#         beta = self.beta
#         alpha = self.alpha
        
#         while(finish_num>alpha):
#             #print("-----------------------------------------------------")
#             #print(beta * (M_Matrix * V_Matrix))
#             V_Rank = beta * (M_Matrix * V_Matrix) + (1-beta) * EN_Matrix
#             #print(V_Rank)
            
#         #essaie d'atteindre 1 noeud à chaque étape            
#             rescale = 1 / V_Rank.sum()
#             V_Rank = V_Rank * rescale
#         # Soustrait la nouvelle matrice de la précédente pour créer la condition de terminaison 
#             v_d = abs(V_Rank - V_Matrix)
#             finish_num = v_d.sum()
#          # Mettre à jour le classement de la page         
#             V_Matrix = V_Rank
#         df_all=V_Rank
#         listall_df=list(map(float, df_all))
#         #2 donne la première page
#         return listall_df
    
    def isdangling(self):
        nodes=self.graph.nodes() 
        Links=self.edges
        dangling=[]
        index=[]
        for i in range(len(nodes)):
            if not Links[i] :
                dangling.append(nodes[i])
                index.append(i)
        return dangling,index
    
    def isMatAdj(self,mat) :
        numbers=np.sum(np. unique(mat))
        return numbers==1 & (mat.shape[0]== mat.shape[1])

    # implementation 0
    @dispatch(float,float,int, np.ndarray)   
    def pagerank(self,beta,epsilon,max_iterations,matrix):
        if (self.isMatAdj(matrix)): 
            adjacency_matrix_T=self.Transpose_Matrix(matrix)
            self.node_num=adjacency_matrix_T.shape[0]
            Convert_to_markov=self.Spars_Matrix(adjacency_matrix_T)
            Sparse_matrix=self.Spars_Matrix(Convert_to_markov)
            matrix=Sparse_matrix
        network_graph = self.build_graph_from_file(matrix)
        self.graph=network_graph
        self.edges = self.graph.get_connections()
        nodes = self.graph.nodes()
        self.node_num=len(nodes)
        graph_size = len(nodes)
        print('graph_size',graph_size)
        if graph_size == 0:
            return {}

        for node in self.graph.nodes():
            if len(self.graph.neighbors(node)) == 0:
                for node2 in self.graph.nodes():
                    self.graph.add_edge((node, node2))

        # initialize the page rank dict with 1/N for all nodes
        final_rank_vector = np.zeros(self.node_num)
        initial_rank_vector = np.fromiter([1 / self.node_num for _ in range(self.node_num)], dtype='float')

        pagerank = dict.fromkeys(nodes, 1.0 / graph_size)
        pg = plotGraph(self.edges, interval=3000)
        iterations = 0
        diff = math.inf
        print('self.max_iterations',self.max_iterations)
        for i in range(self.max_iterations):
        #TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT
            new_rank_vector = np.zeros(self.node_num)
            for parent in self.edges:
                for child in self.edges[parent]:
                    new_rank_vector[child] += (initial_rank_vector[parent] /
                    len(self.edges[parent]))
            leaked_rank = (1-sum(new_rank_vector))/self.node_num
            final_rank_vector = new_rank_vector + leaked_rank
            initial_rank_vector = final_rank_vector
            #TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT
            diff = 0  # total difference compared to last iteraction
            # computes each node PageRank based on inbound links
            for node in nodes:
                rank = (1.0 - self.beta) / graph_size
                for referring_page in self.graph.incidents(node):
                    rank += self.beta * pagerank[referring_page] / len(self.graph.neighbors(referring_page))

                diff += abs(pagerank[node] - rank)
                pagerank[node] = rank
                iterations += 1
            # stop if PageRank has converged
            if diff < self.epsilon:
                break
        pg.plot(graph_size, pagerank)
        df_links,df_dang,matrice_adj,matrice_Tele,matrice_Trans,Rsurf_Prob=self.Allin()
        return pagerank,df_links,df_dang,matrice_adj,matrice_Tele,matrice_Trans,Rsurf_Prob
    
    @dispatch(float,float,int, str)
    def pagerank(self,beta,epsilon,max_iterations,file_matrix):
        pagerank={}
        #if (self.isMatAdj(file_matrix)):  #fichier contenant matrice
        if file_matrix.endswith('.xml'):
            adjacency_matrix=self.getXML(file_matrix)
        elif file_matrix.endswith('.txt'):
            adjacency_matrix=self.create_adj(file_matrix)
        pagerank=self.pagerank(beta,epsilon,max_iterations,adjacency_matrix)
        return pagerank
    
