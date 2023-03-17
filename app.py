#!flask/usr/bin/python3
from flask import Flask, render_template, request
from PageRank import PageRank
import pandas as pd
app = Flask(__name__)
app.config.from_pyfile('config.py')

@app.route('/', methods = ['GET'])
def main_screen():
    return render_template("index.html")

@app.route('/pagerank', methods=["GET"])
def network_input():
    return render_template("network.html")


@app.route('/pagerankresult', methods = ['POST'])
def custom_recommendation_list():
    beta=0.85
    epsilon=0.000001
    max_iterations=20
    pr=PageRank(beta,epsilon,max_iterations)        
    #file_matrix="adj_matrix.txt"
    file = request.files["file"]
    #adjacency_matrix=pagerank.input_array()
    #pr = PageRank(beta, epsilon, max_iterations, node_num)
    #adjacency_matrix=pagerank.create_adj(file_matrix)
    #adjacency_matrix=pagerank.create_adj(file_matrix)
    # adjacency_matrix=pagerank.getXML(file_xml)
    # print("============ adjacency_matrix==========")
    # print(adjacency_matrix)
    # adjacency_matrix = np.array([[0., 1., 0., 0., 0., 1.],
    #                            [0., 0., 1., 0., 0., 1.],
    #                            [0., 0., 0., 0., 1., 0.],
    #                            [0., 1., 0., 0., 1., 0.],
    #                            [0., 0., 0., 0., 0., 0.],
    #                            [0., 0., 1., 1., 0., 0.]])
    #file_matrixXML="WebPageCours.xml"
    #adjacency_matrix_X=pagerank.getXML(file_matrixXML)
    #print(p)

    # adjacency_matrix_T=pagerank.Transpose_Matrix(adjacency_matrix)
    # print("============ adjacency_matrix_T==========")
    # print(adjacency_matrix_T)

    # Convert_to_markov=pagerank.Spars_Matrix(adjacency_matrix_T)
    # print("============ matrix_Markov==========")
    # print(Convert_to_markov)
    # Sparse_matrix=pagerank.Spars_Matrix(Convert_to_markov)

    # node_num=adjacency_matrix_T.shape[0]
    # network_graph = pagerank.build_graph_from_file(Sparse_matrix)
    # pr = PageRank(beta,epsilon , network_graph,  max_iterations, node_num)
    #df_links,df_dang,matrice_Tele,matrice_Trans,Rsurf_Prob=pr.Allin()
    PageRank_vector,df_links,df_dang,matrice_adj,matrice_Tele,matrice_Trans,Rsurf_Prob = pr.pagerank(beta,epsilon,max_iterations,file.filename)
    sorted_x = sorted(PageRank_vector.items(), key=lambda x: x[1], reverse=True)

    
    f = open('rank.csv', 'w')
    nodes = ['A', 'B', 'C','D','E','F','G','H','I','L','M']
    dict_nodes={k: v for v, k in enumerate(nodes)}
    df = pd.DataFrame(columns=['Lien', 'Score'])
    for name in sorted_x:
        node=list(dict_nodes.keys())[list(dict_nodes.values()).index(int(name[0]))]
        f.write("%s,%s\n" % (node, name[1]*100))
        df = pd.concat([pd.DataFrame([[node,name[1]*100]], columns=df.columns), df], ignore_index=True)
        df = df.sort_values(by=['Score'], ascending=False)
    f.close()
    
    return render_template('result.html',pageranks=list(df.values.tolist()), links=list(df_links.values.tolist()),dangs=list(df_dang.values.tolist()),madj=matrice_adj,mtl=matrice_Tele,mtr=matrice_Trans,rsp=Rsurf_Prob)

